//! End-to-end audio→RTTM offline diarization.
//!
//! `OwnedDiarizationPipeline` is the speakrs-comparable batch
//! entrypoint: take owned 16 kHz mono samples, run segmentation +
//! embedding ONNX inference internally, project through PLDA, run
//! `cluster_vbx`, reconstruct frame-level diarization, and return
//! spans / RTTM. Pyannote `community-1` algorithm.
//!
//! ## Status
//!
//! MVP. End-to-end orchestration works on the captured fixtures.
//! Cross-chunk speaker permutation alignment is *not* performed —
//! `assign_embeddings` (AHC) handles cross-chunk pairing
//! algorithmically via embedding similarity, so the slot ordering
//! within each chunk being arbitrary doesn't break the pipeline.
//! However, the per-output-frame `count` aggregation uses simple
//! averaging-then-binarize across covering chunks, *not* pyannote's
//! PIT-permutation-aware aggregation. This is a known divergence
//! that affects the discrete diarization grid (and thus the
//! reconstruction step's choice of which speakers to emit per
//! frame). DER target: ≤5% on community-1 evaluation sets; bit-
//! exact pyannote parity is reserved for the offline-from-captures
//! path (`offline::diarize_offline`).

use crate::{
  aggregate::try_count_pyannote,
  embed::{EMBEDDING_DIM, EmbedModel},
  offline::{Error, OfflineInput, OfflineOutput, diarize_offline},
  plda::PldaTransform,
  reconstruct::SlidingWindow,
  segment::{
    FRAMES_PER_WINDOW, POWERSET_CLASSES, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S,
    SAMPLE_RATE_HZ, SegmentModel, WINDOW_SAMPLES,
    powerset::{powerset_to_speakers_hard, softmax_row},
  },
  spill::SpillOptions,
};

// `min_duration_off` / `smoothing_epsilon` are validated against the single
// diaric-owned authority — `OwnedPipelineOptions` forwards both into
// `OfflineInput`, so accepting a value `diarize_offline` would later reject
// is a drift bug. Call diaric's exposed predicates rather than re-deriving
// the bounds. (`check_onset` stays local above: the onset knob only flows
// through the audio entrypoints, which diaric's tensor path does not model.)
use diaric::offline::{check_min_duration_off, check_smoothing_epsilon};

/// Number of speaker slots per chunk. Pyannote `segmentation-3.0`
/// trains on 3 simultaneous speakers (the 7 powerset classes).
pub const SLOTS_PER_CHUNK: usize = 3;

/// `const fn` predicate: `v` is finite and in `(0.0, 1.0]`. Mirrors
/// the segmentation `check_hysteresis_threshold` pattern: `f32::is_finite`
/// is not yet `const`, so we phrase the check via `v == v` (NaN check)
/// and direct `>`/`<=` comparisons that work on infinities.
///
/// Exposed `pub(crate)` so `streaming::offline_diarizer` can reuse the
/// same predicate (its diarization config is a re-export of
/// [`OwnedPipelineOptions`]).
#[inline]
pub(crate) const fn check_onset(v: f32) -> bool {
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && v > 0.0 && v <= 1.0
}

/// Configuration for [`OwnedDiarizationPipeline`].
///
/// Defaults match pyannote `speaker-diarization-community-1`:
/// 1-second chunk step, 0.5 onset/offset binarization, threshold/Fa/Fb
/// from the community-1 config.
///
/// Not `Copy`: [`Self::spill_options`] is a `SpillOptions` whose inner
/// `Option<PathBuf>` heap-owns its directory string.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OwnedPipelineOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_step_samples"))]
  step_samples: u32,
  #[cfg_attr(feature = "serde", serde(default = "default_onset"))]
  onset: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_threshold"))]
  threshold: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_fa"))]
  fa: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_fb"))]
  fb: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_max_iters"))]
  max_iters: usize,
  #[cfg_attr(feature = "serde", serde(default))]
  min_duration_off: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_smoothing_epsilon"))]
  smoothing_epsilon: Option<f32>,
  /// Spill backend configuration. Defaults to
  /// [`SpillOptions::default`] (64 MiB heap threshold,
  /// [`std::env::temp_dir`] spill directory).
  /// [`OwnedDiarizationPipeline::run`] passes this by reference to
  /// every [`crate::spill::SpillBytesMut::zeros`] reached transitively
  /// (AHC pdist, reconstruct grids, count buffers), so per-call
  /// configuration is local — no process-global side-effects.
  #[cfg_attr(feature = "serde", serde(default))]
  spill_options: SpillOptions,
}

#[cfg(feature = "serde")]
const fn default_step_samples() -> u32 {
  16_000
}
#[cfg(feature = "serde")]
const fn default_onset() -> f32 {
  0.5
}
#[cfg(feature = "serde")]
const fn default_threshold() -> f64 {
  0.6
}
#[cfg(feature = "serde")]
const fn default_fa() -> f64 {
  0.07
}
#[cfg(feature = "serde")]
const fn default_fb() -> f64 {
  0.8
}
#[cfg(feature = "serde")]
const fn default_max_iters() -> usize {
  20
}
#[cfg(feature = "serde")]
const fn default_smoothing_epsilon() -> Option<f32> {
  // Match pyannote's plain top-k argmax for bit-exact community-1
  // parity. Speakrs-style temporal smoothing (`Some(eps)`) is opt-in
  // via `with_smoothing_epsilon` for callers who want streaming-
  // friendly stable speaker assignments at the cost of segment
  // boundary precision.
  None
}

impl OwnedPipelineOptions {
  /// Construct with `community-1` defaults. `spill_options` defaults
  /// to [`SpillOptions::new`] (64 MiB threshold,
  /// [`std::env::temp_dir`] spill directory).
  pub const fn new() -> Self {
    Self {
      step_samples: 16_000, // 1 s — community-1 config
      onset: 0.5,
      threshold: 0.6,
      fa: 0.07,
      fb: 0.8,
      max_iters: 20,
      min_duration_off: 0.0,
      // `None` matches pyannote's plain top-k argmax in the discrete
      // diarization grid (`pyannote.audio.pipelines.utils.diarization
      // .Diarization.to_diarization`, line 261-266) — needed for
      // bit-exact RTTM segment boundaries on community-1. Callers
      // that want streaming-friendly stable speaker assignments
      // (speakrs-style) can opt in via
      // `with_smoothing_epsilon(Some(eps))` at the cost of merging
      // sub-100ms overlap-region splits.
      smoothing_epsilon: None,
      spill_options: SpillOptions::new(),
    }
  }

  // ── Getters ─────────────────────────────────────────────────────

  /// Sliding-window step in samples. Community-1 uses 16_000 (1 s).
  pub const fn step_samples(&self) -> u32 {
    self.step_samples
  }
  /// Frame-level binarization onset (default: 0.5).
  pub const fn onset(&self) -> f32 {
    self.onset
  }
  /// AHC linkage threshold (community-1: 0.6).
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }
  /// VBx Fa (community-1: 0.07).
  pub const fn fa(&self) -> f64 {
    self.fa
  }
  /// VBx Fb (community-1: 0.8).
  pub const fn fb(&self) -> f64 {
    self.fb
  }
  /// VBx max iterations (community-1 hardcodes 20).
  pub const fn max_iters(&self) -> usize {
    self.max_iters
  }
  /// Span post-processing min_duration_off (seconds).
  pub const fn min_duration_off(&self) -> f64 {
    self.min_duration_off
  }
  /// Temporal smoothing epsilon for top-k reconstruction.
  pub const fn smoothing_epsilon(&self) -> Option<f32> {
    self.smoothing_epsilon
  }
  /// Spill backend configuration. Installed on the process-global at
  /// the start of [`OwnedDiarizationPipeline::run`].
  pub const fn spill_options(&self) -> &SpillOptions {
    &self.spill_options
  }

  // ── Builders ────────────────────────────────────────────────────

  /// Builder: sliding-window step in samples.
  ///
  /// # Panics
  /// Panics if `v == 0` or `v > WINDOW_SAMPLES`. Zero step would hang
  /// the segmenter pump; `step > window` causes silent audio gaps
  /// between consecutive chunks (samples in `[window..step)` per
  /// chunk are never segmented).
  #[must_use]
  pub const fn with_step_samples(mut self, v: u32) -> Self {
    assert!(v > 0, "step_samples must be > 0");
    assert!(
      v <= crate::segment::WINDOW_SAMPLES,
      "step_samples must be <= WINDOW_SAMPLES (160_000)"
    );
    self.step_samples = v;
    self
  }
  /// Builder: frame-level binarization onset.
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `(0.0, 1.0]`. The hard 0/1
  /// segmentation comparison `seg >= onset` degenerates outside this
  /// range: NaN/`> 1.0` makes every frame inactive (empty
  /// diarization), `<= 0.0` makes every frame active (corrupted
  /// masks, embeddings, counts).
  #[must_use]
  pub const fn with_onset(mut self, v: f32) -> Self {
    assert!(check_onset(v), "onset must be finite in (0.0, 1.0]");
    self.onset = v;
    self
  }
  /// Builder: AHC linkage threshold.
  #[must_use]
  pub const fn with_threshold(mut self, v: f64) -> Self {
    self.threshold = v;
    self
  }
  /// Builder: VBx Fa.
  #[must_use]
  pub const fn with_fa(mut self, v: f64) -> Self {
    self.fa = v;
    self
  }
  /// Builder: VBx Fb.
  #[must_use]
  pub const fn with_fb(mut self, v: f64) -> Self {
    self.fb = v;
    self
  }
  /// Builder: VBx max iterations.
  #[must_use]
  pub const fn with_max_iters(mut self, v: usize) -> Self {
    self.max_iters = v;
    self
  }
  /// Builder: span post-processing `min_duration_off` (seconds).
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or negative. RTTM span-merge consumes
  /// this as a non-negative seconds quantity; `+inf` would merge every
  /// same-cluster gap and `NaN` would silently disable the merge
  /// (every comparison becomes false), both producing corrupted
  /// spans without surfacing the misconfiguration.
  #[must_use]
  pub const fn with_min_duration_off(mut self, v: f64) -> Self {
    assert!(
      check_min_duration_off(v),
      "min_duration_off must be finite and >= 0"
    );
    self.min_duration_off = v;
    self
  }
  /// Builder: temporal smoothing epsilon. Pass `None` for bit-exact
  /// pyannote argmax behavior, `Some(0.1)` for `community-1` smoothed
  /// reconstruction.
  ///
  /// # Panics
  /// Panics if `v` is `Some(NaN/±inf)` or `Some(< 0)`. The smoothing
  /// step compares activation differences against this epsilon;
  /// `Some(+inf)` collapses top-k selection onto stable index order,
  /// `Some(NaN)` makes every comparison false, both silently breaking
  /// reconstruction.
  #[must_use]
  pub const fn with_smoothing_epsilon(mut self, v: Option<f32>) -> Self {
    assert!(
      check_smoothing_epsilon(v),
      "smoothing_epsilon must be None or Some(finite >= 0)"
    );
    self.smoothing_epsilon = v;
    self
  }
  /// Builder: replace the spill backend configuration.
  ///
  /// Not `const fn` because dropping the previous `SpillOptions`
  /// runs `<PathBuf as Drop>::drop`, which is not const.
  #[must_use]
  pub fn with_spill_options(mut self, opts: SpillOptions) -> Self {
    self.spill_options = opts;
    self
  }
  /// Mutating: replace the spill backend configuration. Same semantics
  /// as [`Self::with_spill_options`].
  pub fn set_spill_options(&mut self, opts: SpillOptions) -> &mut Self {
    self.spill_options = opts;
    self
  }
}

impl Default for OwnedPipelineOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// End-to-end audio→RTTM offline diarization pipeline.
///
/// Borrows `&mut SegmentModel`, `&mut EmbedModel`, and `&PldaTransform`
/// per [`run`](Self::run) call. Both model types are `!Sync` (ORT
/// session state is single-threaded), so the caller owns them and
/// hands `&mut` references in — same pattern as
/// [`crate::streaming::StreamingOfflineDiarizer::push_voice_range`].
/// Configuration is held in [`OwnedPipelineOptions`].
pub struct OwnedDiarizationPipeline {
  options: OwnedPipelineOptions,
}

impl OwnedDiarizationPipeline {
  /// Construct with the community-1 default options.
  pub const fn new() -> Self {
    Self {
      options: OwnedPipelineOptions::new(),
    }
  }

  /// Construct with explicit options.
  pub fn with_options(options: OwnedPipelineOptions) -> Self {
    Self { options }
  }

  /// Borrow the options.
  pub fn options(&self) -> &OwnedPipelineOptions {
    &self.options
  }

  /// Run diarization on owned 16 kHz mono samples.
  ///
  /// Returns the same [`OfflineOutput`] shape as
  /// [`diarize_offline`](super::diarize_offline) — `(hard_clusters,
  /// discrete_diarization, num_clusters, spans)`.
  ///
  /// # Errors
  ///
  /// - [`Error::Core`] wrapping a [`crate::offline::ShapeError`] if `samples`
  ///   is empty or shorter than one
  ///   segmentation window (`WINDOW_SAMPLES = 160_000` = 10 s).
  /// - All other errors propagate from the underlying ONNX inference,
  ///   PLDA, AHC, VBx, centroid, Hungarian, or reconstruct stages.
  pub fn run(
    &self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    plda: &PldaTransform,
    samples: &[f32],
  ) -> Result<OfflineOutput, Error> {
    let cfg = &self.options;
    if samples.is_empty() {
      return Err(crate::offline::ShapeError::EmptySamples.into());
    }
    let win = WINDOW_SAMPLES as usize;
    let step = cfg.step_samples() as usize;
    if step == 0 {
      return Err(crate::offline::ShapeError::ZeroStepSamples.into());
    }
    // Defense-in-depth: `with_step_samples` panics on > WINDOW_SAMPLES,
    // but serde-deserialized configs bypass that path. Reject here too.
    if step > win {
      return Err(
        crate::offline::ShapeError::StepSamplesExceedsWindow {
          step: cfg.step_samples(),
          window: WINDOW_SAMPLES,
        }
        .into(),
      );
    }
    // Same defense-in-depth for `onset`. The `seg >= onset` mask
    // degenerates with NaN/`> 1.0` (all-inactive → empty diarization)
    // or `<= 0.0` (all-active → corrupted frame masks).
    if !check_onset(cfg.onset()) {
      return Err(crate::offline::ShapeError::OnsetOutOfRange { onset: cfg.onset() }.into());
    }
    // Same defense-in-depth for `min_duration_off` and
    // `smoothing_epsilon`. Both flow into reconstruction/RTTM
    // generation; non-finite or out-of-range values silently corrupt
    // span boundaries and top-k smoothing. See the predicates'
    // doc-comments and the typed error variants for the failure
    // modes each catches.
    if !check_min_duration_off(cfg.min_duration_off()) {
      return Err(
        crate::offline::ShapeError::MinDurationOffOutOfRange {
          value: cfg.min_duration_off(),
        }
        .into(),
      );
    }
    if !check_smoothing_epsilon(cfg.smoothing_epsilon()) {
      return Err(
        crate::offline::ShapeError::SmoothingEpsilonOutOfRange {
          value: cfg.smoothing_epsilon(),
        }
        .into(),
      );
    }
    // Preflight clustering hyperparameters (threshold/fa/fb/max_iters)
    // BEFORE running segmentation + embedding inference. These are
    // re-validated by `assign_embeddings` at the actual clustering
    // boundary, but a misconfigured production deployment with e.g.
    // `threshold = NaN` or `max_iters = 0` would otherwise burn an
    // entire model-inference pass before failing — making config
    // errors data-dependent and slow to detect. Surfacing them
    // upfront keeps validation latency bounded.
    use crate::pipeline::error::ShapeError as PipelineShapeError;
    let to_err = |s: PipelineShapeError| -> Error { crate::pipeline::Error::Shape(s).into() };
    if !cfg.threshold().is_finite() || cfg.threshold() <= 0.0 {
      return Err(to_err(PipelineShapeError::InvalidThreshold));
    }
    if !cfg.fa().is_finite() || cfg.fa() <= 0.0 {
      return Err(to_err(PipelineShapeError::InvalidFa));
    }
    if !cfg.fb().is_finite() || cfg.fb() <= 0.0 {
      return Err(to_err(PipelineShapeError::InvalidFb));
    }
    if cfg.max_iters() == 0 {
      return Err(to_err(PipelineShapeError::ZeroMaxIters));
    }
    if cfg.max_iters() > crate::cluster::vbx::MAX_ITERS_CAP {
      return Err(to_err(PipelineShapeError::MaxItersExceedsCap {
        got: cfg.max_iters(),
        cap: crate::cluster::vbx::MAX_ITERS_CAP,
      }));
    }

    // ── Stage 1: chunked sliding-window segmentation ───────────────
    // Last-chunk zero-pad if `samples` doesn't align with the grid.
    let num_chunks = if samples.len() <= win {
      1
    } else {
      (samples.len() - win).div_ceil(step) + 1
    };

    // `padded_chunk` is fixed at WINDOW_SAMPLES = 160_000 f32 = 640 KB
    // — well under any conceivable spill threshold. Leave on heap.
    let mut padded_chunk = vec![0.0_f32; win];
    // `segmentations` and `raw_embeddings` scale with audio length:
    // `segmentations` ≈ 50 MB / hour (f64), `raw_embeddings` ≈ 11 MB /
    // hour (f32). Multi-hour recordings cross the 64 MiB default
    // spill threshold; route through `SpillBytesMut` so the heap path is
    // bounded and large allocations fall back to file-backed mmap.
    let segs_len = num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK;
    let mut segmentations =
      crate::spill::SpillBytesMut::<f64>::zeros(segs_len, cfg.spill_options())?;
    let segs = segmentations.as_mut_slice();

    for c in 0..num_chunks {
      let start = c * step;
      // Build the (possibly zero-padded) 10s window.
      padded_chunk.fill(0.0);
      let end = (start + win).min(samples.len());
      let lo = start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      let logits = seg_model.infer(&padded_chunk)?;
      // logits is [FRAMES_PER_WINDOW * POWERSET_CLASSES] row-major.
      for f in 0..FRAMES_PER_WINDOW {
        let mut row = [0.0_f32; POWERSET_CLASSES];
        for k in 0..POWERSET_CLASSES {
          row[k] = logits[f * POWERSET_CLASSES + k];
        }
        let probs = softmax_row(&row);
        // Pyannote's `to_multilabel(powerset, soft=False)` picks the
        // argmax powerset class, then maps to the speaker mask. This
        // is the conversion captured `segmentations.npz` reflects —
        // every entry is exactly 0.0 or 1.0. Soft marginals followed
        // by `>= onset` would disagree on 3-way overlap chunks where
        // the marginal sum exceeds 0.5 but argmax picks a different
        // class. Critical for `filter_embeddings`'s `single_active`
        // mask (frames where sum_speakers == 1) and for `count`,
        // both of which assume hard argmax binarization.
        let speakers = powerset_to_speakers_hard(&probs);
        for s in 0..SLOTS_PER_CHUNK {
          segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = speakers[s] as f64;
        }
      }
    }

    // ── Stage 2: per-(chunk, slot) masked embedding ────────────────
    let emb_len = num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM;
    let mut raw_embeddings =
      crate::spill::SpillBytesMut::<f32>::zeros(emb_len, cfg.spill_options())?;
    let embs = raw_embeddings.as_mut_slice();

    // Pyannote's `get_embeddings` (community-1 default
    // `embedding_exclude_overlap=True`) zeroes out frames where two or
    // more speakers are simultaneously active before extracting each
    // speaker's embedding, then falls back to the original mask only
    // when too few "clean" frames remain. The threshold is
    // `min_num_frames = ceil(num_frames * embedding_min_num_samples /
    // (chunk_duration * embedding_sample_rate)) = ceil(589 * 400 /
    // (10 * 16000)) = 2` for the WeSpeaker pyannote ships. Without
    // this exclusion dia's per-(chunk, speaker) embedding mixes the
    // overlap region's competing speakers into a single vector,
    // producing a centroid that's halfway between the two real
    // speakers and flipping AHC threshold decisions on long
    // recordings.
    //
    // pyannote/audio/pipelines/speaker_diarization.py:375-397.
    const EXCLUDE_OVERLAP_MIN_FRAMES: usize = 2;

    for c in 0..num_chunks {
      let start = c * step;
      // Re-slice the same padded window we used for segmentation so
      // mask offsets line up. Zero-pad samples outside the audio range.
      padded_chunk.fill(0.0);
      let end = (start + win).min(samples.len());
      let lo = start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      // Per-frame "clean" indicator: 1 iff fewer than 2 speakers are
      // active in this frame across the full SLOTS_PER_CHUNK = 3 slots.
      // Computed once per chunk and reused across each speaker's
      // overlap-excluded mask construction.
      let mut clean_frame = [false; FRAMES_PER_WINDOW];
      for f in 0..FRAMES_PER_WINDOW {
        let mut active_count = 0u8;
        for s in 0..SLOTS_PER_CHUNK {
          if segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] >= cfg.onset() as f64 {
            active_count += 1;
          }
        }
        clean_frame[f] = active_count < 2;
      }

      for s in 0..SLOTS_PER_CHUNK {
        // Build per-frame binary mask: speaker active iff seg > onset.
        let mut frame_mask = [false; FRAMES_PER_WINDOW];
        let mut any_active = false;
        for f in 0..FRAMES_PER_WINDOW {
          let active =
            segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] >= cfg.onset() as f64;
          frame_mask[f] = active;
          any_active |= active;
        }
        if !any_active {
          // Zero the segmentation column so filter_embeddings drops
          // this (c, s) pair. Without this, sub-onset segmentation
          // sums (e.g. 0.0001 from ONNX softmax noise) would still
          // satisfy `sum > 0` and admit a zero-embedding into PLDA,
          // failing `RawEmbedding::from_raw_array`'s norm guard.
          for f in 0..FRAMES_PER_WINDOW {
            segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
          }
          continue;
        }

        // Build overlap-excluded clean mask + count clean-active
        // frames. Match pyannote's exact rule: use the clean mask only
        // when its active-frame count strictly exceeds
        // `EXCLUDE_OVERLAP_MIN_FRAMES = 2`. The strict-greater-than
        // here matters — pyannote uses `np.sum(clean_mask) >
        // min_num_frames`, not `>=`, so an exactly-2-frame clean
        // mask falls back to the full mask just like dia does here.
        let mut used_mask = [false; FRAMES_PER_WINDOW];
        let mut clean_count = 0usize;
        for f in 0..FRAMES_PER_WINDOW {
          let v = frame_mask[f] && clean_frame[f];
          used_mask[f] = v;
          if v {
            clean_count += 1;
          }
        }
        if clean_count <= EXCLUDE_OVERLAP_MIN_FRAMES {
          used_mask = frame_mask;
        }

        // Run pyannote-style chunk + frame-mask embedding. The
        // EmbedModel's `embed_chunk_with_frame_mask` dispatches based
        // on the active backend: ORT zeroes audio + sliding-window
        // aggregates (approximate); tch passes (audio, mask) directly
        // to the TorchScript wrapper which delegates to pyannote's
        // `WeSpeakerResNet34.forward(waveforms, weights=mask)` —
        // bit-exact pyannote.
        let raw = match embed_model.embed_chunk_with_frame_mask(&padded_chunk, &used_mask) {
          Ok(v) => v,
          Err(crate::embed::Error::Core(diaric::embed::Error::InvalidClip { .. }))
          | Err(crate::embed::Error::Core(diaric::embed::Error::DegenerateEmbedding)) => {
            for f in 0..FRAMES_PER_WINDOW {
              segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
            }
            continue;
          }
          Err(e) => return Err(e.into()),
        };
        // Reject non-finite embedding output as a hard error. Previously
        // a NaN/inf vector was lumped together with the legitimate
        // low-norm drop path below, silently turning ONNX/provider
        // corruption into "inactive speaker" and producing diarization
        // with missing speech instead of surfacing the failure.
        if raw.iter().any(|v| !v.is_finite()) {
          return Err(crate::embed::Error::Core(diaric::embed::Error::NonFiniteOutput).into());
        }
        // Pre-validate: if the raw norm is below the PLDA min, drop.
        // PLDA min is 0.01 (RawEmbedding::from_raw_array). Computing
        // the L2 norm here lets us drop the slot before
        // `diarize_offline` rejects it later. Norm is finite by the
        // check above, so `< 0.01` is the only path that fires here.
        let norm_sq: f64 = raw.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        if norm_sq.sqrt() < 0.01 {
          for f in 0..FRAMES_PER_WINDOW {
            segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
          }
          continue;
        }
        let dst = (c * SLOTS_PER_CHUNK + s) * EMBEDDING_DIM;
        embs[dst..dst + EMBEDDING_DIM].copy_from_slice(&raw);
      }
    }
    // Drop the mutable handles before reborrowing as immutable for
    // the count + offline-input dispatch below.
    let _ = (segs, embs);

    // ── Stage 3: build count tensor + sliding-window timing ────────
    //
    // Bit-exact to pyannote 4.0.4's
    // `SpeakerDiarizationMixin.speaker_count` →
    // `Inference.aggregate(hamming=False, skip_average=False,
    // missing=0.0)` with `warm_up=(0.0, 0.0)` (community-1's explicit
    // override of the default `(0.1, 0.1)`).
    //
    // Critical algorithmic property: per-frame count is uniform-
    // averaged across non-NaN contributing chunks, NOT
    // hamming-weighted. The previous implementation used hamming
    // weights and divided by total weight rather than overlap count;
    // see `aggregate::count_pyannote` source for the algorithm and
    // `aggregate::parity_tests` for the bit-exact fixture parity.
    let chunk_duration_s = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    let chunk_step_s = cfg.step_samples() as f64 / SAMPLE_RATE_HZ as f64;
    let chunks_sw = SlidingWindow::new(0.0, chunk_duration_s, chunk_step_s);
    let frames_sw_template =
      SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S);
    // Use the fallible variant: a malformed `onset` (NaN/inf via the
    // public `with_onset` builder) would panic the infallible
    // `count_pyannote` wrapper at `try_count_pyannote.expect(...)`.
    // Surface it as a typed `Error::Aggregate` instead so untrusted
    // config can never crash the process.
    let (count, frames_sw) = try_count_pyannote(
      segmentations.as_slice(),
      num_chunks,
      FRAMES_PER_WINDOW,
      SLOTS_PER_CHUNK,
      cfg.onset() as f64,
      chunks_sw,
      frames_sw_template,
      cfg.spill_options(),
    )?
    .into_parts();
    let num_output_frames = count.len();

    // ── Stage 4: dispatch to diarize_offline ───────────────────────
    let input = OfflineInput::new(
      raw_embeddings.as_slice(),
      num_chunks,
      SLOTS_PER_CHUNK,
      segmentations.as_slice(),
      FRAMES_PER_WINDOW,
      &count,
      num_output_frames,
      chunks_sw,
      frames_sw,
      plda,
    )
    .with_threshold(cfg.threshold())
    .with_fa(cfg.fa())
    .with_fb(cfg.fb())
    .with_max_iters(cfg.max_iters())
    .with_min_duration_off(cfg.min_duration_off())
    .with_smoothing_epsilon(cfg.smoothing_epsilon())
    .with_spill_options(cfg.spill_options().clone());
    diarize_offline(&input).map_err(Error::Core)
  }
}

impl Default for OwnedDiarizationPipeline {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod option_validation_tests {
  use super::*;

  #[test]
  fn check_onset_predicate() {
    assert!(check_onset(0.5));
    assert!(check_onset(1.0));
    assert!(check_onset(f32::EPSILON));
    assert!(!check_onset(0.0));
    assert!(!check_onset(-0.01));
    assert!(!check_onset(1.01));
    assert!(!check_onset(f32::NAN));
    assert!(!check_onset(f32::INFINITY));
    assert!(!check_onset(f32::NEG_INFINITY));
  }

  #[test]
  #[should_panic(expected = "step_samples must be > 0")]
  fn with_step_samples_zero_panics() {
    let _ = OwnedPipelineOptions::new().with_step_samples(0);
  }

  /// `step > WINDOW_SAMPLES` would skip `step - window` samples per
  /// chunk in the offline planner. Reject at validation.
  #[test]
  #[should_panic(expected = "step_samples must be <= WINDOW_SAMPLES")]
  fn with_step_samples_above_window_panics() {
    let _ = OwnedPipelineOptions::new().with_step_samples(crate::segment::WINDOW_SAMPLES + 1);
  }

  /// Boundary: step == WINDOW_SAMPLES is allowed (no-overlap chunks).
  #[test]
  fn with_step_samples_equal_to_window_ok() {
    let o = OwnedPipelineOptions::new().with_step_samples(crate::segment::WINDOW_SAMPLES);
    assert_eq!(o.step_samples(), crate::segment::WINDOW_SAMPLES);
  }

  #[test]
  #[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
  fn with_onset_zero_panics() {
    let _ = OwnedPipelineOptions::new().with_onset(0.0);
  }

  #[test]
  #[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
  fn with_onset_negative_panics() {
    let _ = OwnedPipelineOptions::new().with_onset(-0.01);
  }

  #[test]
  #[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
  fn with_onset_above_one_panics() {
    let _ = OwnedPipelineOptions::new().with_onset(1.01);
  }

  #[test]
  #[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
  fn with_onset_nan_panics() {
    let _ = OwnedPipelineOptions::new().with_onset(f32::NAN);
  }

  #[test]
  #[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
  fn with_onset_inf_panics() {
    let _ = OwnedPipelineOptions::new().with_onset(f32::INFINITY);
  }

  /// Boundary: onset == 1.0 is allowed (degenerate but valid).
  #[test]
  fn with_onset_one_ok() {
    let o = OwnedPipelineOptions::new().with_onset(1.0);
    assert_eq!(o.onset(), 1.0);
  }

  // ── min_duration_off / smoothing_epsilon validation ──────────────
  //
  // The predicates (`check_min_duration_off`, `check_smoothing_epsilon`)
  // are owned by `diaric` and imported from `diaric::offline`. These
  // contract guards pin that this crate validates against that single
  // authority: a diaric bound change surfaces here rather than letting the
  // preflight silently diverge from what `diarize_offline` enforces.

  #[test]
  fn check_min_duration_off_predicate() {
    assert!(check_min_duration_off(0.0));
    assert!(check_min_duration_off(0.5));
    assert!(check_min_duration_off(1e10));
    assert!(!check_min_duration_off(-0.0001));
    assert!(!check_min_duration_off(f64::NAN));
    assert!(!check_min_duration_off(f64::INFINITY));
    assert!(!check_min_duration_off(f64::NEG_INFINITY));
  }

  #[test]
  fn check_smoothing_epsilon_predicate() {
    assert!(check_smoothing_epsilon(None));
    assert!(check_smoothing_epsilon(Some(0.0)));
    assert!(check_smoothing_epsilon(Some(0.1)));
    assert!(check_smoothing_epsilon(Some(1e6)));
    assert!(!check_smoothing_epsilon(Some(-0.001)));
    assert!(!check_smoothing_epsilon(Some(f32::NAN)));
    assert!(!check_smoothing_epsilon(Some(f32::INFINITY)));
    assert!(!check_smoothing_epsilon(Some(f32::NEG_INFINITY)));
  }

  #[test]
  #[should_panic(expected = "min_duration_off must be finite and >= 0")]
  fn with_min_duration_off_nan_panics() {
    let _ = OwnedPipelineOptions::new().with_min_duration_off(f64::NAN);
  }

  #[test]
  #[should_panic(expected = "min_duration_off must be finite and >= 0")]
  fn with_min_duration_off_inf_panics() {
    let _ = OwnedPipelineOptions::new().with_min_duration_off(f64::INFINITY);
  }

  #[test]
  #[should_panic(expected = "min_duration_off must be finite and >= 0")]
  fn with_min_duration_off_negative_panics() {
    let _ = OwnedPipelineOptions::new().with_min_duration_off(-0.5);
  }

  #[test]
  #[should_panic(expected = "smoothing_epsilon must be None or Some(finite >= 0)")]
  fn with_smoothing_epsilon_nan_panics() {
    let _ = OwnedPipelineOptions::new().with_smoothing_epsilon(Some(f32::NAN));
  }

  #[test]
  #[should_panic(expected = "smoothing_epsilon must be None or Some(finite >= 0)")]
  fn with_smoothing_epsilon_inf_panics() {
    let _ = OwnedPipelineOptions::new().with_smoothing_epsilon(Some(f32::INFINITY));
  }

  #[test]
  #[should_panic(expected = "smoothing_epsilon must be None or Some(finite >= 0)")]
  fn with_smoothing_epsilon_negative_panics() {
    let _ = OwnedPipelineOptions::new().with_smoothing_epsilon(Some(-0.001));
  }

  /// Boundary: zero is allowed for both knobs.
  #[test]
  fn with_min_duration_off_zero_ok() {
    let o = OwnedPipelineOptions::new().with_min_duration_off(0.0);
    assert_eq!(o.min_duration_off(), 0.0);
  }

  #[test]
  fn with_smoothing_epsilon_none_ok() {
    let o = OwnedPipelineOptions::new().with_smoothing_epsilon(None);
    assert_eq!(o.smoothing_epsilon(), None);
  }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
  use super::*;

  #[test]
  fn owned_pipeline_config_default_roundtrip() {
    let cfg = OwnedPipelineOptions::new();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: OwnedPipelineOptions = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg.step_samples(), back.step_samples());
    assert_eq!(cfg.threshold(), back.threshold());
    assert_eq!(cfg.fa(), back.fa());
    assert_eq!(cfg.fb(), back.fb());
    assert_eq!(cfg.max_iters(), back.max_iters());
    assert_eq!(cfg.smoothing_epsilon(), back.smoothing_epsilon());
  }

  /// Empty JSON object → all defaults filled in.
  #[test]
  fn owned_pipeline_config_empty_json_uses_defaults() {
    let cfg: OwnedPipelineOptions = serde_json::from_str("{}").expect("deserialize");
    let want = OwnedPipelineOptions::new();
    assert_eq!(cfg.step_samples(), want.step_samples());
    assert_eq!(cfg.onset(), want.onset());
    assert_eq!(cfg.threshold(), want.threshold());
    assert_eq!(cfg.smoothing_epsilon(), want.smoothing_epsilon());
  }
}
