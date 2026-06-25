//! Voice-range-driven streaming diarizer that produces pyannote-
//! equivalent global speaker assignments.
//!
//! Architecture: [`StreamingOfflineDiarizer::push_voice_range`] runs
//! the heavy stages 1+2 (sliding-window segmentation + masked
//! embedding) on each VAD-emitted voice range and accumulates the
//! derived tensors. [`StreamingOfflineDiarizer::finalize`] runs the
//! single global pyannote `cluster_vbx` pass (PLDA + AHC + VBx +
//! centroid + Hungarian) on the union of accumulated chunks, then
//! reconstructs per-range frame-level diarization and maps spans
//! back to the original timeline.
//!
//! ## Why not per-range clustering with cross-range bank
//!
//! The previous `StreamingDiarizationPipeline` ran full pyannote
//! offline diarization on each voice range independently and matched
//! cluster centroids across ranges via cosine bank. Two problems:
//!
//! 1. **Per-range AHC has no cross-range context.** A speaker who
//!    appears only briefly in range A and dominantly in range B can
//!    be merged with a different speaker in A (because A doesn't
//!    have enough evidence) and become a separate cluster from B.
//! 2. **Cosine bank in raw-embedding space is noisier than PLDA**.
//!    Pyannote clusters in PLDA-projected space because PLDA
//!    suppresses channel/session variance. Raw cosine bank inherits
//!    the unsuppressed variance and over- or under-merges.
//!
//! Running global AHC + VBx on the union of all voice ranges' chunks
//! mirrors what pyannote does on the full recording — each voice
//! range contributes its (chunk, slot) embeddings to one global
//! clustering, so cross-range identity is established by the same
//! algorithm pyannote uses, not a side-channel cosine bank.
//!
//! ## Memory & latency
//!
//! Per chunk: 589 frames × 3 slots × 8 B (segmentations) + 3 slots
//! × 256 dims × 4 B (raw embeddings) + ~10 KB count tensor ≈ 17 KB.
//! For 1 hour of audio with the community-1 1 s chunk step that's
//! ~3600 chunks ≈ 60 MB of accumulated tensors — bounded and small
//! relative to the PCM buffer the previous pipeline retained.
//!
//! Latency is `finalize`-bound: the offline clustering pass scales
//! roughly as O(num_train²) for AHC and O(num_train · plda_dim²) for
//! VBx, where `num_train` ≈ active (chunk, slot) pairs. For a 1 h
//! conversation that's ~10 000 pairs — multi-second clustering. For
//! near-realtime indexing this is acceptable; sub-range live-streaming
//! latency would need an online clusterer that dia does not currently
//! ship.

use std::sync::Arc;

use crate::{
  aggregate::try_count_pyannote,
  embed::{EMBEDDING_DIM, EmbedModel},
  offline::{OfflineInput, OwnedPipelineOptions, diarize_offline},
  ops::spill::SpillOptions,
  plda::PldaTransform,
  reconstruct::{
    ReconstructInput, RttmSpan, SlidingWindow, discrete_to_spans, reconstruct as reconstruct_grid,
  },
  segment::{
    FRAMES_PER_WINDOW, POWERSET_CLASSES, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S,
    SAMPLE_RATE_HZ, SegmentModel, WINDOW_SAMPLES,
    powerset::{powerset_to_speakers_hard, softmax_row},
  },
  streaming::RangeEmbeddings,
};

/// Number of speaker slots per chunk. Same as
/// [`crate::offline::SLOTS_PER_CHUNK`]; duplicated here for module
/// independence.
const SLOTS_PER_CHUNK: usize = 3;

/// Errors from the streaming offline diarizer.
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
  /// Input shape / call ordering is invalid — see
  /// `StreamingShapeError`.
  #[error("streaming: shape: {0}")]
  Shape(#[from] StreamingShapeError),
  /// Wraps a segmentation-stage failure message (typically an ONNX
  /// inference error stringified upfront because
  /// [`crate::segment::Error`] doesn't always satisfy `Send`).
  #[error("streaming: segment: {0}")]
  Segment(String),
  /// Wraps an embedding-stage failure message.
  #[error("streaming: embed: {0}")]
  Embed(String),
  /// Propagated from the underlying [`crate::offline`] entrypoint
  /// invoked by `finalize`.
  #[error("streaming: offline: {0}")]
  Offline(#[from] crate::offline::Error),
  /// Propagated from [`crate::reconstruct`].
  #[error("streaming: reconstruct: {0}")]
  Reconstruct(#[from] crate::reconstruct::Error),
  /// Propagated from `aggregate::try_count_pyannote` when the count
  /// tensor cannot be computed (e.g. NaN/inf `onset` from a
  /// misconfigured `OwnedPipelineOptions`). Replaces a panic path
  /// through the infallible `count_pyannote` wrapper.
  #[error("streaming: aggregate: {0}")]
  Aggregate(#[from] crate::aggregate::Error),
  /// Propagated from `crate::ops::spill::SpillBytesMut::zeros` when the
  /// per-range or concatenated scratch buffers cannot be allocated.
  /// At multi-hour scale these cross the 64 MiB default threshold
  /// and route through the file-backed mmap path; this surfaces
  /// tempfile / mmap failures from a `Result`-returning API.
  #[error("streaming: spill: {0}")]
  Spill(#[from] crate::ops::spill::SpillError),
}

/// Specific shape-violation reasons for [`StreamingError::Shape`].
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq)]
pub enum StreamingShapeError {
  #[error("voice range samples is empty")]
  EmptyVoiceRange,
  #[error("step_samples must be > 0")]
  ZeroStepSamples,
  #[error("all accumulated voice ranges are empty")]
  AllRangesEmpty,
  /// `step_samples` exceeds `WINDOW_SAMPLES`. The chunk planner uses
  /// `start = c * step` and stops after
  /// `(samples.len() - win).div_ceil(step) + 1` chunks; with `step >
  /// win`, samples in `[win .. step)` per chunk are never segmented
  /// or embedded — silent data loss returning `Ok(_)` with missing
  /// speech. Same constraint as `OwnedPipelineOptions::with_step_samples`.
  #[error("step_samples ({step}) must not exceed WINDOW_SAMPLES ({window})")]
  StepSamplesExceedsWindow { step: u32, window: u32 },
  /// `onset` is outside the documented `(0.0, 1.0]` range. Same
  /// constraint as `OwnedPipelineOptions::with_onset`. The hard 0/1
  /// segmentation mask `seg >= onset` degenerates: NaN/`> 1.0` makes
  /// every frame inactive, `<= 0.0` makes every frame active.
  #[error("onset ({onset}) must be finite in (0.0, 1.0]")]
  OnsetOutOfRange { onset: f32 },
  /// `min_duration_off` is NaN/±inf or negative. Same constraint as
  /// `OwnedPipelineOptions::with_min_duration_off`. Catches serde-
  /// bypassed configs whose value reaches the run path unchecked.
  #[error("min_duration_off ({value}) must be finite and >= 0")]
  MinDurationOffOutOfRange { value: f64 },
  /// `smoothing_epsilon` is `Some(NaN/±inf)` or `Some(< 0)`. Same
  /// constraint as `OwnedPipelineOptions::with_smoothing_epsilon`.
  #[error("smoothing_epsilon ({value:?}) must be None or Some(finite >= 0)")]
  SmoothingEpsilonOutOfRange { value: Option<f32> },
  /// AHC merge threshold is non-finite or non-positive. Caught
  /// upfront so a misconfigured config doesn't burn per-range
  /// segmentation + embedding inference before failing at the
  /// final clustering boundary.
  #[error("threshold ({value}) must be a positive finite scalar")]
  InvalidThreshold { value: f64 },
  /// VBx EM `fa` is non-finite or non-positive.
  #[error("fa ({value}) must be a positive finite scalar")]
  InvalidFa { value: f64 },
  /// VBx EM `fb` is non-finite or non-positive.
  #[error("fb ({value}) must be a positive finite scalar")]
  InvalidFb { value: f64 },
  /// `max_iters == 0`. Caught upfront in the streaming push path.
  #[error("max_iters must be at least 1")]
  ZeroMaxIters,
  /// `max_iters` exceeds the documented cap. Caught upfront.
  #[error("max_iters ({got}) exceeds cap ({cap})")]
  MaxItersExceedsCap { got: usize, cap: usize },
}

/// Configuration for [`StreamingOfflineDiarizer`].
///
/// The spill backend configuration lives on the inner
/// [`OwnedPipelineOptions`]; there is no separate
/// `StreamingOfflineOptions::spill_options` field. Single source
/// of truth means [`Self::with_diarization`] correctly carries
/// the caller's spill settings through.
///
/// Not `Copy`: [`OwnedPipelineOptions`] holds a `SpillOptions` value
/// (heap-owned `Option<PathBuf>`).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingOfflineOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  diarization: OwnedPipelineOptions,
}

impl StreamingOfflineOptions {
  /// Construct with `community-1` diarization defaults (which
  /// include the default spill configuration).
  pub const fn new() -> Self {
    Self {
      diarization: OwnedPipelineOptions::new(),
    }
  }

  /// Borrow the inner diarization parameters.
  pub const fn diarization(&self) -> &OwnedPipelineOptions {
    &self.diarization
  }

  /// Borrow the spill backend configuration. Delegates to the
  /// inner [`OwnedPipelineOptions::spill_options`] — there is no
  /// separate streaming-level field.
  pub const fn spill_options(&self) -> &SpillOptions {
    self.diarization.spill_options()
  }

  /// Builder: replace the diarization parameters. Carries the
  /// new options' spill configuration through automatically.
  ///
  /// Not `const fn`: [`OwnedPipelineOptions`] has a non-const
  /// destructor through [`SpillOptions`]'s `PathBuf`.
  #[must_use]
  pub fn with_diarization(mut self, diarization: OwnedPipelineOptions) -> Self {
    self.diarization = diarization;
    self
  }

  /// Builder: replace the spill backend configuration on the inner
  /// [`OwnedPipelineOptions`]. Equivalent to
  /// `with_diarization(self.diarization().clone().with_spill_options(opts))`,
  /// but without the intermediate clone.
  #[must_use]
  pub fn with_spill_options(mut self, opts: SpillOptions) -> Self {
    self.diarization.set_spill_options(opts);
    self
  }

  /// Mutating: replace the spill backend configuration on the inner
  /// [`OwnedPipelineOptions`].
  pub fn set_spill_options(&mut self, opts: SpillOptions) -> &mut Self {
    self.diarization.set_spill_options(opts);
    self
  }
}

/// One diarized span in the original audio timeline.
#[derive(Debug, Clone)]
pub struct DiarizedSpan {
  start_sample: u64,
  end_sample: u64,
  speaker_id: u32,
}

impl DiarizedSpan {
  /// Construct.
  pub const fn new(start_sample: u64, end_sample: u64, speaker_id: u32) -> Self {
    Self {
      start_sample,
      end_sample,
      speaker_id,
    }
  }

  /// Absolute start sample (relative to the start of the input
  /// audio stream that drove `push_voice_range`).
  pub const fn start_sample(&self) -> u64 {
    self.start_sample
  }

  /// Absolute end sample.
  pub const fn end_sample(&self) -> u64 {
    self.end_sample
  }

  /// Globally-tracked speaker id, consistent across all voice
  /// ranges pushed before `finalize`.
  pub const fn speaker_id(&self) -> u32 {
    self.speaker_id
  }
}

/// Voice-range-driven streaming diarizer.
///
/// Caller drives VAD externally and pushes one voice range per VAD
/// segment. At end-of-stream, [`finalize`](Self::finalize) runs the
/// global clustering pass and returns spans on the original
/// timeline.
pub struct StreamingOfflineDiarizer {
  options: StreamingOfflineOptions,
  ranges: Vec<RangeEmbeddings>,
}

impl StreamingOfflineDiarizer {
  /// Construct an empty diarizer.
  ///
  /// Push voice ranges via [`Self::push_voice_range`] as the VAD
  /// emits them, then call [`Self::finalize`] once at end-of-stream
  /// to run global clustering and emit RTTM spans. `options` carries
  /// the spill threshold and reconstruction knobs forwarded into the
  /// underlying offline pipeline.
  pub fn new(options: StreamingOfflineOptions) -> Self {
    Self {
      options,
      ranges: Vec::new(),
    }
  }

  /// Borrow the options.
  pub fn options(&self) -> &StreamingOfflineOptions {
    &self.options
  }

  /// Number of voice ranges accumulated so far.
  pub fn num_ranges(&self) -> usize {
    self.ranges.len()
  }

  /// Push one voice range. Runs segmentation + embedding + count
  /// tensor computation on the supplied PCM and stores the derived
  /// tensors. Does NOT cluster — that happens at
  /// [`finalize`](Self::finalize).
  ///
  /// `abs_start_sample` is the absolute sample index in the
  /// original audio stream where this range starts; it's used at
  /// `finalize` to remap output spans back to the original timeline.
  ///
  /// # Errors
  ///
  /// - [`StreamingError::Shape`] if `samples.is_empty()` or
  ///   `step_samples == 0`.
  /// - [`StreamingError::Segment`] / [`StreamingError::Embed`] for
  ///   ONNX inference failures on the range.
  pub fn push_voice_range(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    abs_start_sample: u64,
    samples: &[f32],
  ) -> Result<(), StreamingError> {
    let range = build_range(
      &self.options,
      seg_model,
      embed_model,
      abs_start_sample,
      samples,
    )?;
    self.ranges.push(range);
    Ok(())
  }

  /// Run global clustering on the union of accumulated voice ranges
  /// and return original-timeline spans.
  ///
  /// Operationally:
  /// 1. Concatenate all ranges' segmentations / raw_embeddings into
  ///    a single `(total_chunks, FRAMES_PER_WINDOW, SLOTS_PER_CHUNK)`
  ///    tensor and a single `(total_chunks, SLOTS_PER_CHUNK,
  ///    EMBEDDING_DIM)` embedding tensor.
  /// 2. Concatenate count tensors. The chunks_sw passed to
  ///    `diarize_offline` is irrelevant for the clustering stages
  ///    (they ignore timing); we pass the first range's chunks_sw
  ///    so the output's reconstruct stage sees a valid SlidingWindow.
  ///    We then re-run reconstruct PER RANGE with each range's local
  ///    timing and the corresponding slice of `hard_clusters`.
  /// 3. Per range, build spans via `discrete_to_spans` and offset
  ///    by `abs_start_sample / SR`.
  ///
  /// # Errors
  ///
  /// - [`StreamingError::Shape`] if no voice ranges have been
  ///   pushed or any range's chunk count is zero.
  /// - All other errors propagate from `diarize_offline` /
  ///   `reconstruct`.
  pub fn finalize(&self, plda: &PldaTransform) -> Result<Arc<[DiarizedSpan]>, StreamingError> {
    cluster_ranges_inner(&self.ranges, &self.options, plda)
  }

  /// Drop accumulated tensors. Useful for reusing the same diarizer
  /// across multiple sessions. Does not reset speaker-id assignment
  /// since IDs are decided at `finalize`-time, not held as state.
  pub fn reset(&mut self) {
    self.ranges.clear();
  }
}

/// Run the fused segmentation + masked-embedding + count pipeline on one
/// voice range and return the public [`RangeEmbeddings`] carrier. Shared
/// by [`StreamingOfflineDiarizer::push_voice_range`] (which accumulates
/// the carrier internally) and
/// [`crate::streaming::StreamingEmbedder::push`] (which hands it to the
/// caller). Emits RAW, unnormalized WeSpeaker vectors — never
/// L2-normalized.
pub(crate) fn build_range(
  options: &StreamingOfflineOptions,
  seg_model: &mut SegmentModel,
  embed_model: &mut EmbedModel,
  abs_start_sample: u64,
  samples: &[f32],
) -> Result<RangeEmbeddings, StreamingError> {
  let cfg = options.diarization();
  if samples.is_empty() {
    return Err(StreamingShapeError::EmptyVoiceRange.into());
  }
  let win = WINDOW_SAMPLES as usize;
  let step = cfg.step_samples() as usize;
  if step == 0 {
    return Err(StreamingShapeError::ZeroStepSamples.into());
  }
  // Defense-in-depth: `OwnedPipelineOptions::with_step_samples`
  // panics on > WINDOW_SAMPLES, but serde-deserialized configs
  // bypass that path. See StreamingShapeError::StepSamplesExceedsWindow.
  if step > win {
    return Err(
      StreamingShapeError::StepSamplesExceedsWindow {
        step: cfg.step_samples(),
        window: WINDOW_SAMPLES,
      }
      .into(),
    );
  }
  // Same defense-in-depth for onset.
  if !crate::offline::check_onset(cfg.onset()) {
    return Err(StreamingShapeError::OnsetOutOfRange { onset: cfg.onset() }.into());
  }
  if !crate::offline::check_min_duration_off(cfg.min_duration_off()) {
    return Err(
      StreamingShapeError::MinDurationOffOutOfRange {
        value: cfg.min_duration_off(),
      }
      .into(),
    );
  }
  if !crate::offline::check_smoothing_epsilon(cfg.smoothing_epsilon()) {
    return Err(
      StreamingShapeError::SmoothingEpsilonOutOfRange {
        value: cfg.smoothing_epsilon(),
      }
      .into(),
    );
  }
  // Preflight clustering hyperparameters BEFORE running per-range
  // segmentation + embedding inference. `finalize` re-validates,
  // but a misconfigured `threshold`/`fa`/`fb`/`max_iters` would
  // otherwise burn every range's model-inference pass before
  // failing at the global clustering boundary. Surface the error
  // upfront on the first `push_voice_range` call.
  if !cfg.threshold().is_finite() || cfg.threshold() <= 0.0 {
    return Err(
      StreamingShapeError::InvalidThreshold {
        value: cfg.threshold(),
      }
      .into(),
    );
  }
  if !cfg.fa().is_finite() || cfg.fa() <= 0.0 {
    return Err(StreamingShapeError::InvalidFa { value: cfg.fa() }.into());
  }
  if !cfg.fb().is_finite() || cfg.fb() <= 0.0 {
    return Err(StreamingShapeError::InvalidFb { value: cfg.fb() }.into());
  }
  if cfg.max_iters() == 0 {
    return Err(StreamingShapeError::ZeroMaxIters.into());
  }
  if cfg.max_iters() > crate::cluster::vbx::MAX_ITERS_CAP {
    return Err(
      StreamingShapeError::MaxItersExceedsCap {
        got: cfg.max_iters(),
        cap: crate::cluster::vbx::MAX_ITERS_CAP,
      }
      .into(),
    );
  }

  let num_chunks = if samples.len() <= win {
    1
  } else {
    (samples.len() - win).div_ceil(step) + 1
  };

  let mut padded_chunk = vec![0.0_f32; win];
  // Spill-back per-range tensors: a single voice range that runs
  // for hours would otherwise OOM the heap. See
  // `OwnedDiarizationPipeline::run` for the same pattern.
  let segs_len = num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK;
  let mut segmentations = crate::ops::spill::SpillBytesMut::<f64>::zeros(
    segs_len,
    options.diarization().spill_options(),
  )?;
  {
    let segs = segmentations.as_mut_slice();

    // ── Stage 1: chunked sliding-window segmentation ───────────────
    for c in 0..num_chunks {
      let chunk_start = c * step;
      padded_chunk.fill(0.0);
      let end = (chunk_start + win).min(samples.len());
      let lo = chunk_start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      let logits = seg_model
        .infer(&padded_chunk)
        .map_err(|e| StreamingError::Segment(format!("{e}")))?;
      for f in 0..FRAMES_PER_WINDOW {
        let mut row = [0.0_f32; POWERSET_CLASSES];
        for k in 0..POWERSET_CLASSES {
          row[k] = logits[f * POWERSET_CLASSES + k];
        }
        let probs = softmax_row(&row);
        // Pyannote's `to_multilabel(soft=False)` — see the long
        // comment in `crate::offline::owned::OwnedDiarizationPipeline
        // ::run` stage 1 for the rationale.
        let speakers = powerset_to_speakers_hard(&probs);
        for s in 0..SLOTS_PER_CHUNK {
          segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = speakers[s] as f64;
        }
      }
    }
  }

  // ── Stage 2: per-(chunk, slot) masked embedding ────────────────
  let emb_len = num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM;
  let mut raw_embeddings =
    crate::ops::spill::SpillBytesMut::<f32>::zeros(emb_len, options.diarization().spill_options())?;
  {
    let segs = segmentations.as_mut_slice();
    let embs = raw_embeddings.as_mut_slice();

    for c in 0..num_chunks {
      let chunk_start = c * step;
      padded_chunk.fill(0.0);
      let end = (chunk_start + win).min(samples.len());
      let lo = chunk_start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      for s in 0..SLOTS_PER_CHUNK {
        let mut frame_mask = [false; FRAMES_PER_WINDOW];
        let mut any_active = false;
        for f in 0..FRAMES_PER_WINDOW {
          let active =
            segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] >= cfg.onset() as f64;
          frame_mask[f] = active;
          any_active |= active;
        }
        if !any_active {
          for f in 0..FRAMES_PER_WINDOW {
            segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
          }
          continue;
        }

        let raw = match embed_model.embed_chunk_with_frame_mask(&padded_chunk, &frame_mask) {
          Ok(v) => v,
          Err(crate::embed::Error::InvalidClip { .. })
          | Err(crate::embed::Error::DegenerateEmbedding) => {
            for f in 0..FRAMES_PER_WINDOW {
              segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
            }
            continue;
          }
          Err(e) => return Err(StreamingError::Embed(format!("{e}"))),
        };
        // Reject non-finite embedding output as a hard error. Mirrors
        // `offline::owned`'s split: NaN/inf is upstream corruption that
        // must surface, not get silently drop-listed as "inactive
        // speaker" alongside legitimate low-norm vectors.
        if raw.iter().any(|v| !v.is_finite()) {
          return Err(StreamingError::Embed(format!(
            "{}",
            crate::embed::Error::NonFiniteOutput
          )));
        }
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
  }

  // ── Stage 3: count tensor (local to this range) ────────────────
  let chunk_duration_s = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
  let chunk_step_s = cfg.step_samples() as f64 / SAMPLE_RATE_HZ as f64;
  let chunks_sw_local = SlidingWindow::new(0.0, chunk_duration_s, chunk_step_s);
  let frames_sw_template =
    SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S);
  // Use the fallible variant: a malformed `onset` (NaN/inf via the
  // public `with_onset` builder) would panic the infallible wrapper
  // at `try_count_pyannote.expect(...)`. Surface it as a typed
  // `StreamingError::Aggregate` so untrusted config can never crash.
  let (count, frames_sw_local) = try_count_pyannote(
    segmentations.as_slice(),
    num_chunks,
    FRAMES_PER_WINDOW,
    SLOTS_PER_CHUNK,
    cfg.onset() as f64,
    chunks_sw_local,
    frames_sw_template,
    options.diarization().spill_options(),
  )?
  .into_parts();

  Ok(RangeEmbeddings::from_validated(
    abs_start_sample,
    num_chunks,
    segmentations.as_slice().to_vec(),
    raw_embeddings.as_slice().to_vec(),
    count.to_vec(),
    chunks_sw_local,
    frames_sw_local,
  ))
}

/// Shared cluster-and-reconstruct core used by both
/// [`StreamingOfflineDiarizer::finalize`] and the public
/// [`crate::streaming::cluster_ranges`] entry point. Concatenates all
/// ranges' tensors, runs the single global `diarize_offline` pass, then
/// re-reconstructs per range on its local timing and re-anchors spans to
/// the original stream. Deterministic — `diarize_offline` has no RNG, so
/// identical inputs produce identical spans.
pub(crate) fn cluster_ranges_inner(
  ranges: &[RangeEmbeddings],
  options: &StreamingOfflineOptions,
  plda: &PldaTransform,
) -> Result<Arc<[DiarizedSpan]>, StreamingError> {
  if ranges.is_empty() {
    return Ok(Arc::from([] as [DiarizedSpan; 0]));
  }
  let total_chunks: usize = ranges.iter().map(|r| r.num_chunks()).sum();
  if total_chunks == 0 {
    return Err(StreamingShapeError::AllRangesEmpty.into());
  }
  let spill = options.diarization().spill_options();

  // ── 1. Concatenate per-range tensors ───────────────────────────
  //
  // The concatenated tensors are the dominant memory footprint at
  // multi-hour scale: `all_segs` ≈ 50 MB / hour, plus
  // `all_emb` ≈ 11 MB / hour. Both cross the 64 MiB default
  // threshold past ~5 h of accumulated voice. Spill-back so we
  // don't OOM the heap.
  let total_segs_len = total_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK;
  let total_emb_len = total_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM;
  let mut all_segs = crate::ops::spill::SpillBytesMut::<f64>::zeros(total_segs_len, spill)?;
  let mut all_emb = crate::ops::spill::SpillBytesMut::<f32>::zeros(total_emb_len, spill)?;
  {
    let segs = all_segs.as_mut_slice();
    let embs = all_emb.as_mut_slice();
    let mut s_off = 0;
    let mut e_off = 0;
    for r in ranges {
      let s_n = r.segmentations().len();
      segs[s_off..s_off + s_n].copy_from_slice(r.segmentations());
      s_off += s_n;
      let e_n = r.raw_embeddings().len();
      embs[e_off..e_off + e_n].copy_from_slice(r.raw_embeddings());
      e_off += e_n;
    }
  }

  // ── 2. Concatenate count tensors (per-range adjacent in output) ─
  let total_output_frames: usize = ranges.iter().map(|r| r.count().len()).sum();
  let mut all_count = crate::ops::spill::SpillBytesMut::<u8>::zeros(total_output_frames, spill)?;
  {
    let buf = all_count.as_mut_slice();
    let mut off = 0;
    for r in ranges {
      let n = r.count().len();
      buf[off..off + n].copy_from_slice(r.count());
      off += n;
    }
  }

  // ── 3. Run global cluster_vbx via diarize_offline ──────────────
  //
  // `diarize_offline`'s reconstruct stage uses `chunks_sw` /
  // `frames_sw` to map per-chunk frames onto the global output
  // grid. With our concatenated chunks (which have non-uniform
  // gaps in absolute time), this global reconstruct would emit
  // garbage timings. So we ignore its reconstruct output and
  // re-reconstruct per range below.
  let cfg = options.diarization();
  let chunks_sw_global = ranges[0].chunks_sw();
  let frames_sw_global = ranges[0].frames_sw();
  let input = OfflineInput::new(
    all_emb.as_slice(),
    total_chunks,
    SLOTS_PER_CHUNK,
    all_segs.as_slice(),
    FRAMES_PER_WINDOW,
    all_count.as_slice(),
    total_output_frames,
    chunks_sw_global,
    frames_sw_global,
    plda,
  )
  .with_threshold(cfg.threshold())
  .with_fa(cfg.fa())
  .with_fb(cfg.fb())
  .with_max_iters(cfg.max_iters())
  .with_min_duration_off(cfg.min_duration_off())
  .with_smoothing_epsilon(cfg.smoothing_epsilon())
  .with_spill_options(spill.clone());
  let offline_out = diarize_offline(&input)?;
  let hard_clusters = offline_out.hard_clusters();
  debug_assert_eq!(hard_clusters.len(), total_chunks);

  // ── 4. Per-range reconstruct → spans → original timeline ───────
  //
  // `reconstruct` sizes its output grid as `(num_output_frames,
  // num_clusters_local)` where `num_clusters_local =
  // max(max(hard_clusters_slice) + 1, max(count_slice), 1)`. We
  // recompute it the same way so `discrete_to_spans`'s shape
  // assertion holds. Span cluster ids are the global hard-cluster
  // ids regardless of `num_clusters_local`, so cross-range identity
  // is preserved automatically.
  let mut all_spans: Vec<DiarizedSpan> = Vec::new();
  let sr = SAMPLE_RATE_HZ as f64;
  let mut chunk_offset = 0usize;
  for r in ranges {
    let hc_slice = &hard_clusters[chunk_offset..chunk_offset + r.num_chunks()];
    chunk_offset += r.num_chunks();

    let recon_input = ReconstructInput::new(
      r.segmentations(),
      r.num_chunks(),
      FRAMES_PER_WINDOW,
      SLOTS_PER_CHUNK,
      hc_slice,
      r.count(),
      r.count().len(),
      r.chunks_sw(),
      r.frames_sw(),
    )
    .with_smoothing_epsilon(cfg.smoothing_epsilon())
    .with_spill_options(spill.clone());
    let discrete = reconstruct_grid(&recon_input)?;

    let max_cluster_local = hc_slice
      .iter()
      .flat_map(|row| row.iter())
      .copied()
      .max()
      .unwrap_or(-1);
    let max_count_local = r.count().iter().copied().max().unwrap_or(0) as usize;
    let num_clusters_local = if max_cluster_local < 0 {
      // No assigned clusters → reconstruct returns a 1D
      // `num_output_frames`-length zero vector (see
      // `reconstruct::algo::reconstruct` early-out at
      // `max_cluster < 0`). `discrete_to_spans` would then assert
      // on `grid.len() == num_output_frames * num_clusters`, so
      // skip the call entirely.
      debug_assert_eq!(discrete.len(), r.count().len());
      continue;
    } else {
      ((max_cluster_local + 1) as usize).max(max_count_local.max(1))
    };

    let local_spans: Vec<RttmSpan> = discrete_to_spans(
      discrete.as_slice(),
      r.count().len(),
      num_clusters_local,
      r.frames_sw(),
      cfg.min_duration_off(),
    );

    for span in local_spans {
      let start_off_samples = (span.start() * sr).max(0.0) as u64;
      let dur_samples = (span.duration() * sr).max(0.0) as u64;
      all_spans.push(DiarizedSpan::new(
        r.abs_start_sample().saturating_add(start_off_samples),
        r.abs_start_sample()
          .saturating_add(start_off_samples)
          .saturating_add(dur_samples),
        span.cluster() as u32,
      ));
    }
  }

  // Sort by start time so callers can stream the output in order.
  all_spans.sort_by_key(|s| s.start_sample());
  // One-time `Vec`→`Arc<[T]>` copy at the boundary. `all_spans` is
  // built by `Vec::push` because span count is unknown a-priori
  // (it depends on per-range `discrete_to_spans` output); converting
  // to `Arc<[DiarizedSpan]>` lets downstream consumers fan out
  // cheaply via `Arc::clone`.
  Ok(Arc::from(all_spans))
}

#[cfg(test)]
mod refactor_tests {
  use super::*;
  use crate::{reconstruct::SlidingWindow, segment::FRAMES_PER_WINDOW};

  const SLOTS: usize = 3;

  /// `cluster_ranges_inner` on a single all-silent range returns Ok
  /// with no spans (the `num_train < 2` pyannote fast path yields
  /// hard_clusters all-zero, and an all-zero range reconstructs to no
  /// emitted speakers). Proves the extracted helper is callable and the
  /// carrier→OfflineInput wiring holds.
  #[test]
  fn cluster_ranges_inner_empty_range_is_ok_empty() {
    let plda = PldaTransform::new().expect("plda");
    let opts = StreamingOfflineOptions::new();
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS];
    let emb = vec![0.0_f32; num_chunks * SLOTS * EMBEDDING_DIM];
    // One chunk's worth of output frames, all count 0 (no active
    // speakers). `diarize_offline`'s reconstruct stage requires
    // `num_output_frames >= FRAMES_PER_WINDOW` for a single chunk, so a
    // 1-frame count is rejected before the clustering fast path; size
    // it to the chunk's frame count to exercise the all-silent path.
    let count = vec![0_u8; FRAMES_PER_WINDOW];
    let chunk_dur = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    let chunks_sw = SlidingWindow::new(0.0, chunk_dur, 1.0);
    let frames_sw = SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S);
    let range =
      crate::streaming::RangeEmbeddings::new(0, num_chunks, seg, emb, count, chunks_sw, frames_sw)
        .expect("carrier");
    let spans = cluster_ranges_inner(&[range], &opts, &plda).expect("cluster ok");
    assert_eq!(spans.len(), 0, "all-silent range yields no spans");
  }
}

#[cfg(test)]
mod options_tests {
  use super::*;

  /// Regression: `StreamingOfflineOptions` must use ONE source of
  /// truth for spill configuration. The previous design carried a
  /// duplicate top-level field that `with_diarization` silently
  /// ignored, so a caller building
  /// `StreamingOfflineOptions::default().with_diarization(
  ///   OwnedPipelineOptions::new().with_spill_options(custom))`
  /// would get the streaming default instead of `custom`.
  ///
  /// This test pins the corrected plumbing in place: the streaming
  /// view of `spill_options` must equal the inner diarization's
  /// `spill_options`, regardless of which builder set the value.
  #[test]
  fn with_diarization_carries_spill_options_through() {
    let custom = SpillOptions::new()
      .with_threshold_bytes(7 * 1024 * 1024)
      .with_spill_dir(Some("/var/tmp/dia-streaming".into()));

    // Path A: configure spill on the inner OwnedPipelineOptions and
    // pass it through `with_diarization`.
    let owned = OwnedPipelineOptions::new().with_spill_options(custom.clone());
    let streaming = StreamingOfflineOptions::default().with_diarization(owned);
    assert_eq!(streaming.spill_options(), &custom);
    assert_eq!(streaming.diarization().spill_options(), &custom);

    // Path B: configure spill via the streaming-level builder. The
    // value must land on the inner diarization (single source).
    let streaming = StreamingOfflineOptions::default().with_spill_options(custom.clone());
    assert_eq!(streaming.spill_options(), &custom);
    assert_eq!(streaming.diarization().spill_options(), &custom);
  }
}
