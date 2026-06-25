//! WeSpeaker ResNet34 embedding inference (spec §4.2).
//!
//! Multi-backend wrapper. The same `EmbedModel` API supports two
//! inference engines:
//!
//! - **ONNX (default)**: pulls in `ort` (ONNX Runtime). Fast, no
//!   dynamic linking. Constructed via [`EmbedModel::from_file`] /
//!   [`EmbedModel::from_memory`].
//! - **TorchScript** (feature `tch`): pulls in `tch` (libtorch C++
//!   bindings). Heavier (libtorch shared lib at runtime) but matches
//!   pyannote's PyTorch inference bit-exactly on hard cases — useful
//!   when ONNX→ORT diverges from PyTorch numerically. Constructed via
//!   [`EmbedModel::from_torchscript_file`].
//!
//! `Send` but **not** `Sync` (single-session-per-thread for both ort
//! and tch). Matches [`SegmentModel`](crate::segment::SegmentModel).
//!
//! The 256-d output of `embed_features` / `embed_features_batch` is
//! the **raw, un-normalized** embedding straight from the model.
//! Higher-level methods (`embed`, `embed_weighted`, `embed_masked`)
//! wrap this with the §5.1 sliding-window aggregation and L2-normalize
//! the result via [`Embedding::normalize_from`].

use core::time::Duration;
use std::path::Path;

use crate::embed::{
  Error,
  embedder::{embed_unweighted, embed_weighted_inner},
  options::{EMBEDDING_DIM, MIN_CLIP_SAMPLES, SAMPLE_RATE_HZ},
  types::{Embedding, EmbeddingMeta, EmbeddingResult},
};
// `FBANK_FRAMES` and `FBANK_NUM_MELS` are only consumed inside the
// `#[cfg(feature = "ort")]` backend. Importing them unconditionally
// triggers `-D warnings` on `--no-default-features --features tch`.
#[cfg(feature = "ort")]
use crate::embed::options::{FBANK_FRAMES, FBANK_NUM_MELS};

#[cfg(feature = "ort")]
use crate::embed::EmbedModelOptions;

// ── Backend trait ───────────────────────────────────────────────────

/// Backend-agnostic interface for embedding inference.
///
/// Implementations: `OrtBackend` (ONNX via ort), `TchBackend`
/// (TorchScript via tch). Both produce raw, un-normalized 256-d
/// embeddings.
///
/// Two methods cover the two pyannote use cases:
///
/// 1. [`embed_audio_clips_batch`] — bare audio clips (no mask). Used
///    by the high-level `embed`, `embed_weighted`, `embed_masked`
///    helpers for variable-length clips with sliding-window
///    aggregation.
/// 2. [`embed_chunk_with_frame_mask`] — pyannote-style 10s chunk +
///    589-frame segmentation mask. The mask is interpreted as
///    pooling weights: the WeSpeaker statistics-pooling layer
///    ignores frames with zero weight. This is the call that
///    [`crate::offline::OwnedDiarizationPipeline`] uses per
///    (chunk, slot) to extract a speaker-specific embedding from
///    a multi-speaker chunk.
///
/// `embed_audio_clips_batch` and `embed_chunk_with_frame_mask` differ
/// in how they handle the segmentation mask:
/// - The audio-clips path masks via audio zeroing (ORT) — the model
///   sees a "filtered" audio with silence in inactive frames.
/// - The frame-mask path uses pyannote's exact `forward(waveforms,
///   weights)` — the model sees the raw audio, and the pooling
///   layer integrates only over active frames. This matches
///   pyannote's bit-exact embedding extraction; the audio-zeroing
///   approach is an approximation that diverges by O(1) per element
///   on overlap-heavy chunks.
///
/// [`embed_audio_clips_batch`]: EmbedBackend::embed_audio_clips_batch
/// [`embed_chunk_with_frame_mask`]: EmbedBackend::embed_chunk_with_frame_mask
pub(crate) trait EmbedBackend: Send {
  /// Embed a batch of audio clips. Each clip must be exactly
  /// `EMBED_WINDOW_SAMPLES = 32_000` samples long (2 s @ 16 kHz);
  /// the Rust embedder zero-pads shorter clips before calling.
  fn embed_audio_clips_batch(
    &mut self,
    clips: &[&[f32]],
  ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error>;

  /// Embed a 10-second chunk (160_000 samples) using a 589-frame
  /// per-frame mask as pooling weights. Pyannote's exact embedding
  /// extraction call.
  ///
  /// The default implementation **gathers** samples in the
  /// mask-active frames (drops inactive regions entirely) and runs
  /// sliding-window inference on the gathered audio. This is what
  /// the ORT backend uses — the bundled ONNX model doesn't accept a
  /// weights input, so we fall back to the audio-zeroing
  /// approximation that was the previous behavior. The tch backend
  /// overrides to pass weights directly to the TorchScript module
  /// (bit-exact pyannote).
  fn embed_chunk_with_frame_mask(
    &mut self,
    chunk_samples: &[f32],
    frame_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    use crate::embed::options::{EMBED_WINDOW_SAMPLES, HOP_SAMPLES, MIN_CLIP_SAMPLES};
    let total_samples = chunk_samples.len();
    let frame_count = frame_mask.len();
    if frame_count == 0 {
      return Err(Error::InvalidClip {
        len: 0,
        min: MIN_CLIP_SAMPLES as usize,
      });
    }

    // Build per-sample mask from per-frame mask, then GATHER active
    // samples (matching the previous `embed_masked_raw` semantics).
    let samples_per_frame = total_samples as f64 / frame_count as f64;
    let mut sample_mask = vec![false; total_samples];
    for (f, &active) in frame_mask.iter().enumerate() {
      if !active {
        continue;
      }
      let s0 = (f as f64 * samples_per_frame).round() as usize;
      let s1 = ((f + 1) as f64 * samples_per_frame).round() as usize;
      let lo = s0.min(total_samples);
      let hi = s1.min(total_samples);
      for v in &mut sample_mask[lo..hi] {
        *v = true;
      }
    }
    let gathered: Vec<f32> = chunk_samples
      .iter()
      .zip(sample_mask.iter())
      .filter_map(|(&s, &keep)| keep.then_some(s))
      .collect();
    if gathered.len() < MIN_CLIP_SAMPLES as usize {
      return Err(Error::InvalidClip {
        len: gathered.len(),
        min: MIN_CLIP_SAMPLES as usize,
      });
    }

    let win = EMBED_WINDOW_SAMPLES as usize;
    let mut sum = [0.0_f32; EMBEDDING_DIM];
    if gathered.len() <= win {
      let mut padded = vec![0.0_f32; win];
      padded[..gathered.len()].copy_from_slice(&gathered);
      let raws = self.embed_audio_clips_batch(&[padded.as_slice()])?;
      sum.copy_from_slice(&raws[0]);
      return Ok(sum);
    }
    let hop = HOP_SAMPLES as usize;
    let k_max = (gathered.len() - win) / hop;
    let mut starts: Vec<usize> = (0..=k_max).map(|k| k * hop).collect();
    starts.push(gathered.len() - win);
    starts.sort_unstable();
    starts.dedup();
    let clips: Vec<&[f32]> = starts.iter().map(|&s| &gathered[s..s + win]).collect();
    let raws = self.embed_audio_clips_batch(&clips)?;
    for raw in &raws {
      for (s, r) in sum.iter_mut().zip(raw.iter()) {
        *s += r;
      }
    }
    Ok(sum)
  }
}

// ── ORT (ONNX) backend ──────────────────────────────────────────────

#[cfg(feature = "ort")]
mod ort_backend {
  use super::*;
  use crate::embed::fbank::compute_fbank;
  use ort::{session::Session as OrtSession, value::TensorRef};

  pub(crate) struct OrtBackend {
    pub(crate) session: OrtSession,
  }

  /// Number of segmentation frames per 10s chunk in pyannote's
  /// community-1 config. Used as the default `weights` length when
  /// the high-level audio-clips path doesn't carry a per-frame mask
  /// (we pass all-ones to disable weighted pooling).
  const SEG_FRAMES_PER_CHUNK: usize = 589;

  fn run_inference(
    session: &mut OrtSession,
    n: usize,
    fbank_flat: &[f32],
    fbank_frames: usize,
    weights_flat: &[f32],
    num_weights: usize,
  ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
    let outputs = session.run(ort::inputs![
      "fbank" => TensorRef::from_array_view((
        [n, fbank_frames, FBANK_NUM_MELS],
        fbank_flat,
      ))?,
      "weights" => TensorRef::from_array_view((
        [n, num_weights],
        weights_flat,
      ))?,
    ])?;
    // Guard against zero-output sessions before positional indexing.
    // `outputs[0]` panics at the FFI boundary (ort's Index<usize>
    // panics for OOB), which would turn a malformed-model error into
    // a library-caller panic. A graceful typed error is the right
    // contract.
    let first_output = outputs
      .values()
      .next()
      .ok_or(Error::MissingInferenceOutput)?;
    let (shape, data) = first_output.try_extract_tensor::<f32>()?;
    // Per-call shape contract: the ResNet's output must be exactly
    // `[n, EMBEDDING_DIM]`. Validating only the element count (`n *
    // EMBEDDING_DIM`) lets a custom/exporter-drifted model that emits
    // `[EMBEDDING_DIM, n]`, `[1, n * EMBEDDING_DIM]`, or any rank-1
    // flattening pass through. Each chunk would then be silently
    // mis-stridden into PLDA/clustering as if it were `[n, 256]` — the
    // resulting embeddings are corrupted but finite, so no downstream
    // validation catches it. We reject any shape divergence at the ABI
    // boundary before reading rows.
    let dims: &[i64] = shape.as_ref();
    let expected_n = n as i64;
    let expected_dim = EMBEDDING_DIM as i64;
    if dims.len() != 2 || dims[0] != expected_n || dims[1] != expected_dim {
      return Err(Error::InferenceOutputShape {
        got: dims.to_vec(),
        n,
        embedding_dim: EMBEDDING_DIM,
      });
    }
    let expected = n * EMBEDDING_DIM;
    if data.len() != expected {
      return Err(Error::InferenceShapeMismatch {
        expected,
        got: data.len(),
      });
    }
    Ok(
      data
        .chunks_exact(EMBEDDING_DIM)
        .take(n)
        .map(|chunk| {
          let mut row = [0.0f32; EMBEDDING_DIM];
          row.copy_from_slice(chunk);
          row
        })
        .collect(),
    )
  }

  impl super::EmbedBackend for OrtBackend {
    fn embed_audio_clips_batch(
      &mut self,
      clips: &[&[f32]],
    ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
      let n = clips.len();
      if n == 0 {
        return Ok(Vec::new());
      }
      // 2s clips → 200-frame fbank. Pass all-ones weights at the
      // same length so the resnet's pooling layer treats every frame
      // equally. Length matches `FBANK_FRAMES = 200`; pyannote's
      // pooling layer accepts mismatched fbank/weights lengths via
      // resampling but the trivial all-ones case avoids that path.
      let mut flat = Vec::with_capacity(n * FBANK_FRAMES * FBANK_NUM_MELS);
      for clip in clips.iter() {
        let fbank = compute_fbank(clip)?;
        for row in fbank.iter() {
          flat.extend_from_slice(row);
        }
      }
      let weights_flat = vec![1.0_f32; n * FBANK_FRAMES];
      run_inference(
        &mut self.session,
        n,
        &flat,
        FBANK_FRAMES,
        &weights_flat,
        FBANK_FRAMES,
      )
    }

    fn embed_chunk_with_frame_mask(
      &mut self,
      chunk_samples: &[f32],
      frame_mask: &[bool],
    ) -> Result<[f32; EMBEDDING_DIM], Error> {
      // Pyannote's exact embedding extraction: 10s chunk → fbank →
      // resnet+pool with frame_mask as weights → embedding. We
      // compute the fbank in Rust (kaldi-native-fbank) since
      // torchaudio's kaldi.fbank doesn't export to ONNX.
      use crate::embed::fbank::compute_full_fbank;
      let fbank = compute_full_fbank(chunk_samples)?;
      let num_frames = fbank.len() / FBANK_NUM_MELS;
      let weights_flat: Vec<f32> = frame_mask
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
      let _ = SEG_FRAMES_PER_CHUNK; // doc reference
      let mut out = run_inference(
        &mut self.session,
        1,
        &fbank,
        num_frames,
        &weights_flat,
        frame_mask.len(),
      )?;
      Ok(out.pop().expect("n=1 batch"))
    }
  }
}

// ── tch (TorchScript) backend ───────────────────────────────────────

#[cfg(feature = "tch")]
mod tch_backend {
  use super::*;
  use tch::{CModule, Device, Kind, Tensor};

  pub(crate) struct TchBackend {
    pub(crate) module: CModule,
  }

  impl super::EmbedBackend for TchBackend {
    fn embed_audio_clips_batch(
      &mut self,
      clips: &[&[f32]],
    ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
      // The TorchScript module signature is `forward(waveforms,
      // weights)`. For unweighted aggregation, pass an all-ones
      // weights tensor of the matching frame count. Pyannote's
      // segmentation model emits 589 frames per 10s window; for
      // 2s windows the resnet's pooling layer interpolates the
      // weights as needed. We pass `(seg_frames * window_secs / 10)`
      // weights — the wrapper was traced at 589, so we always pass
      // 589-element ones here for batch=1.
      let n = clips.len();
      if n == 0 {
        return Ok(Vec::new());
      }
      let mut out = Vec::with_capacity(n);
      for clip in clips.iter() {
        let len = clip.len();
        let input = Tensor::from_slice(clip).reshape([1, len as i64]);
        let weights = Tensor::ones([1, 589], (Kind::Float, Device::Cpu));
        let output = self.module.forward_ts(&[input, weights])?;
        let expected_shape = [1_i64, EMBEDDING_DIM as i64];
        if output.size() != expected_shape {
          return Err(Error::InferenceShapeMismatch {
            expected: EMBEDDING_DIM,
            got: output.numel(),
          });
        }
        let mut row = [0.0_f32; EMBEDDING_DIM];
        output.copy_data(&mut row, EMBEDDING_DIM);
        out.push(row);
      }
      Ok(out)
    }

    fn embed_chunk_with_frame_mask(
      &mut self,
      chunk_samples: &[f32],
      frame_mask: &[bool],
    ) -> Result<[f32; EMBEDDING_DIM], Error> {
      // Pyannote's exact embedding extraction: pass the full chunk
      // audio + the per-frame mask as pooling weights. The
      // TorchScript wrapper handles fbank + resnet + statistics
      // pooling internally; the weights drive the pooling layer
      // (active frames count, inactive frames are skipped).
      let len = chunk_samples.len();
      let input = Tensor::from_slice(chunk_samples).reshape([1, len as i64]);
      let weights_data: Vec<f32> = frame_mask
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
      let weights = Tensor::from_slice(&weights_data).reshape([1, frame_mask.len() as i64]);
      let output = self.module.forward_ts(&[input, weights])?;
      let expected_shape = [1_i64, EMBEDDING_DIM as i64];
      if output.size() != expected_shape {
        return Err(Error::InferenceShapeMismatch {
          expected: EMBEDDING_DIM,
          got: output.numel(),
        });
      }
      let mut row = [0.0_f32; EMBEDDING_DIM];
      output.copy_data(&mut row, EMBEDDING_DIM);
      Ok(row)
    }
  }
}

// ── EmbedModel — public wrapper ─────────────────────────────────────

/// WeSpeaker ResNet34 embedding inference. Holds one backend session
/// (ORT or tch). `Send`-only; one instance per worker thread.
pub struct EmbedModel {
  backend: Box<dyn EmbedBackend>,
  /// Basename of the file this model was loaded from, or `None` for an
  /// in-memory model. Feeds the dynamic version of [`Self::identity`].
  source: Option<String>,
}

// Manual `Debug` so callers can `dbg!()` / `{:?}`-format an
// `EmbedModel` (and propagate `Debug` through `Result<EmbedModel, _>`
// in `unwrap_err` diagnostics). The inner `EmbedBackend` trait object
// holds an ORT session / TorchScript module — neither has a useful
// `Debug` impl, so we just print the wrapper name.
impl core::fmt::Debug for EmbedModel {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("EmbedModel").finish_non_exhaustive()
  }
}

impl EmbedModel {
  /// Load the ONNX model from disk with default options.
  ///
  /// Available with the `ort` feature (on by default).
  ///
  /// **Embedding inference defaults to ORT-CPU dispatch** even when
  /// per-EP cargo features (e.g. `coreml`, `cuda`) are compiled in.
  /// This is intentional: ORT's CoreML EP is known to mistranslate
  /// the WeSpeaker ResNet34-LM graph and emit NaN/Inf on common
  /// inputs (independent of compute-unit / model-format /
  /// static-shape knobs); auto-registering CoreML for embed would
  /// cause a hard pipeline failure on most realistic clips. We have
  /// no parity coverage proving CUDA/TensorRT/DirectML/ROCm produce
  /// finite output on this model either, so dia treats CPU as the
  /// only known-safe default for embed and leaves the override
  /// explicit.
  ///
  /// Callers on a vetted EP host can opt in by passing providers
  /// explicitly:
  ///
  /// ```ignore
  /// # // ignored: requires the `cuda` cargo feature + a CUDA host
  /// # // AND prior parity validation on your model + EP combination
  /// # // (see warning below).
  /// use diarization::{
  ///   embed::{EmbedModel, EmbedModelOptions},
  ///   ep::CUDA,
  /// };
  /// let opts = EmbedModelOptions::default()
  ///   .with_providers(vec![CUDA::default().build()]);
  /// let mut emb = EmbedModel::from_file_with_options(
  ///   "wespeaker_resnet34_lm.onnx",
  ///   opts,
  /// )?;
  /// # Ok::<(), Box<dyn std::error::Error>>(())
  /// ```
  ///
  /// **Do NOT pass `CoreML` here.** ORT's CoreML EP miscompiles the
  /// WeSpeaker graph and produces NaN/Inf on most inputs across
  /// every CoreML compute-unit / model-format / static-shape
  /// combination — the `EmbedModel` finite-output validator will
  /// abort the pipeline. The example above uses CUDA because it is
  /// the most common request; CUDA / TensorRT / DirectML / ROCm /
  /// OpenVINO are NOT parity-validated by dia on this model. Run
  /// your own DER + finite-output check before committing an EP
  /// override into production.
  ///
  /// `SegmentModel::bundled()` does auto-register per-EP-compiled
  /// providers because the segmentation graph is CoreML-safe — see
  /// [`crate::segment::SegmentModel::bundled`] for that contract.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    Self::from_file_with_options(path, EmbedModelOptions::default())
  }

  /// Load the ONNX model from disk with custom options.
  ///
  /// Honors the caller's `opts` verbatim — including any execution
  /// providers explicitly set via
  /// [`EmbedModelOptions::with_providers`].
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_file_with_options<P: AsRef<Path>>(
    path: P,
    opts: EmbedModelOptions,
  ) -> Result<Self, Error> {
    use ort::session::Session as OrtSession;
    let path = path.as_ref();
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    Ok(Self {
      backend: Box::new(ort_backend::OrtBackend { session }),
      source: path.file_name().map(|n| n.to_string_lossy().into_owned()),
    })
  }

  /// Load the ONNX model from an in-memory byte buffer (default options).
  ///
  /// CPU dispatch — see [`Self::from_file`] for the rationale.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
    Self::from_memory_with_options(bytes, EmbedModelOptions::default())
  }

  /// Load the ONNX model from an in-memory byte buffer with custom options.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_memory_with_options(bytes: &[u8], opts: EmbedModelOptions) -> Result<Self, Error> {
    use ort::session::Session as OrtSession;
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder.commit_from_memory(bytes)?;
    Ok(Self {
      backend: Box::new(ort_backend::OrtBackend { session }),
      source: None,
    })
  }

  /// Load a TorchScript module from disk.
  ///
  /// Available with the `tch` feature. The module must accept a single
  /// `[N, FBANK_FRAMES, FBANK_NUM_MELS] = [N, 200, 80]` f32 tensor and
  /// return `[N, EMBEDDING_DIM] = [N, 256]` raw embeddings. See
  /// `scripts/export-wespeaker-torchscript.py` for the conversion from
  /// pyannote's PyTorch model.
  #[cfg(feature = "tch")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
  pub fn from_torchscript_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    let path = path.as_ref();
    let module = tch::CModule::load(path).map_err(|source| Error::LoadTorchScript {
      path: path.to_path_buf(),
      source,
    })?;
    Ok(Self {
      backend: Box::new(tch_backend::TchBackend { module }),
      source: path.file_name().map(|n| n.to_string_lossy().into_owned()),
    })
  }

  /// Basename of the file this model was loaded from, or `None` for an
  /// in-memory model. Feeds the dynamic version of [`Self::identity`].
  pub fn source_basename(&self) -> Option<&str> {
    self.source.as_deref()
  }

  /// The WeSpeaker embedding identity for provenance stamping. Family
  /// is [`crate::provenance::WESPEAKER_FAMILY`]; version is the loaded
  /// file basename when available, else the family name.
  pub fn identity(&self) -> crate::provenance::ModelIdentity {
    let version = self
      .source
      .clone()
      .unwrap_or_else(|| crate::provenance::WESPEAKER_FAMILY.to_string());
    crate::provenance::ModelIdentity::new(crate::provenance::WESPEAKER_FAMILY, version)
  }

  /// Embed a single 2-second audio clip. Returns the raw (un-normalized)
  /// 256-d embedding. `samples.len()` must be exactly
  /// `EMBED_WINDOW_SAMPLES = 32_000`; the high-level methods
  /// (`embed`, `embed_weighted`, `embed_masked`) handle padding and
  /// sliding-window aggregation automatically.
  pub(crate) fn embed_audio_clip(
    &mut self,
    samples: &[f32],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    let mut out = self.backend.embed_audio_clips_batch(&[samples])?;
    let raw = out
      .pop()
      .expect("backend returned a non-empty batch for n=1 input");
    if raw.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteOutput);
    }
    Ok(raw)
  }

  /// Batched audio-clip inference. Returns N raw (un-normalized)
  /// 256-d embeddings. An empty input returns `Vec::new()` without
  /// invoking the backend.
  pub(crate) fn embed_audio_clips_batch(
    &mut self,
    clips: &[&[f32]],
  ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
    let raws = self.backend.embed_audio_clips_batch(clips)?;
    // Centralized finite check at the EmbedModel boundary: neither the
    // ORT nor tch backend validates per-element finiteness on its own,
    // and the high-level `embed`/`embed_weighted`/`embed_masked`
    // helpers go straight from this batch into per-window axpy
    // accumulation. A NaN/inf raw row would propagate through the L2
    // normalize and feed PLDA/clustering as a "valid" speaker vector.
    for raw in raws.iter() {
      if raw.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteOutput);
      }
    }
    Ok(raws)
  }

  /// Pyannote-style speaker embedding for a 10-second chunk + per-
  /// frame segmentation mask. Returns the raw (un-normalized) 256-d
  /// embedding for the speaker whose activity is in `frame_mask`.
  ///
  /// Backend dispatches:
  /// - **ORT**: zeroes audio in inactive frames, runs sliding-window
  ///   inference, sums the per-window outputs. Approximate (the
  ///   bundled ONNX model doesn't accept a weights input).
  /// - **tch**: passes `(audio, frame_mask)` directly to the
  ///   TorchScript wrapper, which delegates to pyannote's
  ///   `WeSpeakerResNet34.forward(waveforms, weights=mask)` —
  ///   bit-exact pyannote.
  pub fn embed_chunk_with_frame_mask(
    &mut self,
    chunk_samples: &[f32],
    frame_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    // Centralized boundary validation that cannot be bypassed by a
    // backend's `embed_chunk_with_frame_mask` override. The `EmbedBackend`
    // trait provides default empty/short-mask guards via its
    // gather-then-window fallback, but the ORT and tch overrides skip
    // them and pass `frame_mask` straight to the model.
    //
    // Strict shape contract: the documented input is a pyannote-style
    // 10-second chunk (`WINDOW_SAMPLES = 160_000` samples @ 16 kHz)
    // with a 589-frame segmentation mask (`FRAMES_PER_WINDOW`). Both
    // backends feed `frame_mask.len()` directly as the pooling-layer
    // weights dimension and compute fbank from the full chunk. A
    // non-pyannote-sized chunk or off-by-one mask passes the model
    // and yields a finite-but-wrong 256-d embedding that silently
    // corrupts downstream PLDA/clustering.
    let expected_samples = crate::segment::WINDOW_SAMPLES as usize;
    if chunk_samples.len() != expected_samples {
      return Err(Error::ChunkSamplesShapeMismatch {
        expected: expected_samples,
        got: chunk_samples.len(),
      });
    }
    let expected_frames = crate::segment::FRAMES_PER_WINDOW;
    if frame_mask.len() != expected_frames {
      return Err(Error::FrameMaskShapeMismatch {
        expected: expected_frames,
        got: frame_mask.len(),
      });
    }
    // Empty/all-false mask → all-zero pooling weights →
    // division-by-zero in statistics pooling → NaN/inf row. Reject
    // before backend dispatch.
    if !frame_mask.iter().any(|&b| b) {
      return Err(Error::EmptyOrInactiveMask);
    }
    // Backend-independent finite-input guard. ORT routes through
    // `compute_full_fbank`, which itself rejects non-finite samples
    // upfront. The tch path builds a tensor directly from
    // `chunk_samples` and forwards it into TorchScript, where NaN
    // either propagates to a corrupted-but-finite embedding (passes
    // the post-output check) or surfaces as a backend-specific error.
    // Reject at the boundary so both backends behave identically.
    if chunk_samples.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    let raw = self
      .backend
      .embed_chunk_with_frame_mask(chunk_samples, frame_mask)?;
    if raw.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteOutput);
    }
    Ok(raw)
  }

  // ── High-level methods (spec §4.2) ────────────────────────────────────

  /// Compute the L2-normalized embedding of a clip (spec §5.1).
  ///
  /// For clips up to `EMBED_WINDOW_SAMPLES` (2 s @ 16 kHz), runs a single
  /// inference on the zero-padded clip. For longer clips, runs sliding-
  /// window inference and aggregates via per-window unweighted sum, then
  /// L2-normalizes the result.
  ///
  /// Returns [`Error::InvalidClip`] if `samples.len() < MIN_CLIP_SAMPLES`,
  /// or [`Error::DegenerateEmbedding`] if the aggregated sum has near-zero
  /// L2 norm (effectively unreachable on real audio; signals caller bug).
  pub fn embed(&mut self, samples: &[f32]) -> Result<EmbeddingResult, Error> {
    self.embed_with_meta(samples, EmbeddingMeta::default())
  }

  /// [`embed`](Self::embed) with explicit observability metadata
  /// ([`EmbeddingMeta`]). Returns a typed [`EmbeddingResult<A, T>`].
  pub fn embed_with_meta<A, T>(
    &mut self,
    samples: &[f32],
    meta: EmbeddingMeta<A, T>,
  ) -> Result<EmbeddingResult<A, T>, Error> {
    let (sum, windows_used) = embed_unweighted(self, samples)?;
    let embedding = Embedding::normalize_from(sum).ok_or(Error::DegenerateEmbedding)?;
    let duration = duration_from_samples(samples.len());
    Ok(EmbeddingResult::new(
      embedding,
      duration,
      windows_used,
      windows_used as f32,
      meta,
    ))
  }

  /// Voice-probability-weighted embedding (spec §5.2).
  ///
  /// Per-window weight = mean of `voice_probs[start..start + WINDOW]`.
  /// Aggregates per-window outputs as a weighted sum, then L2-normalizes.
  ///
  /// Errors:
  /// - [`Error::WeightShapeMismatch`] if `voice_probs.len() != samples.len()`.
  /// - [`Error::InvalidClip`] if `samples.len() < MIN_CLIP_SAMPLES`.
  /// - [`Error::AllSilent`] if every per-window weight is below `NORM_EPSILON`.
  /// - [`Error::DegenerateEmbedding`] if the weighted sum has near-zero norm.
  pub fn embed_weighted(
    &mut self,
    samples: &[f32],
    voice_probs: &[f32],
  ) -> Result<EmbeddingResult, Error> {
    self.embed_weighted_with_meta(samples, voice_probs, EmbeddingMeta::default())
  }

  /// [`embed_weighted`](Self::embed_weighted) with explicit observability metadata.
  pub fn embed_weighted_with_meta<A, T>(
    &mut self,
    samples: &[f32],
    voice_probs: &[f32],
    meta: EmbeddingMeta<A, T>,
  ) -> Result<EmbeddingResult<A, T>, Error> {
    if voice_probs.len() != samples.len() {
      return Err(Error::WeightShapeMismatch {
        samples_len: samples.len(),
        weights_len: voice_probs.len(),
      });
    }
    let (sum, windows_used, weight_sum) = embed_weighted_inner(self, samples, voice_probs)?;
    let embedding = Embedding::normalize_from(sum).ok_or(Error::DegenerateEmbedding)?;
    let duration = duration_from_samples(samples.len());
    Ok(EmbeddingResult::new(
      embedding,
      duration,
      windows_used,
      weight_sum,
      meta,
    ))
  }

  /// Mask-gated embedding: same windowing as
  /// [`embed`](Self::embed), but each fbank row is **zeroed out**
  /// where `keep_mask` is `false` for the corresponding sample window.
  /// Equivalent to running pyannote's masked-clip embedding.
  pub fn embed_masked(
    &mut self,
    samples: &[f32],
    keep_mask: &[bool],
  ) -> Result<EmbeddingResult, Error> {
    self.embed_masked_with_meta(samples, keep_mask, EmbeddingMeta::default())
  }

  /// Raw masked embedding — returns the un-normalized 256-d output.
  /// Useful for downstream PLDA stages that consume raw embeddings.
  ///
  /// Gathers samples where `keep_mask` is true (drops the rest), then
  /// runs the standard sliding-window pipeline on the gathered audio.
  pub fn embed_masked_raw(
    &mut self,
    samples: &[f32],
    keep_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    if keep_mask.len() != samples.len() {
      return Err(Error::MaskShapeMismatch {
        samples_len: samples.len(),
        mask_len: keep_mask.len(),
      });
    }
    // Validate the FULL input slice for non-finite values before
    // gathering. Without this check, a NaN/inf at a masked-out
    // position is dropped by the `filter_map` and never reaches the
    // finite guard in `embed_unweighted` — `Ok(_)` would silently
    // mask upstream buffer corruption that callers using
    // `Error::NonFiniteInput` as a quarantine signal need to see.
    if samples.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    let gathered: Vec<f32> = samples
      .iter()
      .zip(keep_mask.iter())
      .filter_map(|(&s, &keep)| keep.then_some(s))
      .collect();
    if gathered.len() < MIN_CLIP_SAMPLES as usize {
      return Err(Error::InvalidClip {
        len: gathered.len(),
        min: MIN_CLIP_SAMPLES as usize,
      });
    }
    let (sum, _windows_used) = embed_unweighted(self, &gathered)?;
    Ok(sum)
  }

  /// Mask-gated embedding with metadata.
  pub fn embed_masked_with_meta<A, T>(
    &mut self,
    samples: &[f32],
    keep_mask: &[bool],
    meta: EmbeddingMeta<A, T>,
  ) -> Result<EmbeddingResult<A, T>, Error> {
    if keep_mask.len() != samples.len() {
      return Err(Error::MaskShapeMismatch {
        samples_len: samples.len(),
        mask_len: keep_mask.len(),
      });
    }
    // Same full-slice finite check as `embed_masked_raw` — masked-out
    // NaN/inf would otherwise be filtered before `embed_unweighted`
    // sees them.
    if samples.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    let gathered: Vec<f32> = samples
      .iter()
      .zip(keep_mask.iter())
      .filter_map(|(&s, &keep)| keep.then_some(s))
      .collect();
    if gathered.len() < MIN_CLIP_SAMPLES as usize {
      return Err(Error::InvalidClip {
        len: gathered.len(),
        min: MIN_CLIP_SAMPLES as usize,
      });
    }
    let (sum, windows_used) = embed_unweighted(self, &gathered)?;
    let embedding = Embedding::normalize_from(sum).ok_or(Error::DegenerateEmbedding)?;
    let duration = duration_from_samples(gathered.len());
    Ok(EmbeddingResult::new(
      embedding,
      duration,
      windows_used,
      windows_used as f32,
      meta,
    ))
  }
}

#[inline]
fn duration_from_samples(samples: usize) -> Duration {
  Duration::from_secs_f64(samples as f64 / SAMPLE_RATE_HZ as f64)
}

#[cfg(all(test, feature = "ort"))]
mod tests {
  use super::*;
  use crate::embed::options::EMBED_WINDOW_SAMPLES;
  use std::path::PathBuf;

  fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("DIA_EMBED_MODEL_PATH") {
      return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx")
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn loads_and_infers_silent_clip() {
    let path = model_path();
    if !path.exists() {
      panic!(
        "model not found at {}; set DIA_EMBED_MODEL_PATH or download via models/",
        path.display()
      );
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.0f32; EMBED_WINDOW_SAMPLES as usize];
    let raw = model.embed_audio_clip(&samples).expect("infer silence");
    assert_eq!(raw.len(), EMBEDDING_DIM);
    assert!(raw.iter().all(|v| v.is_finite()));
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn batch_inference_matches_single() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let single = model.embed_audio_clip(&samples).expect("single");
    let batch = model.embed_audio_clips_batch(&[&samples]).expect("batch");
    assert_eq!(batch.len(), 1);
    assert_eq!(single, batch[0]);
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_round_trips_on_2s_clip() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let r = model.embed(&samples).expect("embed succeeds");
    let n_sq: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
    let norm = n_sq.sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
    assert_eq!(r.windows_used(), 1);
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_long_clip_uses_sliding_window() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; 2 * EMBED_WINDOW_SAMPLES as usize];
    let r = model.embed(&samples).expect("embed succeeds");
    assert_eq!(r.windows_used(), 3);
    let n_sq: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
    assert!((n_sq.sqrt() - 1.0).abs() < 1e-5);
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_weighted_rejects_mismatched_lengths() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let probs = vec![1.0f32; EMBED_WINDOW_SAMPLES as usize - 1];
    let r = model.embed_weighted(&samples, &probs);
    assert!(matches!(
      r,
      Err(Error::WeightShapeMismatch {
        samples_len: 32_000,
        weights_len: 31_999,
      })
    ));
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_masked_rejects_short_gathered_clip() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let mut mask = vec![false; EMBED_WINDOW_SAMPLES as usize];
    for m in mask.iter_mut().take(100) {
      *m = true;
    }
    let r = model.embed_masked(&samples, &mask);
    assert!(matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })));
  }

  /// `EmbedModel::embed_chunk_with_frame_mask` rejects a wrong-length
  /// `chunk_samples` slice at the public boundary BEFORE invoking the
  /// backend. The contract is `WINDOW_SAMPLES = 160_000` (pyannote 10s
  /// @ 16 kHz); a 2-second `EMBED_WINDOW_SAMPLES = 32_000` clip used
  /// for unweighted aggregation would otherwise produce a finite-but-
  /// wrong embedding (different fbank frame count, different pooling
  /// geometry).
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_chunk_with_frame_mask_rejects_wrong_chunk_length() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    // 2-second clip when the contract requires 10s.
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let mask = vec![true; crate::segment::FRAMES_PER_WINDOW];
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(
      matches!(r, Err(Error::ChunkSamplesShapeMismatch { .. })),
      "got {r:?}"
    );
  }

  /// `EmbedModel::embed_chunk_with_frame_mask` rejects an off-by-one /
  /// sample-level mask at the public boundary. Backends pass
  /// `frame_mask.len()` as the pooling-layer weights dim; a wrong-
  /// sized mask changes the integration window.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_chunk_with_frame_mask_rejects_wrong_mask_length() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; crate::segment::WINDOW_SAMPLES as usize];
    // 588 instead of 589 — off by one.
    let mask = vec![true; crate::segment::FRAMES_PER_WINDOW - 1];
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(
      matches!(r, Err(Error::FrameMaskShapeMismatch { .. })),
      "got {r:?}"
    );
  }

  /// `EmbedModel::embed_chunk_with_frame_mask` rejects empty
  /// `frame_mask` (caught by the shape check first).
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_chunk_with_frame_mask_rejects_empty_mask() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; crate::segment::WINDOW_SAMPLES as usize];
    let mask: Vec<bool> = Vec::new();
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(
      matches!(r, Err(Error::FrameMaskShapeMismatch { .. })),
      "got {r:?}"
    );
  }

  /// All-false `frame_mask` (correct length) produces all-zero pooling
  /// weights → division-by-zero in statistics pooling → NaN/inf raw
  /// vector downstream. We reject it at the EmbedModel boundary
  /// instead.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_chunk_with_frame_mask_rejects_all_false_mask() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; crate::segment::WINDOW_SAMPLES as usize];
    let mask = vec![false; crate::segment::FRAMES_PER_WINDOW];
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(matches!(r, Err(Error::EmptyOrInactiveMask)), "got {r:?}");
  }

  /// NaN/inf samples must be rejected at the public boundary, before
  /// backend dispatch. ORT routes through `compute_full_fbank` which
  /// rejects non-finite samples upfront, but tch builds a tensor
  /// directly from `chunk_samples` and lets TorchScript decide. The
  /// boundary guard makes both backends behave identically and
  /// prevents NaN-driven corruption from reaching the model.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_chunk_with_frame_mask_rejects_non_finite_samples() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let mut samples = vec![0.001f32; crate::segment::WINDOW_SAMPLES as usize];
    samples[42] = f32::NAN;
    let mask = vec![true; crate::segment::FRAMES_PER_WINDOW];
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(matches!(r, Err(Error::NonFiniteInput)), "got {r:?}");

    samples[42] = f32::INFINITY;
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(matches!(r, Err(Error::NonFiniteInput)), "got {r:?}");

    samples[42] = f32::NEG_INFINITY;
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(matches!(r, Err(Error::NonFiniteInput)), "got {r:?}");
  }

  /// `embed`/`embed_with_meta` (high-level entry points routed through
  /// `embed_unweighted`) must reject non-finite samples at the public
  /// boundary, before backend dispatch. Same threat shape as
  /// `embed_chunk_with_frame_mask`: ORT routes through fbank
  /// (rejects), tch builds a tensor directly (corrupted-but-finite
  /// embedding can pass post-output check).
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_rejects_non_finite_samples() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let mut samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    samples[100] = f32::NAN;
    let r = model.embed(&samples);
    assert!(matches!(r, Err(Error::NonFiniteInput)), "got {r:?}");

    samples[100] = f32::INFINITY;
    let r = model.embed(&samples);
    assert!(matches!(r, Err(Error::NonFiniteInput)), "got {r:?}");
  }

  /// `embed_weighted` must reject non-finite samples and voice_probs
  /// outside `[0.0, 1.0]` (including NaN/inf weights). NaN weights
  /// would bypass the `total_weight < NORM_EPSILON` "all-silent"
  /// guard since every comparison with NaN is false.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_weighted_rejects_invalid_inputs() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];

    // NaN weight.
    let mut probs = vec![0.5f32; samples.len()];
    probs[200] = f32::NAN;
    let r = model.embed_weighted(&samples, &probs);
    assert!(matches!(r, Err(Error::InvalidVoiceProbs)), "NaN: got {r:?}");

    // Negative weight.
    probs[200] = -0.1;
    let r = model.embed_weighted(&samples, &probs);
    assert!(matches!(r, Err(Error::InvalidVoiceProbs)), "neg: got {r:?}");

    // > 1 weight.
    probs[200] = 1.5;
    let r = model.embed_weighted(&samples, &probs);
    assert!(
      matches!(r, Err(Error::InvalidVoiceProbs)),
      "above 1: got {r:?}"
    );

    // +inf weight.
    probs[200] = f32::INFINITY;
    let r = model.embed_weighted(&samples, &probs);
    assert!(
      matches!(r, Err(Error::InvalidVoiceProbs)),
      "+inf: got {r:?}"
    );

    // Non-finite samples.
    let probs = vec![0.5f32; samples.len()];
    let mut bad_samples = samples.clone();
    bad_samples[100] = f32::NAN;
    let r = model.embed_weighted(&bad_samples, &probs);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "NaN sample: got {r:?}"
    );
  }

  /// `embed_weighted` must surface [`Error::AllSilent`] when every
  /// per-window weight is below `NORM_EPSILON`. Without this guard,
  /// the post-aggregation L2 normalize would either divide by ~0
  /// (`DegenerateEmbedding`) or pass a noise-floor unit vector
  /// downstream — both are wrong for "silent input".
  ///
  /// Two paths must be covered:
  ///   1. Single-window (`samples.len() <= EMBED_WINDOW_SAMPLES`):
  ///      the weight is `voice_probs.iter().sum() / len`.
  ///   2. Multi-window: the guard checks `total_weight` summed across
  ///      `plan_starts`.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_weighted_rejects_all_silent() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");

    // Single-window path: 2s clip, all-zero voice probabilities.
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let probs = vec![0.0f32; samples.len()];
    let r = model.embed_weighted(&samples, &probs);
    assert!(
      matches!(r, Err(Error::AllSilent)),
      "single-window all-zero weights must surface AllSilent, got {r:?}"
    );

    // Multi-window path: 6s clip → 3 sliding windows, all-zero weights.
    let samples = vec![0.001f32; (EMBED_WINDOW_SAMPLES as usize) * 3];
    let probs = vec![0.0f32; samples.len()];
    let r = model.embed_weighted(&samples, &probs);
    assert!(
      matches!(r, Err(Error::AllSilent)),
      "multi-window all-zero weights must surface AllSilent, got {r:?}"
    );

    // Sub-epsilon-but-nonzero weights (well below NORM_EPSILON = 1e-12
    // per `embed::options::NORM_EPSILON`) — still AllSilent. Picking
    // 1e-15 puts total_weight at ~5e-15 across 5 sliding windows,
    // safely below the threshold.
    let probs = vec![1e-15f32; samples.len()];
    let r = model.embed_weighted(&samples, &probs);
    assert!(
      matches!(r, Err(Error::AllSilent)),
      "sub-epsilon weights must surface AllSilent, got {r:?}"
    );
  }

  /// Both masked-embedding entry points (`embed_masked_raw` and
  /// `embed_masked_with_meta`) must scan the FULL input slice for
  /// non-finite values, not just the gathered subset. A NaN at a
  /// masked-out position is dropped by the `filter_map` and would
  /// silently bypass the finite guard in `embed_unweighted` —
  /// upstream buffer corruption must surface as
  /// `Error::NonFiniteInput`, not be masked away.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_masked_rejects_non_finite_in_masked_out_position() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    // Build a clip with NaN at index 5; mark index 5 as masked-OUT
    // (keep = false). The gathered subset has no NaN, but the input
    // slice does — the public API contract is "input must be
    // finite", so this must reject.
    let mut samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize * 3];
    samples[5] = f32::NAN;
    let mut mask = vec![true; samples.len()];
    mask[5] = false; // NaN is at a masked-out position.

    let r = model.embed_masked_raw(&samples, &mask);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "embed_masked_raw must reject NaN at masked-out position: got {r:?}"
    );

    let r = model.embed_masked(&samples, &mask);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "embed_masked must reject NaN at masked-out position: got {r:?}"
    );

    // And inf at a masked-out position.
    samples[5] = f32::INFINITY;
    let r = model.embed_masked_raw(&samples, &mask);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "embed_masked_raw must reject +inf at masked-out position: got {r:?}"
    );

    // Sanity: a clean clip with the SAME mask layout still succeeds
    // (proves the rejection is the input check, not the mask shape).
    let clean = vec![0.001f32; samples.len()];
    let _ok = model
      .embed_masked_raw(&clean, &mask)
      .expect("clean clip with same mask must succeed");
  }
}
