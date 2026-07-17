//! Offline (non-streaming) diarization.
//!
//! The batch pyannote `cluster_vbx` flow — PLDA projection → AHC initial
//! clustering → VBx EM → centroid computation → cosine cdist + constrained
//! Hungarian assignment → frame-level reconstruction → RTTM emission — is
//! backend-free and lives in `diaric::offline`; [`diarize_offline`],
//! [`OfflineInput`], [`OfflineOutput`], and [`ShapeError`] are re-exported
//! here. This module adds the ONNX audio entrypoint
//! ([`OwnedDiarizationPipeline`], under `feature = "ort"`) that runs the
//! segmentation + embedding models for you and forwards into
//! [`diarize_offline`].
//!
//! ## Where this fits
//!
//! - [`diarize_offline`] runs the full pyannote `community-1` clustering
//!   flow as a *batch* operation on already-computed segmentation +
//!   raw-embedding tensors.
//! - For audio-in / RTTM-out, pair with [`OwnedDiarizationPipeline`]
//!   (under `feature = "ort"`), which calls the segmentation +
//!   embedding ONNX models for you and forwards into
//!   [`diarize_offline`].
//! - For an *incremental* push-style entrypoint (good for VAD-driven
//!   streaming where you produce voice ranges over time but only need
//!   one final RTTM), see
//!   [`crate::streaming::StreamingOfflineDiarizer`].
//!
//! ## What [`diarize_offline`] accepts
//!
//! [`OfflineInput`] takes pre-computed (segmentation, raw embedding)
//! tensors. The caller is responsible for running segmentation +
//! embedding ONNX inference. Two production sources:
//!
//! 1. The captured pyannote fixtures (`tests/parity/fixtures/*/`).
//! 2. Custom ONNX inference using [`crate::segment::SegmentModel`] +
//!    [`crate::embed::EmbedModel`].

use thiserror::Error;

#[cfg(feature = "ort")]
mod owned;

#[cfg(all(test, feature = "ort"))]
mod owned_smoke_tests;

// The backend-free offline pipeline — `diarize_offline` plus its
// `OfflineInput` / `OfflineOutput` / `ShapeError` types — lives in
// `diaric::offline` and is re-exported here so `diarization::offline::*`
// keeps resolving.
pub use diaric::offline::{OfflineInput, OfflineOutput, ShapeError, diarize_offline};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use owned::{OwnedDiarizationPipeline, OwnedPipelineOptions, SLOTS_PER_CHUNK};

/// Reused by [`crate::streaming::offline_diarizer`] for the same
/// onset / min_duration_off / smoothing_epsilon validation it performs on
/// its [`OwnedPipelineOptions`]-derived config. `min_duration_off` and
/// `smoothing_epsilon` are validated against the single diaric-owned
/// authority ([`diaric::offline::check_min_duration_off`] /
/// [`check_smoothing_epsilon`](diaric::offline::check_smoothing_epsilon)),
/// so this crate's preflight cannot drift from what `diarize_offline`
/// enforces on the pure tensor path. `check_onset` stays local: the onset
/// knob only flows through the audio entrypoints, which diaric's tensor
/// path does not model.
#[cfg(feature = "ort")]
pub(crate) use diaric::offline::{check_min_duration_off, check_smoothing_epsilon};
#[cfg(feature = "ort")]
pub(crate) use owned::check_onset;

/// Errors from the offline diarization pipeline.
///
/// The backend-free pipeline errors — input shape/config validation
/// ([`ShapeError`]), PLDA projection, clustering/pipeline,
/// reconstruction/RTTM, frame aggregation, and spill-buffer allocation —
/// are defined in [`diaric::offline::Error`] and reach this type through
/// the [`Core`](Error::Core) wrapper. The two variants below carry ONNX
/// inference failures from the `OwnedDiarizationPipeline` audio
/// entrypoint.
#[derive(Debug, Error)]
pub enum Error {
  /// A backend-free offline pipeline error from the `diaric` core: input
  /// shape/config validation, PLDA projection, clustering/pipeline,
  /// reconstruction/RTTM, frame aggregation, or spill-buffer allocation.
  #[error(transparent)]
  Core(#[from] diaric::offline::Error),

  /// Propagated from segmentation ONNX inference inside the
  /// `OwnedDiarizationPipeline` audio entrypoint.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("offline: segment: {0}")]
  Segment(#[from] crate::segment::Error),

  /// Propagated from embedding ONNX inference inside the
  /// `OwnedDiarizationPipeline` audio entrypoint.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("offline: embed: {0}")]
  Embed(#[from] crate::embed::Error),
}

// Convenience `From` forwarders so the `OwnedDiarizationPipeline` audio
// entrypoint can construct / `?`-propagate the backend-free inner errors
// directly. Each funnels through `diaric::offline::Error`'s own `#[from]`
// conversion and then through [`Error::Core`], keeping the audio-path call
// sites clean without widening the wrapper beyond `Core`/`Segment`/`Embed`.
impl From<diaric::offline::ShapeError> for Error {
  fn from(e: diaric::offline::ShapeError) -> Self {
    Self::Core(e.into())
  }
}
impl From<diaric::pipeline::Error> for Error {
  fn from(e: diaric::pipeline::Error) -> Self {
    Self::Core(e.into())
  }
}
impl From<diaric::aggregate::Error> for Error {
  fn from(e: diaric::aggregate::Error) -> Self {
    Self::Core(e.into())
  }
}
impl From<diaric::spill::SpillError> for Error {
  fn from(e: diaric::spill::SpillError) -> Self {
    Self::Core(e.into())
  }
}
