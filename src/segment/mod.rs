//! Speaker segmentation: the `diaric` sans-I/O state machine + this
//! crate's optional ONNX driver.
//!
//! The backend-free segmentation surface — the [`Segmenter`] sans-I/O
//! windowing/hysteresis state machine, powerset decoding, the
//! [`SegmentOptions`] config, and the value types — lives in
//! `diaric::segment` and is re-exported here. This crate adds the ONNX
//! segmentation model runner ([`SegmentModel`]).
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.

mod error;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
mod model;

pub use error::Error;

// Backend-free segmentation surface, re-exported from `diaric::segment`
// so `diarization::segment::*` keeps resolving.
pub use diaric::segment::{
  Action, Event, FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, PYANNOTE_FRAME_DURATION_S,
  PYANNOTE_FRAME_STEP_S, SAMPLE_RATE_HZ, SAMPLE_RATE_TB, SegmentOptions, Segmenter,
  SpeakerActivity, WINDOW_SAMPLES, WindowId, powerset,
};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use model::{SegmentModel, SegmentModelOptions, SegmenterExt};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::ep::ExecutionProviderDispatch;
/// Re-exported ort types used by [`SegmentModelOptions`] builders.
///
/// We re-export so callers can compose provider/optimization configurations
/// without importing `ort` directly. `GraphOptimizationLevel` mirrors what
/// silero exposes; `ExecutionProviderDispatch` is dia's deliberate
/// divergence — silero hard-codes provider selection, but dia exposes a
/// `with_providers` builder so we have to re-export the type it takes.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::session::builder::GraphOptimizationLevel;

// Compile-time trait assertions (spec §9). Catch a future field-type
// change that would silently regress Send/Sync auto-derive.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Segmenter>();

  #[cfg(feature = "ort")]
  fn assert_send<T: Send>() {}
  // SegmentModel: Send (auto-derived). The !Sync property rides on
  // ort::Session and is not asserted here without static_assertions.
  #[cfg(feature = "ort")]
  assert_send::<SegmentModel>();
};
