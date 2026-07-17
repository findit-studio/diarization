//! Error type for the segmentation module.

#[cfg(feature = "ort")]
use std::path::PathBuf;

use thiserror::Error;

/// All errors produced by `diarization::segment`.
///
/// The backend-free segmentation errors — [`SegmentOptions`] validation,
/// inference-scores shape/finiteness, and unknown-[`WindowId`] — are
/// defined in [`diaric::segment::Error`] and reach this type through the
/// [`Core`](Error::Core) wrapper. The variants below are specific to the
/// ONNX model runner in this crate.
///
/// [`SegmentOptions`]: crate::segment::SegmentOptions
/// [`WindowId`]: crate::segment::WindowId
#[derive(Debug, Error)]
pub enum Error {
  /// A backend-free segmentation error from the `diaric` core: invalid
  /// [`SegmentOptions`](crate::segment::SegmentOptions), inference-scores
  /// length mismatch or non-finite scores, or scores pushed for an
  /// unknown [`WindowId`](crate::segment::WindowId).
  #[error(transparent)]
  Core(#[from] diaric::segment::Error),

  /// `SegmentModel::infer` produced one or more non-finite logits
  /// (`NaN`, `+inf`, `-inf`) — e.g. from a degraded ONNX provider, a
  /// non-finite input sample, or numeric corruption upstream.
  ///
  /// This surfaces from the direct `SegmentModel::infer` entrypoint used
  /// by the owned and streaming offline paths (which do not own a
  /// `Segmenter`). Callers should treat this as a transient backend
  /// failure and retry, or surface the error.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("inference output contains non-finite logits (NaN / +inf / -inf)")]
  NonFiniteOutput,

  /// `SegmentModel::infer` was called with one or more non-finite
  /// input samples (`NaN`, `+inf`, `-inf`). Realistic upstream sources
  /// of bad samples are decoder bugs and corrupted audio buffers; we
  /// reject them at the boundary so they cannot poison the ONNX
  /// session and cascade into NaN logits / NaN-driven hard decisions.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("input samples contain non-finite values (NaN / +inf / -inf)")]
  NonFiniteInput,

  /// ONNX `session.run()` returned a zero-output `SessionOutputs`.
  /// Realistic causes are a malformed model export (no graph outputs)
  /// or ABI drift in `ort` itself. Without this typed error,
  /// `outputs[0]` would panic at the FFI boundary instead of
  /// surfacing as a recoverable error to library callers.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("inference returned no outputs (malformed model graph or ORT ABI drift)")]
  MissingInferenceOutput,

  /// A loaded ONNX model's input or output dimensions don't match what
  /// `diarization::segment` expects (`[*, 1, 160000]` for input, `[*, 589, 7]` for
  /// output, where `*` is a free batch dimension).
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("model {tensor} dims {got:?}, expected {expected:?}")]
  IncompatibleModel {
    /// Which tensor (`"input"` or `"output"`).
    tensor: &'static str,
    /// Expected dimension list. `-1` indicates a dynamic dimension.
    expected: &'static [i64],
    /// Actual dimensions reported by the loaded model.
    got: Vec<i64>,
  },

  /// The `ort::Session` failed to load the model file.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("failed to load model from {path}: {source}", path = path.display())]
  LoadModel {
    /// Path passed to `from_file`.
    path: PathBuf,
    /// Underlying ort error.
    #[source]
    source: ort::Error,
  },

  /// Generic ort runtime error from `SegmentModel::infer` or session ops.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error(transparent)]
  Ort(#[from] ort::Error),
}
