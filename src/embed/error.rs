//! Error type for `diarization::embed`.

#[cfg(feature = "ort")]
use std::path::PathBuf;

use thiserror::Error;

/// Errors returned by `diarization::embed` APIs.
///
/// Marked `#[non_exhaustive]` so callers must include a `_ =>` arm in
/// any `match`.
///
/// The backend-free numerical / boundary conditions (invalid clip, weight
/// and mask shape mismatch, non-finite input/output, shape drift, …) are
/// defined once in [`diaric::embed::Error`] and reach this type through
/// the [`Core`](Error::Core) wrapper (the `#[from]` conversion fires when
/// a `diaric` embedding helper such as `compute_fbank` propagates via
/// `?`). The variants below are specific to the ONNX/Torch model runners
/// that live in this crate.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
  /// A backend-free embedding error from the `diaric` core: invalid clip,
  /// weight/mask shape mismatch, out-of-range voice probabilities,
  /// all-silent aggregation, non-finite input/output, degenerate
  /// (zero-norm) embedding, or inference-output element-count drift.
  #[error(transparent)]
  Core(#[from] diaric::embed::Error),

  /// ONNX `session.run()` returned a zero-output `SessionOutputs`.
  /// Realistic causes are a malformed model export (no graph outputs)
  /// or ABI drift in `ort` itself. Without this typed error,
  /// `outputs[0]` would panic at the FFI boundary instead of
  /// surfacing as a recoverable error to library callers.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("inference returned no outputs (malformed model graph or ORT ABI drift)")]
  MissingInferenceOutput,

  /// ONNX inference output had an unexpected tensor shape (rank or per-axis size),
  /// even when the total element count would otherwise have matched. Catches
  /// silently corrupting layout drift like `[EMBEDDING_DIM, n]` or
  /// `[1, n * EMBEDDING_DIM]` from a custom/exporter-drifted model.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("inference output shape {got:?}, expected [{n}, {embedding_dim}]")]
  InferenceOutputShape {
    /// Actual shape from the ORT tensor.
    got: Vec<i64>,
    /// Batch dimension (clip count) the dispatcher passed in.
    n: usize,
    /// Per-row width the model is contracted to emit.
    embedding_dim: usize,
  },

  /// Load-time model shape verification failed.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("model {tensor} dims {got:?}, expected {expected:?}")]
  IncompatibleModel {
    /// Name of the tensor whose shape is wrong (e.g. `"input"` /
    /// `"output"`).
    tensor: &'static str,
    /// Shape the dia contract expects.
    expected: &'static [i64],
    /// Shape the loaded ONNX file actually declares.
    got: Vec<i64>,
  },

  /// Failed to load the ONNX model from disk.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("failed to load model from {path}: {source}", path = path.display())]
  LoadModel {
    /// Path to the ONNX file the loader attempted.
    path: PathBuf,
    /// Underlying error from `ort`.
    #[source]
    source: ort::Error,
  },

  /// Wrap an `ort::Error` from session/inference.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// Failed to load a TorchScript module from disk.
  #[cfg(feature = "tch")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
  #[error("failed to load TorchScript model from {path}: {source}", path = path.display())]
  LoadTorchScript {
    /// Path to the TorchScript module the loader attempted.
    path: std::path::PathBuf,
    /// Underlying error from `tch`.
    #[source]
    source: tch::TchError,
  },

  /// Wrap a `tch::TchError` from inference.
  #[cfg(feature = "tch")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
  #[error(transparent)]
  Tch(#[from] tch::TchError),
}
