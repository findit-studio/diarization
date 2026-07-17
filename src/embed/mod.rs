//! Speaker fingerprint generation: WeSpeaker ResNet34 ONNX/Torch runner
//! over the `diaric` embedding value types + kaldi-compatible fbank.
//!
//! The backend-free half of the embedding pipeline — the [`Embedding`]
//! value types, the [`cosine_similarity`] helper, the kaldi-fbank feature
//! extractor ([`compute_fbank`], [`compute_full_fbank`]), and the
//! pyannote geometry constants — lives in `diaric::embed` and is
//! re-exported here. This crate adds the ONNX/Torch model runner
//! ([`EmbedModel`]) and its option builder ([`EmbedModelOptions`]).
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.
//! Layered API:
//! - High-level: `EmbedModel::embed`, `embed_weighted`, `embed_masked`
//! - Low-level: `compute_fbank`, `EmbedModel::embed_features`,
//!   `EmbedModel::embed_features_batch`

// `embedder` and `model` need to compile under either backend feature.
// `EmbedModel::from_torchscript_file` lives inside `model.rs` gated on
// `feature = "tch"`; if `model` is gated only on `ort`, a downstream
// build with `--no-default-features --features tch` cannot reach the
// TorchScript constructor at all.
#[cfg(any(feature = "ort", feature = "tch"))]
mod embedder;
mod error;
#[cfg(any(feature = "ort", feature = "tch"))]
mod model;
mod options;

pub use error::Error;
#[cfg(any(feature = "ort", feature = "tch"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "ort", feature = "tch"))))]
pub use model::EmbedModel;
// `EmbedModelOptions` wraps `ort::SessionBuilder` knobs; it has no
// counterpart on the tch backend, so it stays ORT-only.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use options::EmbedModelOptions;
// The pyannote geometry constants live in `diaric::embed`; re-exported
// through `options` (see `options.rs`) so the ort/tch runners can keep
// referencing `crate::embed::options::*`.
pub use options::{
  EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS, HOP_SAMPLES, MIN_CLIP_SAMPLES,
  NORM_EPSILON, SAMPLE_RATE_HZ,
};
// Backend-free embedding value types + the kaldi-fbank DSP, from `diaric`.
pub use diaric::embed::{
  Embedding, EmbeddingMeta, EmbeddingResult, compute_fbank, compute_full_fbank, cosine_similarity,
};

// Compile-time trait assertions. Catches a future field-type change that
// would silently regress Send/Sync auto-derive on the public types.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Embedding>();
  assert_send_sync::<EmbeddingMeta>();
  assert_send_sync::<EmbeddingResult>();
  assert_send_sync::<Error>();
};
