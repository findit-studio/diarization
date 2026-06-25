//! Streaming voice-range-driven diarization.
//!
//! Architecture: caller drives a VAD (silero, webrtc, etc.) and pushes
//! one bounded voice range at a time via
//! [`StreamingOfflineDiarizer::push_voice_range`]. Each push runs the
//! heavy stages 1+2 (sliding-window segmentation + masked embedding)
//! eagerly and accumulates the derived tensors. At end-of-stream,
//! [`StreamingOfflineDiarizer::finalize`] runs a single global
//! pyannote-equivalent `cluster_vbx` pass over the union of
//! accumulated chunks and emits original-timeline spans with
//! consistent speaker ids across ranges.
//!
//! ## Accuracy
//!
//! Global clustering on the union of voice-range chunks is the same
//! algorithm pyannote runs on the full recording — the only audio
//! pyannote sees that we don't is the silence-gated portions, which
//! pyannote's segmentation model would mark inactive anyway. Cross-
//! range identity is established by AHC + VBx in PLDA space, not by a
//! cosine centroid bank — fixing the over- and under-merge failure
//! modes of the previous fingerprint architecture.
//!
//! ## When NOT to use this
//!
//! Latency is `finalize`-bound — the global clustering pass does not
//! emit spans incrementally. If you need *sub-range* latency (live
//! captioning, real-time speaker labels), this entrypoint is the
//! wrong shape — you would need an online clusterer that emits
//! spans as voice ranges close, which dia does not currently ship.

mod offline_diarizer;
mod range;

#[cfg(feature = "ort")]
mod embedder;

pub use offline_diarizer::{
  DiarizedSpan, StreamingError, StreamingOfflineDiarizer, StreamingOfflineOptions,
};
pub use range::{RangeEmbeddings, RangeShapeError};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use embedder::StreamingEmbedder;
