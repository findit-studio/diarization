//! Online (streaming) speaker clustering — a greedy centroid matcher ported
//! from FluidAudio's `SpeakerManager`.
//!
//! # A different algorithm class from [`cluster_offline`](crate::cluster)
//! This is **not** the pyannote AHC→VBx pipeline and makes no attempt to be.
//! It assigns each embedding *as it arrives* to the nearest running centroid
//! (or spawns a new speaker), which agglomerative-then-VBx clustering
//! structurally cannot do. The trade-offs are deliberate and load-bearing:
//!
//! - **Order-dependent by design.** The same multiset of embeddings fed in a
//!   different order can yield different speaker assignments and even a
//!   different speaker count. This is inherent to greedy online assignment,
//!   not a bug (see the order-dependence example in the tests).
//! - **Not pyannote-parity.** Its correctness gate is *parity with the
//!   FluidAudio Swift `SpeakerManager`* on the same embedding sequences — an
//!   out-of-tree oracle — never DER against a pyannote reference. Do not score
//!   this engine's output with a diarization-error-rate corpus.
//! - **Cosine on raw WeSpeaker embeddings, no PLDA.** Matching is plain cosine
//!   distance between L2-normalized embeddings; the PLDA projection that the
//!   offline pipeline applies has no part here.
//!
//! This module is un-gated (no `ort`, no I/O): slices in, labels out. It is
//! the dia-side realization of the online engine in the speakerkit clustering
//! design of record (`docs/superpowers/specs/2026-07-16-clustering-backends-design.md`,
//! architecture point 3).
//!
//! # Usage
//! Construct an [`OnlineClusterer`] with [`OnlineClusterOptions`], then call
//! [`OnlineClusterer::assign`] once per `(embedding, speech_duration)` in the
//! order they occur. Each call returns an [`Assignment`] — a reused speaker, a
//! newly created one, or a drop.
//!
//! ```
//! use diarization::cluster::online::{OnlineClusterer, OnlineClusterOptions, Assignment};
//! use diarization::embed::{EMBEDDING_DIM, Embedding};
//!
//! let mut basis = [0.0f32; EMBEDDING_DIM];
//! basis[0] = 1.0;
//! let a = Embedding::normalize_from(basis).unwrap();
//!
//! let mut clusterer = OnlineClusterer::new(OnlineClusterOptions::default());
//! // First sufficiently long segment seeds speaker 1.
//! assert_eq!(clusterer.assign(&a, 2.0), Assignment::New(1));
//! // The identical embedding (distance 0) reuses speaker 1.
//! assert_eq!(clusterer.assign(&a, 2.0), Assignment::Existing(1));
//! ```

#[cfg(test)]
pub(crate) mod algo;
#[cfg(not(test))]
mod algo;
mod options;

pub use algo::{Assignment, EMA_ALPHA, OnlineClusterer, RAW_HISTORY_CAP};
pub use options::{
  DEFAULT_EMBEDDING_THRESHOLD, DEFAULT_MIN_SPEECH_DURATION, DEFAULT_SPEAKER_THRESHOLD,
  OnlineClusterOptions,
};

#[cfg(test)]
mod tests;
