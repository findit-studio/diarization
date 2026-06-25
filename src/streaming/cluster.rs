//! `cluster_ranges`: the public from-raw clustering entry point for a
//! desktop `cluster` node.

use std::sync::Arc;

use crate::{
  plda::PldaTransform,
  streaming::{DiarizedSpan, RangeEmbeddings, StreamingError, StreamingOfflineOptions},
};

/// Cluster a sequence of [`RangeEmbeddings`] carriers (produced by a
/// [`crate::streaming::StreamingEmbedder`]) into globally-consistent
/// speaker spans on the original timeline.
///
/// This is the reduce-at-close half of the decomposed pipeline: the
/// `segment+embed` node streams `RangeEmbeddings` (audio dropped per
/// window), the `cluster` node buffers them, and at stream close calls
/// this function. It runs the single global pyannote `cluster_vbx` pass
/// ([`crate::offline::diarize_offline`] — PLDA → AHC → VBx → centroid →
/// Hungarian) over the union of all ranges, then reconstructs per-range
/// frame-level diarization and re-anchors spans by each range's absolute
/// start. PLDA is applied **internally** to the raw embeddings; the
/// carrier never holds normalized vectors.
///
/// Speaker ids are consistent across all ranges in the slice (the global
/// clustering establishes cross-range identity). Deterministic: identical
/// `ranges` produce identical spans (the offline path has no RNG).
///
/// # Errors
/// [`StreamingError::Shape`] if every range is empty;
/// [`StreamingError::Offline`] / [`StreamingError::Reconstruct`] /
/// [`StreamingError::Spill`] propagate from the clustering core.
pub fn cluster_ranges(
  ranges: &[RangeEmbeddings],
  plda: &PldaTransform,
  options: &StreamingOfflineOptions,
) -> Result<Arc<[DiarizedSpan]>, StreamingError> {
  crate::streaming::offline_diarizer::cluster_ranges_inner(ranges, options, plda)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cluster_ranges_empty_is_empty() {
    let plda = PldaTransform::new().expect("plda");
    let opts = StreamingOfflineOptions::new();
    let spans = cluster_ranges(&[], &plda, &opts).expect("ok on empty");
    assert_eq!(spans.len(), 0);
  }
}
