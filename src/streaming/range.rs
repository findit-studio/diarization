//! `RangeEmbeddings`: the public per-VAD-range carrier crossing the
//! `segment+embed → cluster` boundary.

use crate::{embed::EMBEDDING_DIM, reconstruct::SlidingWindow, segment::FRAMES_PER_WINDOW};

/// Speaker slots per chunk (pyannote powerset = 3). Local copy for
/// module independence; equals [`crate::offline::SLOTS_PER_CHUNK`].
const SLOTS_PER_CHUNK: usize = 3;

/// Shape-violation reasons for [`RangeEmbeddings::new`].
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq)]
pub enum RangeShapeError {
  /// `num_chunks == 0` — a range must cover at least one segmentation
  /// window.
  #[error("num_chunks must be at least 1")]
  ZeroChunks,
  /// `segmentations.len()` does not equal
  /// `num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK`.
  #[error(
    "segmentations.len() {got} must equal num_chunks * {FRAMES_PER_WINDOW} * {SLOTS_PER_CHUNK} = {expected}"
  )]
  SegmentationsLenMismatch {
    /// Expected length.
    expected: usize,
    /// Actual length.
    got: usize,
  },
  /// `raw_embeddings.len()` does not equal
  /// `num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM`.
  #[error(
    "raw_embeddings.len() {got} must equal num_chunks * {SLOTS_PER_CHUNK} * {EMBEDDING_DIM} = {expected}"
  )]
  RawEmbeddingsLenMismatch {
    /// Expected length.
    expected: usize,
    /// Actual length.
    got: usize,
  },
  /// `count` is empty — every non-degenerate range yields at least one
  /// output frame.
  #[error("count must be non-empty")]
  EmptyCount,
}

/// One VAD voice range's derived diarization tensors, the unit a
/// `segment+embed` node emits and a `cluster` node consumes.
///
/// **Why the whole bundle and not just embeddings:** pyannote
/// clustering ([`crate::offline::diarize_offline`]) needs the
/// per-`(chunk, frame, slot)` segmentation activity to (a) select the
/// clean training subset (`filter_embeddings`), (b) mask inactive
/// speakers in the constrained assignment, and (c) reconstruct
/// frame-level diarization. The embeddings alone are insufficient — so
/// this carrier transports the segmentation activity, the count
/// tensor, the sliding-window timing, and the absolute start sample
/// alongside the raw 256-d vectors.
///
/// **Raw-not-normalized invariant:** `raw_embeddings` are the raw,
/// unnormalized WeSpeaker outputs (flattened `[chunk][slot][dim]`),
/// the distribution PLDA expects. They are NEVER L2-normalized at this
/// boundary; clustering applies PLDA internally.
#[derive(Debug, Clone)]
pub struct RangeEmbeddings {
  abs_start_sample: u64,
  num_chunks: usize,
  segmentations: Vec<f64>,
  raw_embeddings: Vec<f32>,
  count: Vec<u8>,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
}

impl RangeEmbeddings {
  /// Construct from a range's derived tensors, validating the flattened
  /// lengths against `num_chunks`.
  ///
  /// - `abs_start_sample`: absolute sample index where this range
  ///   starts in the original stream (used to re-anchor output spans).
  /// - `num_chunks`: number of 10 s segmentation chunks in this range.
  /// - `segmentations`: hard 0/1 activity, flattened
  ///   `[chunk][frame][slot]`, length
  ///   `num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK`.
  /// - `raw_embeddings`: raw WeSpeaker vectors, flattened
  ///   `[chunk][slot][dim]`, length
  ///   `num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM`.
  /// - `count`: per-output-frame speaker count (from
  ///   `aggregate::count_pyannote`).
  /// - `chunks_sw` / `frames_sw`: local (range-start = 0) timing.
  ///
  /// # Errors
  /// [`RangeShapeError`] when `num_chunks == 0`, either flattened
  /// length mismatches, or `count` is empty.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    abs_start_sample: u64,
    num_chunks: usize,
    segmentations: Vec<f64>,
    raw_embeddings: Vec<f32>,
    count: Vec<u8>,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
  ) -> Result<Self, RangeShapeError> {
    if num_chunks == 0 {
      return Err(RangeShapeError::ZeroChunks);
    }
    let expected_seg = num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK;
    if segmentations.len() != expected_seg {
      return Err(RangeShapeError::SegmentationsLenMismatch {
        expected: expected_seg,
        got: segmentations.len(),
      });
    }
    let expected_emb = num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM;
    if raw_embeddings.len() != expected_emb {
      return Err(RangeShapeError::RawEmbeddingsLenMismatch {
        expected: expected_emb,
        got: raw_embeddings.len(),
      });
    }
    if count.is_empty() {
      return Err(RangeShapeError::EmptyCount);
    }
    Ok(Self {
      abs_start_sample,
      num_chunks,
      segmentations,
      raw_embeddings,
      count,
      chunks_sw,
      frames_sw,
    })
  }

  /// Construct without re-validating shapes. Used by
  /// [`crate::streaming::StreamingOfflineDiarizer::push_voice_range`],
  /// which builds the tensors itself and has already enforced the
  /// length invariants by construction.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn from_validated(
    abs_start_sample: u64,
    num_chunks: usize,
    segmentations: Vec<f64>,
    raw_embeddings: Vec<f32>,
    count: Vec<u8>,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
  ) -> Self {
    Self {
      abs_start_sample,
      num_chunks,
      segmentations,
      raw_embeddings,
      count,
      chunks_sw,
      frames_sw,
    }
  }

  /// Absolute sample index where this range starts in the original stream.
  pub const fn abs_start_sample(&self) -> u64 {
    self.abs_start_sample
  }
  /// Number of segmentation chunks in this range.
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Hard 0/1 per-`(chunk, frame, slot)` activity, flattened `[c][f][s]`.
  pub fn segmentations(&self) -> &[f64] {
    &self.segmentations
  }
  /// Raw, unnormalized WeSpeaker vectors, flattened `[c][s][d]`. NOT
  /// L2-normalized — PLDA input.
  pub fn raw_embeddings(&self) -> &[f32] {
    &self.raw_embeddings
  }
  /// Per-output-frame instantaneous speaker count.
  pub fn count(&self) -> &[u8] {
    &self.count
  }
  /// Chunk-level (range-local) sliding window.
  pub const fn chunks_sw(&self) -> SlidingWindow {
    self.chunks_sw
  }
  /// Frame-level (range-local) sliding window.
  pub const fn frames_sw(&self) -> SlidingWindow {
    self.frames_sw
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn sw() -> SlidingWindow {
    SlidingWindow::new(0.0, 10.0, 1.0)
  }

  #[test]
  fn new_accepts_consistent_shapes_and_exposes_accessors() {
    let num_chunks = 2;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let count = vec![1_u8; 4];
    let r = RangeEmbeddings::new(48_000, num_chunks, seg, emb, count, sw(), sw())
      .expect("consistent shapes");
    assert_eq!(r.abs_start_sample(), 48_000);
    assert_eq!(r.num_chunks(), 2);
    assert_eq!(
      r.segmentations().len(),
      2 * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK
    );
    assert_eq!(
      r.raw_embeddings().len(),
      2 * SLOTS_PER_CHUNK * EMBEDDING_DIM
    );
    assert_eq!(r.count().len(), 4);
  }

  #[test]
  fn new_rejects_mismatched_shapes() {
    let seg = vec![0.0_f64; FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb_bad = vec![0.0_f32; 999];
    let count = vec![1_u8; 4];
    let r = RangeEmbeddings::new(0, 1, seg.clone(), emb_bad, count.clone(), sw(), sw());
    assert!(matches!(
      r,
      Err(RangeShapeError::RawEmbeddingsLenMismatch { .. })
    ));

    let r = RangeEmbeddings::new(0, 0, Vec::new(), Vec::new(), count, sw(), sw());
    assert!(matches!(r, Err(RangeShapeError::ZeroChunks)));
  }
}
