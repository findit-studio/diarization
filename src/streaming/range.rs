//! `RangeEmbeddings`: the public per-VAD-range carrier crossing the
//! `segment+embed → cluster` boundary.

use std::sync::Arc;

use crate::{
  aggregate::try_num_output_frames_pyannote, embed::EMBEDDING_DIM, ops::spill::SpillBytes,
  reconstruct::SlidingWindow, segment::FRAMES_PER_WINDOW,
};

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
  /// A flattened-length product (`num_chunks * FRAMES_PER_WINDOW *
  /// SLOTS_PER_CHUNK` or `num_chunks * SLOTS_PER_CHUNK *
  /// EMBEDDING_DIM`) overflowed `usize`. A `num_chunks` large enough
  /// to overflow the expected-length computation would, with plain
  /// `*`, wrap to a small value in release and let a mismatched buffer
  /// pass the length guard (or panic in debug); we reject it as a
  /// typed error at the boundary instead.
  #[error("expected flattened length overflows usize for num_chunks {num_chunks} ({what})")]
  ShapeOverflow {
    /// `num_chunks` that triggered the overflow.
    num_chunks: usize,
    /// Which product overflowed (`"segmentations"` /
    /// `"raw_embeddings"`).
    what: &'static str,
  },
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
  /// The supplied `chunks_sw` / `frames_sw` do not yield a valid
  /// pyannote output-frame geometry (zero/non-finite duration or step,
  /// or an output-frame count that overflows / exceeds the cap). The
  /// expected `count` length is derived from this geometry via the
  /// same `aggregate::try_num_output_frames_pyannote` helper the
  /// count-tensor stage uses, so an invalid window makes the length
  /// uncheckable; reject it here rather than letting the carrier
  /// through with an unvalidatable `count`.
  #[error(
    "invalid chunks_sw/frames_sw geometry: cannot derive the pyannote output-frame count \
     (zero/non-finite duration or step, or output-frame count overflow)"
  )]
  InvalidGeometry,
  /// `count.len()` does not equal the exact pyannote output-frame
  /// count derived from `num_chunks` + `chunks_sw` + `frames_sw`.
  ///
  /// `cluster_ranges_inner` treats `count.len()` as the authoritative
  /// `num_output_frames` for the range's reconstruct stage: a
  /// too-long `count` with positive trailing values makes reconstruct
  /// fabricate/extend spans beyond the audio, while a too-short one
  /// silently truncates trailing frames. Both must be rejected at the
  /// public boundary, not surfaced (or hidden) later.
  #[error(
    "count.len() {got} must equal the pyannote output-frame count for num_chunks/chunks_sw/frames_sw = {expected}"
  )]
  CountLenMismatch {
    /// Expected length (exact pyannote `num_output_frames`).
    expected: usize,
    /// Actual length.
    got: usize,
  },
}

/// Compute the exact pyannote `num_output_frames` for one range from
/// the same geometry the count-tensor aggregation uses, so the
/// `count` length can be validated at the carrier boundary against the
/// authoritative value `cluster_ranges_inner` will later assume.
///
/// Mirrors `aggregate::try_count_pyannote`'s internal derivation:
/// `chunk_duration = chunks_sw.duration()`, `chunk_step =
/// chunks_sw.step()`, `frame_step = frames_sw.step()`, fed to
/// [`try_num_output_frames_pyannote`]. Reusing that helper (rather than
/// reimplementing the `round(last_chunk_end / frame_step) + 1`
/// geometry) keeps the carrier's accepted `count` length in lockstep
/// with what `build_range` produces and what reconstruct consumes.
fn expected_count_len(
  num_chunks: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Result<usize, RangeShapeError> {
  try_num_output_frames_pyannote(
    num_chunks,
    chunks_sw.duration(),
    chunks_sw.step(),
    frames_sw.step(),
  )
  .map_err(|_| RangeShapeError::InvalidGeometry)
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
///
/// **Spill-backed storage:** `segmentations` and `raw_embeddings` are
/// held as [`SpillBytes`] rather than `Vec`, so a multi-hour single
/// voice range (or many accumulated ranges) that the internal
/// `build_range` path allocates file-backed stays file-backed all the
/// way through the carrier — no `to_vec` re-materializes the full
/// buffer onto the heap. The public [`new`](Self::new) constructor
/// (which takes owned `Vec`s built from raw model output) wraps those
/// `Vec`s as heap-backed `SpillBytes` zero-copy; that path is already
/// heap-resident, so nothing is lost. `count` is `Arc<[u8]>` — small
/// (one cell per output frame) and shared without copying.
#[derive(Debug, Clone)]
pub struct RangeEmbeddings {
  abs_start_sample: u64,
  num_chunks: usize,
  segmentations: SpillBytes<f64>,
  raw_embeddings: SpillBytes<f32>,
  count: Arc<[u8]>,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
}

impl RangeEmbeddings {
  /// Construct from a range's derived tensors, validating the flattened
  /// lengths against `num_chunks` and the `count` length against the
  /// exact pyannote output-frame geometry.
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
  ///   `aggregate::count_pyannote`). Its length must equal the exact
  ///   pyannote `num_output_frames` for `num_chunks` + `chunks_sw` +
  ///   `frames_sw` — `cluster_ranges_inner` treats `count.len()` as
  ///   the authoritative output-frame count, so a too-long or
  ///   too-short `count` is rejected here.
  /// - `chunks_sw` / `frames_sw`: local (range-start = 0) timing.
  ///
  /// The owned `Vec`s are wrapped as heap-backed [`SpillBytes`]
  /// zero-copy (no element copy): this constructor's inputs already
  /// live on the heap.
  ///
  /// # Errors
  /// [`RangeShapeError`] when `num_chunks == 0`, an expected length
  /// overflows `usize`, either flattened length mismatches, the
  /// sliding-window geometry is invalid, or `count.len()` does not
  /// equal the derived output-frame count.
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
    // Checked shape arithmetic: a `num_chunks` near `usize::MAX`
    // overflows the expected-length product. Plain `*` panics in debug
    // and wraps to a small value in release — the wrapped value could
    // coincidentally equal a (small) supplied buffer length and pass
    // the guard. Surface a typed overflow error instead.
    let expected_seg = num_chunks
      .checked_mul(FRAMES_PER_WINDOW)
      .and_then(|n| n.checked_mul(SLOTS_PER_CHUNK))
      .ok_or(RangeShapeError::ShapeOverflow {
        num_chunks,
        what: "segmentations",
      })?;
    if segmentations.len() != expected_seg {
      return Err(RangeShapeError::SegmentationsLenMismatch {
        expected: expected_seg,
        got: segmentations.len(),
      });
    }
    let expected_emb = num_chunks
      .checked_mul(SLOTS_PER_CHUNK)
      .and_then(|n| n.checked_mul(EMBEDDING_DIM))
      .ok_or(RangeShapeError::ShapeOverflow {
        num_chunks,
        what: "raw_embeddings",
      })?;
    if raw_embeddings.len() != expected_emb {
      return Err(RangeShapeError::RawEmbeddingsLenMismatch {
        expected: expected_emb,
        got: raw_embeddings.len(),
      });
    }
    // Validate `count.len()` against the EXACT pyannote output-frame
    // count derived from the same geometry the aggregation uses.
    // Replaces the prior empty-only check: that let a too-long count
    // (with positive trailing values) make reconstruct fabricate spans
    // and a too-short count truncate, neither caught at this boundary.
    let expected_count = expected_count_len(num_chunks, chunks_sw, frames_sw)?;
    if count.len() != expected_count {
      return Err(RangeShapeError::CountLenMismatch {
        expected: expected_count,
        got: count.len(),
      });
    }
    Ok(Self {
      abs_start_sample,
      num_chunks,
      segmentations: SpillBytes::from_vec(segmentations),
      raw_embeddings: SpillBytes::from_vec(raw_embeddings),
      count: Arc::from(count),
      chunks_sw,
      frames_sw,
    })
  }

  /// Construct from already-spill-backed buffers without re-validating
  /// shapes. Used by [`crate::streaming::offline_diarizer::build_range`],
  /// which builds the tensors itself (enforcing the length invariants
  /// by construction) as `SpillBytes` — so this path keeps long ranges
  /// file-backed instead of `to_vec`-ing them onto the heap.
  ///
  /// `count` arrives as `Arc<[u8]>` straight from
  /// `aggregate::count_pyannote`, avoiding a copy.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn from_spill_parts(
    abs_start_sample: u64,
    num_chunks: usize,
    segmentations: SpillBytes<f64>,
    raw_embeddings: SpillBytes<f32>,
    count: Arc<[u8]>,
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
    self.segmentations.as_slice()
  }
  /// Raw, unnormalized WeSpeaker vectors, flattened `[c][s][d]`. NOT
  /// L2-normalized — PLDA input.
  pub fn raw_embeddings(&self) -> &[f32] {
    self.raw_embeddings.as_slice()
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
  use crate::{
    reconstruct::SlidingWindow,
    segment::{PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S, SAMPLE_RATE_HZ, WINDOW_SAMPLES},
  };

  /// Chunk-level sliding window matching the bundled `build_range`
  /// geometry: 10 s window, 1 s step (community-1 default).
  fn chunks_sw() -> SlidingWindow {
    let chunk_dur = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    SlidingWindow::new(0.0, chunk_dur, 1.0)
  }

  /// Frame-level sliding window matching `build_range`'s frames_sw.
  fn frames_sw() -> SlidingWindow {
    SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S)
  }

  /// The exact pyannote output-frame count for `num_chunks` under the
  /// test geometry — the only `count` length `new` now accepts.
  fn count_len(num_chunks: usize) -> usize {
    expected_count_len(num_chunks, chunks_sw(), frames_sw()).expect("valid geometry")
  }

  #[test]
  fn new_accepts_consistent_shapes_and_exposes_accessors() {
    let num_chunks = 2;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let n_count = count_len(num_chunks);
    let count = vec![1_u8; n_count];
    let r = RangeEmbeddings::new(
      48_000,
      num_chunks,
      seg,
      emb,
      count,
      chunks_sw(),
      frames_sw(),
    )
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
    assert_eq!(r.count().len(), n_count);
  }

  #[test]
  fn new_rejects_mismatched_shapes() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb_bad = vec![0.0_f32; 999];
    let count = vec![1_u8; count_len(num_chunks)];
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb_bad,
      count.clone(),
      chunks_sw(),
      frames_sw(),
    );
    assert!(matches!(
      r,
      Err(RangeShapeError::RawEmbeddingsLenMismatch { .. })
    ));

    let r = RangeEmbeddings::new(
      0,
      0,
      Vec::new(),
      Vec::new(),
      count,
      chunks_sw(),
      frames_sw(),
    );
    assert!(matches!(r, Err(RangeShapeError::ZeroChunks)));
  }

  /// FINDING 1: a `count` longer than the exact pyannote output-frame
  /// count is rejected at construction — previously only an EMPTY
  /// count was caught, so a too-long count (with positive trailing
  /// values) reached `cluster_ranges_inner` and made reconstruct
  /// fabricate/extend spans.
  #[test]
  fn new_rejects_too_long_count() {
    let num_chunks = 2;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let expected = count_len(num_chunks);
    let count = vec![1_u8; expected + 1];
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, chunks_sw(), frames_sw());
    assert!(
      matches!(
        r,
        Err(RangeShapeError::CountLenMismatch { expected: e, got })
          if e == expected && got == expected + 1
      ),
      "got {r:?}"
    );
  }

  /// FINDING 1: a `count` shorter than the exact output-frame count is
  /// also rejected at the public boundary (it would otherwise truncate
  /// trailing frames, caught only later — or not at all).
  #[test]
  fn new_rejects_too_short_count() {
    let num_chunks = 2;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let expected = count_len(num_chunks);
    let count = vec![1_u8; expected - 1];
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, chunks_sw(), frames_sw());
    assert!(
      matches!(
        r,
        Err(RangeShapeError::CountLenMismatch { expected: e, got })
          if e == expected && got == expected - 1
      ),
      "got {r:?}"
    );
  }

  /// FINDING 1: an empty count is still rejected (now via the exact
  /// mismatch path, since the exact count is always >= 1 for a valid
  /// range).
  #[test]
  fn new_rejects_empty_count() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg,
      emb,
      Vec::new(),
      chunks_sw(),
      frames_sw(),
    );
    assert!(
      matches!(r, Err(RangeShapeError::CountLenMismatch { got: 0, .. })),
      "got {r:?}"
    );
  }

  /// FINDING 3: a `num_chunks` large enough to overflow the
  /// `num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK` product is
  /// rejected as a typed `ShapeOverflow`, not a debug panic / release
  /// wrap that could bypass the length guard. The segmentations buffer
  /// is empty here; the overflow check fires before any length compare.
  #[test]
  fn new_rejects_shape_overflow() {
    let huge = usize::MAX / 2;
    let r = RangeEmbeddings::new(
      0,
      huge,
      Vec::new(),
      Vec::new(),
      Vec::new(),
      chunks_sw(),
      frames_sw(),
    );
    assert!(
      matches!(r, Err(RangeShapeError::ShapeOverflow { num_chunks, .. }) if num_chunks == huge),
      "got {r:?}"
    );
  }

  /// Invalid sliding-window geometry (zero frame step) makes the
  /// output-frame count underivable; rejected as `InvalidGeometry`
  /// rather than panicking the geometry helper.
  #[test]
  fn new_rejects_invalid_geometry() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let bad_frames = SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, 0.0);
    let count = vec![1_u8; 8];
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, chunks_sw(), bad_frames);
    assert!(
      matches!(r, Err(RangeShapeError::InvalidGeometry)),
      "got {r:?}"
    );
  }
}
