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
  /// A `SlidingWindow` start offset is not the local origin (`0.0`)
  /// the split protocol requires. `build_range` always emits both
  /// `chunks_sw` and `frames_sw` with `start == 0.0`, and the count
  /// derivation ([`aggregate::try_num_output_frames_pyannote`]) plus
  /// reconstruct's overlap-add (`chunk_start_time = chunks_sw.start +
  /// c * chunks_sw.step`, `closest_frame` subtracts `frames_sw.start`)
  /// both bake in that origin. A non-zero start is NOT reflected in the
  /// count length (the helper takes only `duration`/`step`), yet
  /// reconstruct honours it — so reconstruction emits every span offset
  /// by `start * SR` BEFORE `abs_start_sample` is added, silently
  /// shifting the whole range off its true timeline. Reject it at the
  /// boundary instead of letting a malformed or version-skewed payload
  /// reach clustering.
  #[error(
    "{which}.start ({got}) must be 0.0 (the split-protocol local origin); \
     a non-zero start shifts reconstructed spans off the timeline"
  )]
  NonZeroWindowStart {
    /// Which window violated the origin (`"chunks_sw"` /
    /// `"frames_sw"`).
    which: &'static str,
    /// The offending `start` value.
    got: f64,
  },
  /// A `SlidingWindow` duration or step is not a finite, strictly
  /// positive scalar. `build_range` derives each from the fixed
  /// community-1 geometry (`WINDOW_SAMPLES / SR`, `step_samples / SR`,
  /// and the pyannote frame constants), all `> 0`. A zero / negative /
  /// non-finite (NaN / ±inf) value would make the pyannote output-frame
  /// count underivable or non-finite (`try_count_pyannote` and
  /// `reconstruct` both reject the same conditions, but only AFTER
  /// clustering has run); reject it here so the `count` length is always
  /// derivable and reconstruct always aligns.
  #[error("{which}.{field} ({got}) must be a finite, strictly positive scalar")]
  NonPositiveWindowParameter {
    /// Which window (`"chunks_sw"` / `"frames_sw"`).
    which: &'static str,
    /// Which scalar (`"duration"` / `"step"`).
    field: &'static str,
    /// The offending value.
    got: f64,
  },
  /// The supplied `chunks_sw` / `frames_sw` pass the start/duration/step
  /// invariants but still do not yield a derivable pyannote
  /// output-frame count — the count overflows `usize` or exceeds the
  /// [`MAX_OUTPUT_FRAMES`](crate::aggregate::MAX_OUTPUT_FRAMES) cap (an
  /// extreme `chunk_duration / frame_step` whose product saturates the
  /// frame-count computation). The expected `count` length is derived
  /// from this geometry via the same
  /// `aggregate::try_num_output_frames_pyannote` helper the count-tensor
  /// stage uses, so an underivable count makes the length uncheckable;
  /// reject it here rather than letting the carrier through with an
  /// unvalidatable `count`.
  #[error(
    "chunks_sw/frames_sw geometry: cannot derive the pyannote output-frame count \
     (count overflows usize or exceeds MAX_OUTPUT_FRAMES)"
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

/// Validate the COMPLETE `SlidingWindow` geometry contract the split
/// protocol requires, BEFORE deriving the expected `count` length.
///
/// The count derivation ([`try_num_output_frames_pyannote`]) only
/// inspects `chunks_sw.duration`, `chunks_sw.step`, and
/// `frames_sw.step` — it drops both `start` offsets and never sees
/// `frames_sw.duration`. But reconstruct (`cluster_ranges_inner` ->
/// `reconstruct`) consumes ALL six scalars: its overlap-add computes
/// `chunk_start_time = chunks_sw.start + c * chunks_sw.step` and
/// `closest_frame(t) = round((t - frames_sw.start - frames_sw.duration
/// / 2) / frames_sw.step)`. So a window that satisfies the count helper
/// can still be geometrically inconsistent with the count it produced —
/// e.g. a non-zero `chunks_sw.start` leaves the count length unchanged
/// yet shifts every reconstructed span. Enforce the full contract here.
///
/// Invariants enforced (exactly what `build_range` guarantees and what
/// reconstruct / `try_count_pyannote` assume):
/// 1. `chunks_sw.start == 0.0` and `frames_sw.start == 0.0` — the
///    split-protocol local origin. The pyannote count formula and
///    reconstruct both bake this in; a non-zero start desynchronizes
///    them and offsets spans off the timeline.
/// 2. `chunks_sw.duration`, `chunks_sw.step`, `frames_sw.duration`,
///    `frames_sw.step` are each finite and strictly positive. Zero /
///    negative / NaN / ±inf make the output-frame count underivable or
///    non-finite (and reconstruct rejects them too, but only after
///    clustering has run).
///
/// `frames_sw.duration` is checked here even though the count helper
/// ignores it, because reconstruct's `center_offset = 0.5 *
/// frames_sw.duration` does use it: a non-finite frame duration would
/// drive `closest_frame` to a non-finite index downstream.
fn validate_geometry(
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Result<(), RangeShapeError> {
  for (which, w) in [("chunks_sw", chunks_sw), ("frames_sw", frames_sw)] {
    if w.start() != 0.0 {
      return Err(RangeShapeError::NonZeroWindowStart {
        which,
        got: w.start(),
      });
    }
    for (field, v) in [("duration", w.duration()), ("step", w.step())] {
      if !v.is_finite() || v <= 0.0 {
        return Err(RangeShapeError::NonPositiveWindowParameter {
          which,
          field,
          got: v,
        });
      }
    }
  }
  Ok(())
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
///
/// [`validate_geometry`] must run first: it guarantees the
/// start/duration/step invariants this helper's caller relies on, so
/// here a remaining error means only an overflow / above-cap
/// output-frame count (the `chunk_duration / frame_step` saturation
/// case), surfaced as [`RangeShapeError::InvalidGeometry`].
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
  /// overflows `usize`, either flattened length mismatches, a
  /// sliding-window `start` is non-zero
  /// ([`NonZeroWindowStart`](RangeShapeError::NonZeroWindowStart)), a
  /// window duration or step is non-finite / non-positive
  /// ([`NonPositiveWindowParameter`](RangeShapeError::NonPositiveWindowParameter)),
  /// the derived output-frame count overflows or exceeds the cap
  /// ([`InvalidGeometry`](RangeShapeError::InvalidGeometry)), or
  /// `count.len()` does not equal the derived output-frame count.
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
    // Validate the COMPLETE SlidingWindow geometry contract BEFORE
    // deriving the count length. The count helper only inspects
    // duration/step (not the `start` offsets, not `frames_sw.duration`),
    // so a non-zero `start` or a non-finite `frames_sw.duration` would
    // slip past the length check yet shift / corrupt every reconstructed
    // span. Reject the full protocol contract here so a malformed or
    // version-skewed payload can never reach clustering.
    validate_geometry(chunks_sw, frames_sw)?;
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

  /// FINDING 1: a non-zero `chunks_sw.start` is rejected at the public
  /// boundary. This is the core bug: the count helper drops `start`, so
  /// the length check passes, but reconstruct honours `start` and emits
  /// every span offset by `start * SR` BEFORE `abs_start_sample` is
  /// added — silently shifting the range off its timeline. The full
  /// geometry validation now catches it.
  #[test]
  fn new_rejects_non_zero_chunk_start() {
    let num_chunks = 2;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    // Otherwise-valid chunk window, but start = 1.0 (one second late).
    let chunk_dur = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    let bad_chunks = SlidingWindow::new(1.0, chunk_dur, 1.0);
    // Length still matches what the count helper derives (it ignores
    // start), so the count-length check would NOT catch this.
    let count = vec![1_u8; count_len(num_chunks)];
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, bad_chunks, frames_sw());
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonZeroWindowStart { which: "chunks_sw", got })
          if got == 1.0
      ),
      "got {r:?}"
    );
  }

  /// FINDING 1: a non-zero `frames_sw.start` is rejected. `closest_frame`
  /// subtracts `frames_sw.start`, so a non-zero value shifts every output
  /// frame index relative to the count tensor (derived as if start == 0).
  #[test]
  fn new_rejects_non_zero_frame_start() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let bad_frames = SlidingWindow::new(0.5, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S);
    let count = vec![1_u8; count_len(num_chunks)];
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, chunks_sw(), bad_frames);
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonZeroWindowStart { which: "frames_sw", got })
          if got == 0.5
      ),
      "got {r:?}"
    );
  }

  /// FINDING 1: a zero / negative `frames_sw.duration` is rejected as a
  /// non-positive window parameter. Reconstruct's `center_offset = 0.5 *
  /// frames_sw.duration` consumes it, so a non-positive duration corrupts
  /// frame placement; the count helper never inspects it, so only the
  /// full geometry validation catches it.
  #[test]
  fn new_rejects_non_positive_frame_duration() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let count = vec![1_u8; count_len(num_chunks)];
    // duration = 0.0 (zero), step valid.
    let zero_dur = SlidingWindow::new(0.0, 0.0, PYANNOTE_FRAME_STEP_S);
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb.clone(),
      count.clone(),
      chunks_sw(),
      zero_dur,
    );
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonPositiveWindowParameter {
          which: "frames_sw",
          field: "duration",
          got
        }) if got == 0.0
      ),
      "got {r:?}"
    );
    // duration = -1.0 (negative).
    let neg_dur = SlidingWindow::new(0.0, -1.0, PYANNOTE_FRAME_STEP_S);
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, chunks_sw(), neg_dur);
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonPositiveWindowParameter {
          which: "frames_sw",
          field: "duration",
          got
        }) if got == -1.0
      ),
      "got {r:?}"
    );
  }

  /// FINDING 1: a zero / negative `chunks_sw.step` is rejected as a
  /// non-positive window parameter. `try_num_output_frames_pyannote`
  /// only guards `frame_step`, so a zero chunk step would otherwise reach
  /// `try_count_pyannote` / reconstruct and fail only after clustering.
  #[test]
  fn new_rejects_non_positive_chunk_step() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let count = vec![1_u8; count_len(num_chunks)];
    let chunk_dur = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    // step = 0.0 (zero).
    let zero_step = SlidingWindow::new(0.0, chunk_dur, 0.0);
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb.clone(),
      count.clone(),
      zero_step,
      frames_sw(),
    );
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonPositiveWindowParameter {
          which: "chunks_sw",
          field: "step",
          got
        }) if got == 0.0
      ),
      "got {r:?}"
    );
    // step = -1.0 (negative).
    let neg_step = SlidingWindow::new(0.0, chunk_dur, -1.0);
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, neg_step, frames_sw());
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonPositiveWindowParameter {
          which: "chunks_sw",
          field: "step",
          got
        }) if got == -1.0
      ),
      "got {r:?}"
    );
  }

  /// FINDING 1: a non-finite (NaN / +inf) window duration is rejected.
  /// NaN/inf durations make the output-frame count non-finite (the count
  /// helper would saturate or reject) and corrupt reconstruct's
  /// `center_offset`; the full geometry validation rejects them up front.
  #[test]
  fn new_rejects_non_finite_duration() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let count = vec![1_u8; count_len(num_chunks)];
    // NaN chunk duration.
    let nan_dur = SlidingWindow::new(0.0, f64::NAN, 1.0);
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb.clone(),
      count.clone(),
      nan_dur,
      frames_sw(),
    );
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonPositiveWindowParameter {
          which: "chunks_sw",
          field: "duration",
          got
        }) if got.is_nan()
      ),
      "got {r:?}"
    );
    // +inf frame step.
    let inf_step = SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, f64::INFINITY);
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, chunks_sw(), inf_step);
    assert!(
      matches!(
        r,
        Err(RangeShapeError::NonPositiveWindowParameter {
          which: "frames_sw",
          field: "step",
          got
        }) if got == f64::INFINITY
      ),
      "got {r:?}"
    );
  }

  /// A geometry that passes the start/duration/step invariants but whose
  /// derived output-frame count overflows the cap is rejected as
  /// `InvalidGeometry` (the residual case after the explicit-invariant
  /// checks). Enormous `chunk_duration` + tiny `frame_step` saturates the
  /// `try_num_output_frames_pyannote` cap.
  #[test]
  fn new_rejects_uncomputable_count_geometry() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    // All positive + finite + start 0, but the count saturates the cap.
    let huge_chunks = SlidingWindow::new(0.0, 1.0e15, 1.0);
    let tiny_frames = SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, 1.0e-15);
    let count = vec![1_u8; 8];
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, count, huge_chunks, tiny_frames);
    assert!(
      matches!(r, Err(RangeShapeError::InvalidGeometry)),
      "got {r:?}"
    );
  }
}
