//! `RangeEmbeddings`: the public per-VAD-range carrier crossing the
//! `segment+embed → cluster` boundary.

use std::sync::Arc;

use crate::{
  aggregate::try_count_pyannote,
  embed::EMBEDDING_DIM,
  ops::spill::{SpillBytes, SpillOptions},
  reconstruct::SlidingWindow,
  segment::FRAMES_PER_WINDOW,
};

/// Speaker slots per chunk (pyannote powerset = 3). Local copy for
/// module independence; equals [`crate::offline::SLOTS_PER_CHUNK`].
const SLOTS_PER_CHUNK: usize = 3;

/// Frame-binarization onset the public [`RangeEmbeddings::new`] derives
/// `count` with — the pyannote community-1 default (`0.5`, matching
/// [`crate::offline::OwnedPipelineOptions`] and the `build_range`
/// internal path). The carrier contract requires `segmentations` to be
/// **hard 0/1** activity, so the derived count is onset-independent for
/// any threshold in `(0, 1]`; `0.5` is the canonical pyannote value and
/// keeps the public-boundary derivation byte-identical to the count the
/// internal path produces from the same hard segmentations.
const DERIVE_ONSET: f64 = 0.5;

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
  /// derivation (`aggregate::try_count_pyannote`, whose internal
  /// output-frame-count formula takes only `duration`/`step`) plus
  /// reconstruct's overlap-add (`chunk_start_time = chunks_sw.start +
  /// c * chunks_sw.step`, `closest_frame` subtracts `frames_sw.start`)
  /// both bake in that origin. A non-zero start is NOT reflected in the
  /// count length (the formula takes only `duration`/`step`), yet
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
  /// frame-count computation). The public boundary DERIVES `count` from
  /// this geometry via `aggregate::try_count_pyannote`, so a geometry
  /// whose output-frame count is underivable cannot produce a count at
  /// all; reject it here rather than letting the carrier through.
  #[error(
    "chunks_sw/frames_sw geometry: cannot derive the pyannote output-frame count \
     (count overflows usize or exceeds MAX_OUTPUT_FRAMES)"
  )]
  InvalidGeometry,
  /// `segmentations` contains a non-finite value (NaN / +inf / -inf).
  ///
  /// The public constructor DERIVES `count` from `segmentations` (it no
  /// longer trusts a caller-supplied count — see [`RangeEmbeddings::new`]);
  /// the derivation thresholds each cell against the onset, a comparison
  /// that is asymmetric on non-finite inputs (NaN/-inf compare false,
  /// +inf compares true), so a degraded segmentation backend emitting
  /// NaN/inf would silently fold into a finite-looking count. Reject it
  /// at the boundary — the same policy `aggregate::try_count_pyannote`
  /// and `reconstruct` apply to the segmentation tensor.
  #[error("segmentations contains a non-finite value (NaN / +inf / -inf)")]
  NonFiniteSegmentations,
  /// Deriving `count` from `segmentations` failed to allocate its
  /// spill-backed scratch buffers (`aggregate::try_count_pyannote`
  /// reserves two `num_output_frames`-long f64 working buffers that may
  /// spill to a tempfile/mmap). A storage/allocation failure surfaces
  /// here as a typed boundary error rather than an opaque panic.
  #[error("failed to derive count from segmentations (scratch allocation failed)")]
  CountDerivationFailed,
}

/// Validate the COMPLETE `SlidingWindow` geometry contract the split
/// protocol requires, BEFORE deriving the `count` from the
/// segmentations + geometry.
///
/// The count derivation's internal output-frame-count formula (inside
/// `aggregate::try_count_pyannote`) only inspects `chunks_sw.duration`,
/// `chunks_sw.step`, and `frames_sw.step` — it drops both `start`
/// offsets and never sees `frames_sw.duration`. But reconstruct
/// (`cluster_ranges_inner` -> `reconstruct`) consumes ALL six scalars:
/// its overlap-add computes `chunk_start_time = chunks_sw.start + c *
/// chunks_sw.step` and `closest_frame(t) = round((t - frames_sw.start -
/// frames_sw.duration / 2) / frames_sw.step)`. So a window that
/// satisfies the count formula can still be geometrically inconsistent
/// with the count it produced — e.g. a non-zero `chunks_sw.start`
/// leaves the count length unchanged yet shifts every reconstructed
/// span. Enforce the full contract here.
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

/// DERIVE the per-output-frame `count` for one range from its
/// `segmentations` + geometry, using the SAME
/// [`aggregate::try_count_pyannote`](crate::aggregate::try_count_pyannote)
/// helper the internal `build_range` path uses — so the public boundary
/// produces a count that is consistent-by-construction with the
/// segmentations this carrier also stores, rather than trusting a
/// caller-supplied one.
///
/// This is the structural fix for the count-forgery class: `count` is
/// always a function of `segmentations`, so a caller can no longer pass
/// all-zero segmentations alongside a fabricated all-one count to
/// conjure speaker spans from silence. An all-zero (silent)
/// segmentation derives an all-zero count of the exact pyannote
/// `num_output_frames` length; reconstruct then emits no spans.
///
/// [`validate_geometry`] must run first (it guarantees the
/// start/duration/step invariants), and the segmentations-length check
/// must already have passed (fixed `FRAMES_PER_WINDOW` /
/// [`SLOTS_PER_CHUNK`]). With those satisfied, the only residual
/// failures are:
/// - a non-finite segmentation cell → [`RangeShapeError::NonFiniteSegmentations`]
///   (checked here directly so the precise typed error does not depend on
///   `aggregate`'s non-re-exported inner `ShapeError`);
/// - a geometry whose derived output-frame count overflows / exceeds the
///   cap → [`RangeShapeError::InvalidGeometry`];
/// - a spill scratch-allocation failure → [`RangeShapeError::CountDerivationFailed`].
///
/// The onset is the canonical [`DERIVE_ONSET`]; because the carrier
/// contract requires hard-0/1 segmentations, the derived count is
/// onset-independent and byte-identical to the count `build_range`
/// produces from the same hard segmentations.
fn derive_count(
  segmentations: &[f64],
  num_chunks: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Result<Arc<[u8]>, RangeShapeError> {
  // Reject non-finite segmentation cells up front with the precise
  // typed error. `try_count_pyannote` rejects them too, but its inner
  // `ShapeError` is not re-exported from `aggregate`, so we cannot
  // name that variant when mapping its error; checking here keeps the
  // boundary error specific without widening another module's API.
  if segmentations.iter().any(|v| !v.is_finite()) {
    return Err(RangeShapeError::NonFiniteSegmentations);
  }
  try_count_pyannote(
    segmentations,
    num_chunks,
    FRAMES_PER_WINDOW,
    SLOTS_PER_CHUNK,
    DERIVE_ONSET,
    chunks_sw,
    frames_sw,
    &SpillOptions::default(),
  )
  .map(|t| t.into_parts().0)
  .map_err(|e| match e {
    // Spill scratch-buffer allocation failure during the derivation.
    crate::aggregate::Error::Spill(_) => RangeShapeError::CountDerivationFailed,
    // Every `ShapeError` reachable here is precluded by this caller's
    // prior checks (`num_chunks != 0`, `validate_geometry`, the
    // exact-length check, fixed positive `FRAMES_PER_WINDOW` /
    // `SLOTS_PER_CHUNK`, finite `DERIVE_ONSET`, the finite-segmentation
    // check above) EXCEPT an output-frame count that overflows or
    // exceeds `MAX_OUTPUT_FRAMES` for an otherwise-valid-looking but
    // saturating geometry — the residual `InvalidGeometry` case. Map
    // the whole shape residue to `InvalidGeometry` so the boundary
    // stays panic-free even if an upstream precondition were relaxed.
    crate::aggregate::Error::Shape(_) => RangeShapeError::InvalidGeometry,
  })
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
  /// lengths against `num_chunks` and the full sliding-window geometry,
  /// then DERIVING the per-output-frame `count` from `segmentations`.
  ///
  /// **`count` is derived, never trusted.** This constructor does NOT
  /// take a caller-supplied count. It computes the count from
  /// `segmentations` via the same
  /// [`aggregate::try_count_pyannote`](crate::aggregate::try_count_pyannote)
  /// helper the internal `build_range` path uses, so the count is a pure
  /// function of the segmentations this carrier also stores. This makes
  /// the count-forgery class unrepresentable: a caller can no longer pass
  /// all-zero (silent) segmentations alongside a fabricated all-one count
  /// to conjure speaker spans from silence — the derived count for silent
  /// segmentations is all-zero, and reconstruct emits no spans. (The
  /// segmentations are the protocol's source of truth: self-consistent by
  /// definition, and the count, the masked-speaker selection, and frame
  /// reconstruction are all functions of them.)
  ///
  /// - `abs_start_sample`: absolute sample index where this range
  ///   starts in the original stream (used to re-anchor output spans).
  /// - `num_chunks`: number of 10 s segmentation chunks in this range.
  /// - `segmentations`: hard 0/1 activity, flattened
  ///   `[chunk][frame][slot]`, length
  ///   `num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK`. The count is
  ///   derived from these, so they must be finite (NaN / ±inf rejected).
  /// - `raw_embeddings`: raw WeSpeaker vectors, flattened
  ///   `[chunk][slot][dim]`, length
  ///   `num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM`.
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
  /// ([`InvalidGeometry`](RangeShapeError::InvalidGeometry)),
  /// `segmentations` contains a non-finite cell
  /// ([`NonFiniteSegmentations`](RangeShapeError::NonFiniteSegmentations)),
  /// or the count derivation's scratch allocation fails
  /// ([`CountDerivationFailed`](RangeShapeError::CountDerivationFailed)).
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    abs_start_sample: u64,
    num_chunks: usize,
    segmentations: Vec<f64>,
    raw_embeddings: Vec<f32>,
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
    // DERIVE `count` from `segmentations` instead of trusting a
    // caller-supplied one. The prior design accepted a `count` argument
    // and only validated its LENGTH against the geometry — but length
    // doesn't prove the count was derived from THESE segmentations, so a
    // caller could pass all-zero segmentations + an exact-length all-one
    // count and fabricate speaker spans from silence (the count-forgery
    // class). Computing the count here from the segmentations via the
    // same helper `build_range` uses makes count a pure function of the
    // segmentations this carrier also stores, so forgery is
    // unrepresentable. (`derive_count` runs after `validate_geometry`,
    // which guarantees the geometry invariants the derivation relies on.)
    let count = derive_count(&segmentations, num_chunks, chunks_sw, frames_sw)?;
    Ok(Self {
      abs_start_sample,
      num_chunks,
      segmentations: SpillBytes::from_vec(segmentations),
      raw_embeddings: SpillBytes::from_vec(raw_embeddings),
      count,
      chunks_sw,
      frames_sw,
    })
  }

  /// Clearly-internal, UNCHECKED constructor: build the carrier from
  /// already-spill-backed buffers and a PRE-COMPUTED `count`, without
  /// re-validating shapes and WITHOUT re-deriving the count. Used ONLY
  /// by [`crate::streaming::offline_diarizer::build_range`], which builds
  /// the segmentations itself (enforcing the length invariants by
  /// construction) and derives `count` from those SAME segmentations via
  /// `aggregate::try_count_pyannote` — so the count is consistent with
  /// the segmentations by construction on this hot path, and re-deriving
  /// it in the public [`new`](Self::new) sense would be redundant work.
  ///
  /// This is the byte-identical path the desktop split-parity flow takes
  /// (`StreamingEmbedder::push` → `build_range` → here); the public
  /// `new` derives the same count from the same segmentations, so the two
  /// paths agree element-for-element for valid input.
  ///
  /// Keeping the tensors as `SpillBytes` lets a multi-hour range stay
  /// file-backed instead of `to_vec`-ing it onto the heap. `count`
  /// arrives as `Arc<[u8]>` straight from `aggregate::count_pyannote`,
  /// avoiding a copy.
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
  /// test geometry — the length `new` now DERIVES the count to.
  fn count_len(num_chunks: usize) -> usize {
    crate::aggregate::num_output_frames_pyannote(
      num_chunks,
      chunks_sw().duration(),
      chunks_sw().step(),
      frames_sw().step(),
    )
  }

  #[test]
  fn new_accepts_consistent_shapes_and_exposes_accessors() {
    let num_chunks = 2;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let n_count = count_len(num_chunks);
    let r = RangeEmbeddings::new(48_000, num_chunks, seg, emb, chunks_sw(), frames_sw())
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
    // The DERIVED count has the exact pyannote output-frame length, and
    // for all-zero (silent) segmentations every cell is 0.
    assert_eq!(r.count().len(), n_count);
    assert!(
      r.count().iter().all(|&c| c == 0),
      "silent segmentations must derive an all-zero count"
    );
  }

  #[test]
  fn new_rejects_mismatched_shapes() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb_bad = vec![0.0_f32; 999];
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb_bad,
      chunks_sw(),
      frames_sw(),
    );
    assert!(matches!(
      r,
      Err(RangeShapeError::RawEmbeddingsLenMismatch { .. })
    ));

    let r = RangeEmbeddings::new(0, 0, Vec::new(), Vec::new(), chunks_sw(), frames_sw());
    assert!(matches!(r, Err(RangeShapeError::ZeroChunks)));
  }

  /// R3 (the structural fix): a public caller passing all-zero
  /// segmentations can NOT fabricate speaker spans by supplying a
  /// crafted count, because `new` no longer accepts a count — it DERIVES
  /// the count from the segmentations. The carrier built from silent
  /// segmentations carries an all-zero count of the exact output-frame
  /// length, and clustering that range yields ZERO spans (no fabricated
  /// speech from silence).
  ///
  /// Before the fix, `new` took a `count: Vec<u8>` argument and only
  /// validated its length, so `new(.., all_zero_seg, .., all_one_count
  /// of exact length, ..)` constructed a carrier whose `count[t] == 1`
  /// for every frame; `cluster_ranges_inner` -> reconstruct then read
  /// `count[t]` as the top-k speaker count and emitted spans even though
  /// every reconstructed activation row was all-zero — speech conjured
  /// from silence. With the derive design that input is unconstructable:
  /// the all-one count cannot be passed at all, and the derived count is
  /// all-zero.
  #[test]
  fn new_derives_count_from_segmentations_no_fabrication_from_silence() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let opts = crate::streaming::StreamingOfflineOptions::new();
    let num_chunks = 2;
    // All-zero (silent) segmentations.
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, chunks_sw(), frames_sw())
      .expect("silent range constructs");
    // The DERIVED count is all-zero (no active speakers in any frame),
    // at the exact output-frame length — NOT a caller-controlled all-one.
    assert_eq!(r.count().len(), count_len(num_chunks));
    assert!(
      r.count().iter().all(|&c| c == 0),
      "derived count for silent segmentations must be all-zero, got {:?}",
      &r.count()[..r.count().len().min(8)]
    );
    // End to end: clustering a silent range fabricates NO spans.
    let spans =
      crate::streaming::cluster_ranges(std::slice::from_ref(&r), &plda, &opts).expect("cluster ok");
    assert_eq!(
      spans.len(),
      0,
      "silent segmentations must not fabricate any speaker spans"
    );
  }

  /// The derive boundary rejects non-finite segmentation cells (NaN /
  /// ±inf) rather than threshold-comparing them into a misleading count.
  /// `try_count_pyannote`'s segmentation finite-check surfaces here as a
  /// typed `NonFiniteSegmentations`.
  #[test]
  fn new_rejects_non_finite_segmentations() {
    let num_chunks = 1;
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
      let mut seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
      seg[0] = bad;
      let r = RangeEmbeddings::new(0, num_chunks, seg, emb.clone(), chunks_sw(), frames_sw());
      assert!(
        matches!(r, Err(RangeShapeError::NonFiniteSegmentations)),
        "got {r:?} for segmentation cell {bad}"
      );
    }
  }

  /// FINDING 3: a `num_chunks` large enough to overflow the
  /// `num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK` product is
  /// rejected as a typed `ShapeOverflow`, not a debug panic / release
  /// wrap that could bypass the length guard. The segmentations buffer
  /// is empty here; the overflow check fires before any length compare.
  #[test]
  fn new_rejects_shape_overflow() {
    let huge = usize::MAX / 2;
    let r = RangeEmbeddings::new(0, huge, Vec::new(), Vec::new(), chunks_sw(), frames_sw());
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
    // `validate_geometry` runs before the count is derived, so the
    // non-zero start is caught up front regardless of the segmentations.
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, bad_chunks, frames_sw());
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
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, chunks_sw(), bad_frames);
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
    // duration = 0.0 (zero), step valid.
    let zero_dur = SlidingWindow::new(0.0, 0.0, PYANNOTE_FRAME_STEP_S);
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb.clone(),
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
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, chunks_sw(), neg_dur);
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
  /// non-positive window parameter by `validate_geometry`, up front —
  /// before the count is derived — so a zero chunk step can never reach
  /// reconstruct and fail only after clustering.
  #[test]
  fn new_rejects_non_positive_chunk_step() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    let chunk_dur = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    // step = 0.0 (zero).
    let zero_step = SlidingWindow::new(0.0, chunk_dur, 0.0);
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb.clone(),
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
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, neg_step, frames_sw());
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
    // NaN chunk duration.
    let nan_dur = SlidingWindow::new(0.0, f64::NAN, 1.0);
    let r = RangeEmbeddings::new(
      0,
      num_chunks,
      seg.clone(),
      emb.clone(),
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
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, chunks_sw(), inf_step);
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
  /// pyannote output-frame-count cap inside the `derive_count` pass.
  #[test]
  fn new_rejects_uncomputable_count_geometry() {
    let num_chunks = 1;
    let seg = vec![0.0_f64; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];
    let emb = vec![0.0_f32; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];
    // All positive + finite + start 0, but the count saturates the cap.
    let huge_chunks = SlidingWindow::new(0.0, 1.0e15, 1.0);
    let tiny_frames = SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, 1.0e-15);
    let r = RangeEmbeddings::new(0, num_chunks, seg, emb, huge_chunks, tiny_frames);
    assert!(
      matches!(r, Err(RangeShapeError::InvalidGeometry)),
      "got {r:?}"
    );
  }
}
