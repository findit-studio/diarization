//! `PldaTransform` — the load-time setup + per-embedding projection.
//!
//! Construction loads the compile-time-embedded weight blobs and runs
//! the generalized-eigh setup once. Thereafter `xvec_transform` and
//! `plda_transform` are pure read-only mappings.

use nalgebra::{DMatrix, DVector};

use crate::{
  embed::NORM_EPSILON,
  plda::{
    EMBEDDING_DIMENSION, PLDA_DIMENSION,
    error::Error,
    loader::{PldaWeights, XvecWeights, load_plda, load_xvec},
  },
};

/// Minimum allowed L2 norm for a raw WeSpeaker embedding at the
/// [`RawEmbedding`] boundary.
///
/// Calibrated against the captured distribution: across the
/// 654 raw WeSpeaker embeddings in
/// `tests/parity/fixtures/01_dialogue/raw_embeddings.npz` the
/// observed range is `[0.536, 6.97]` with median 2.07. `0.01` sits
/// ~50× below the empirical minimum (so a far-out-of-distribution
/// real input still passes) and ~6 billion× above the canonical
/// near-zero attack `[1e-13; 256]` (norm `1.6e-12`), so any input
/// with norm in the `[1e-12, 0.01)` band is rejected.
///
/// # Why a data-calibrated floor instead of `NORM_EPSILON`
///
/// The earlier check rejected only `‖arr‖ < NORM_EPSILON = 1e-12`.
/// A degraded embedder returning tiny non-zero values (e.g.
/// `[1e-13; 256]`, norm 1.6e-12) passed that gate. Then in
/// `xvec_transform` the centering step `x - mean1 ≈ -mean1`
/// produces a centered f64 norm of `‖mean1‖ ≈ 1.42`, well above
/// [`XVEC_CENTERED_MIN_NORM`], so the L2-normalize amplifies a
/// fixed `-mean1`-direction into a finite `sqrt(128)`-normed PLDA
/// stage-1 output that VBx treats as a legitimate (constant)
/// speaker. This produces silent fabricated speaker evidence from
/// a dead embedder.
///
/// # Calibration limitation
///
/// The threshold is derived from a single 2-speaker conversational
/// fixture. Domain-shifted, very short, very quiet, or future-runtime
/// embeddings could in theory produce smaller raw norms and trigger
/// a false [`Error::DegenerateInput`] reject — pyannote itself has
/// no equivalent guard, so this is a deliberate divergence.
///
/// The trade-off was accepted because the alternative (silent
/// fabricated speaker evidence from a dead embedder) is a
/// no-observability failure mode, and the guard returns [`Result`]
/// rather than panicking — the caller owns the health-check policy.
/// The integration layer is the correct place to:
///
/// 1. Re-validate against multi-corpus captures (varied audio
///    domains, very-short utterances, low-energy speech).
/// 2. Add telemetry for `DegenerateInput` events so production can
///    observe rather than silently lose diarization.
/// 3. Decide the right fallback for a low-norm embedding (skip the
///    chunk, use a degraded score, surface to the caller, …).
/// 4. Surface a configuration knob if real production data shows
///    false positives.
///
/// Until then the threshold is a `pub(crate)` constant the
/// integration layer can read or override at compile time.
///
/// If the embedder model is ever changed, this constant must be
/// re-validated against fresh captured raw norms — see
/// `tests/parity/python/capture_intermediates.py`.
pub(crate) const RAW_EMBEDDING_MIN_NORM: f64 = 0.01;

/// Raw, **unnormalized** WeSpeaker output destined for the PLDA
/// transform. Wrapping the `[f32; 256]` in a distinct type prevents
/// the most likely API misuse: feeding
/// [`diarization::embed::Embedding::as_array`](crate::embed::Embedding::as_array),
/// which is L2-normalized.
///
/// Pyannote's `xvec_tf` operates on **raw** WeSpeaker outputs
/// (`pyannote/audio/pipelines/clustering.py:608` —
/// `fea = self.plda(train_embeddings)`, where `train_embeddings` is
/// the un-normalized output of `get_embeddings`; the
/// `train_embeddings_normed` copy is only used for AHC linkage). If a
/// caller feeds an L2-normalized vector here instead, the centering
/// `x - mean1` produces a different intermediate, the LDA projection
/// maps to the wrong subspace, and downstream VBx clustering silently
/// drifts off the captured pyannote distribution. See
/// `normalized_vs_raw_input_produce_materially_different_output` in
/// `src/plda/tests.rs`.
///
/// # Construction
///
/// Construction is `pub(crate)` — downstream crates cannot construct
/// a `RawEmbedding` at all. The only production path from a raw
/// WeSpeaker vector to PLDA features is via the offline diarization
/// pipeline (`offline::diarize_offline`), which constructs
/// `RawEmbedding` per (chunk, speaker) slot internally from the
/// caller's `raw_embeddings: &[f32]`. That keeps the type-safety
/// contract intact: a downstream caller cannot accidentally feed an
/// L2-normalized [`crate::embed::Embedding`] vector into PLDA, since
/// they cannot wrap it as a `RawEmbedding` themselves.
///
/// (A public `plda-fixtures` Cargo feature was previously used as the
/// gate, but additive features are globally unified, so any downstream
/// crate enabling it would have re-exposed the constructor for the
/// entire build. Sealing at the visibility level is the only reliable
/// way to enforce the provenance invariant.)
///
/// # Type-safety contract
///
/// `xvec_transform`'s signature requires `&RawEmbedding`, so passing
/// the L2-normalized `Embedding` vector is a compile error rather
/// than a silent distribution drift. The
/// `normalized_vs_raw_input_produce_materially_different_output`
/// test in `src/plda/tests.rs` is observable evidence the API
/// distinction matters: feeding the same vector raw vs L2-normalized
/// produces materially different `xvec_transform` outputs.
#[derive(Debug, Clone)]
pub struct RawEmbedding([f32; EMBEDDING_DIMENSION]);

impl RawEmbedding {
  /// Wrap a raw, **unnormalized** WeSpeaker embedding vector.
  /// `pub(crate)` — see [`RawEmbedding`]'s type-level docs for the
  /// visibility rationale (sealed-construction provenance contract).
  ///
  /// Validates the array is finite **and** has non-trivial L2 norm.
  /// Both checks matter: `xvec_transform` centers `input - mean1`
  /// before its inner norm guard fires, so a degraded ONNX output of
  /// all zeros would pass the inner guard (centered norm = `‖mean1‖`)
  /// and silently produce a finite `sqrt(128)`-normed PLDA stage-1
  /// vector that downstream VBx would treat as legitimate speaker
  /// evidence. Rejecting at the **uncentered** input here catches
  /// that class.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] if any element is NaN, `+inf`, or
  ///   `-inf`.
  /// - [`Error::DegenerateInput`] if `‖arr‖ < RAW_EMBEDDING_MIN_NORM`
  ///   (`0.01`, calibrated against the captured raw distribution —
  ///   see [`RAW_EMBEDDING_MIN_NORM`]). Catches all-zero, near-zero
  ///   (e.g. `[1e-13; 256]`), and other degraded-embedder outputs
  ///   that an `NORM_EPSILON`-only floor would have passed straight
  ///   through into `xvec_transform`'s centering step.
  ///
  /// The offline diarization path (`offline::OfflineDiarizer`) calls
  /// this on the per-chunk per-speaker masked WeSpeaker output. The
  /// validation is load-bearing: it rejects all-zero / near-zero
  /// degraded embedder outputs that would silently pass
  /// `xvec_transform`'s post-centering norm guard.
  pub(crate) fn from_raw_array(arr: [f32; EMBEDDING_DIMENSION]) -> Result<Self, Error> {
    if !arr.iter().all(|v| v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    // Reject degenerate input *before* `xvec_transform` centers it.
    // The norm is computed in f64 because squaring 256 small f32
    // values can lose precision near the threshold.
    let norm_sq: f64 = arr.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    if norm_sq.sqrt() < RAW_EMBEDDING_MIN_NORM {
      return Err(Error::DegenerateInput);
    }
    Ok(Self(arr))
  }

  /// Wrap a raw, **unnormalized** WeSpeaker embedding for the PLDA
  /// hand-off. This is the public constructor a downstream
  /// `segment+embed` node uses to carry the embedder's raw 256-d
  /// output across a service boundary into clustering.
  ///
  /// **Invariant:** the input must be the raw, unnormalized WeSpeaker
  /// output (norm typically in `[0.5, 7]`), NOT the L2-normalized
  /// [`crate::embed::Embedding`]. PLDA's `xvec_transform` is calibrated
  /// for the raw distribution; feeding a normalized vector drifts
  /// clustering off the captured pyannote distribution (see
  /// `normalized_vs_raw_input_produce_materially_different_output` in
  /// `src/plda/tests.rs`).
  ///
  /// Validation is identical to the internal `from_raw_array`:
  /// rejects non-finite values ([`Error::NonFiniteInput`]) and
  /// below-`RAW_EMBEDDING_MIN_NORM` degenerate output
  /// ([`Error::DegenerateInput`]).
  pub fn from_wespeaker(arr: [f32; EMBEDDING_DIMENSION]) -> Result<Self, Error> {
    Self::from_raw_array(arr)
  }

  /// Borrow the raw, unnormalized 256-d vector. Distinct from
  /// [`crate::embed::Embedding::as_array`], which is L2-normalized.
  pub const fn as_array(&self) -> &[f32; EMBEDDING_DIMENSION] {
    &self.0
  }
}

/// Output of [`PldaTransform::xvec_transform`] / input to
/// [`PldaTransform::plda_transform`]. A 128-d f64 vector with norm
/// `sqrt(PLDA_DIMENSION) ≈ 11.31` — the intermediate distribution
/// that `plda_tf` is mathematically defined for.
///
/// Wrapping the `[f64; 128]` in a distinct type prevents the
/// stage-2 analogue of the `RawEmbedding` misuse: feeding
/// `plda_transform` a vector that wasn't produced by `xvec_transform`
/// (e.g. an L2-normalized 128-d vector with norm 1.0, a stale
/// pyannote capture from a different revision, or hand-constructed
/// input). Without this gate, `plda_transform` would whiten any
/// finite input and return — VBx then clusters wrong-distribution
/// features without any error signal.
///
/// The only production path to a `PostXvecEmbedding` is calling
/// [`PldaTransform::xvec_transform`] (which constructs internally
/// via the `pub(super)` `from_xvec_output`). Parity tests use a
/// `#[cfg(test)] pub(crate)` constructor that loads from a captured
/// pyannote run and validates the norm; that constructor cannot be
/// reached from production builds or downstream crates.
///
/// # Type-safety contract
///
/// `plda_transform`'s signature requires `&PostXvecEmbedding`, so
/// passing a raw `[f64; 128]` is a compile error rather than a
/// silent distribution drift.
#[derive(Debug, Clone)]
pub struct PostXvecEmbedding([f64; PLDA_DIMENSION]);

impl PostXvecEmbedding {
  /// Internal constructor for `xvec_transform`. Skips norm validation
  /// because the algorithm guarantees the invariant by construction.
  pub(super) fn from_xvec_output(arr: [f64; PLDA_DIMENSION]) -> Self {
    Self(arr)
  }

  /// Internal constructor for parity tests that load a `post_xvec`
  /// value from a captured pyannote run. `#[cfg(test)] pub(crate)`
  /// — see [`PostXvecEmbedding`]'s type-level docs for why this is
  /// not reachable from production builds.
  ///
  /// Validates finite + norm within `1e-3` of `sqrt(PLDA_DIMENSION)`.
  /// The norm check is necessary but not sufficient — a synthetic
  /// 128-d vector scaled to `sqrt(128)` would still pass it — which
  /// is precisely why this constructor must remain test-only.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] on any NaN/`±inf` element.
  /// - [`Error::WrongPostXvecNorm`] if the norm is outside the
  ///   expected `sqrt(D_out) ± 1e-3` band — the input is not a
  ///   post-`xvec_tf` vector.
  #[cfg(test)]
  pub(crate) fn from_pyannote_capture(arr: [f64; PLDA_DIMENSION]) -> Result<Self, Error> {
    if !arr.iter().all(|v| v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    let norm: f64 = arr.iter().map(|v| v * v).sum::<f64>().sqrt();
    let expected = (PLDA_DIMENSION as f64).sqrt();
    let tolerance = 1.0e-3;
    if (norm - expected).abs() > tolerance {
      return Err(Error::WrongPostXvecNorm {
        actual: norm,
        expected,
        tolerance,
      });
    }
    Ok(Self(arr))
  }

  /// Borrow the underlying f64 vector. Gated alongside
  /// [`Self::from_pyannote_capture`] so the same visibility rules
  /// apply.
  #[cfg(test)]
  pub(crate) fn as_array(&self) -> &[f64; PLDA_DIMENSION] {
    &self.0
  }
}

/// Minimum allowed `‖input - mean1‖` after the first centering step.
///
/// Calibrated against the captured distribution rather than
/// f32 quantization noise: across the 654 raw WeSpeaker embeddings
/// in `tests/parity/fixtures/01_dialogue/raw_embeddings.npz`, the
/// observed centered-norm range is `[1.36, 7.08]` with median 2.45.
/// `0.1` sits ~14× below the empirical minimum (so a far-out-of-
/// distribution real input still passes) and ~2.86 million× above
/// the f32-roundtrip noise floor of `mean1` (~3.49e-8 for the
/// committed weights), so any centered norm in the
/// `[noise_floor, 0.1)` band is rejected.
///
/// # Why a constant rather than the previous noise-floor × 1000
///
/// The earlier threshold was `‖mean1 - mean1.astype(f32)‖ × 1000`
/// ≈ 3.5e-5. That left a ~38000× attack window between threshold
/// and real signal: an embedder collapsed to `mean1.astype(f32) +
/// jitter` with `‖jitter‖` anywhere in `(3.5e-5, 1.36)` would pass
/// the guard, the L2-normalize would amplify the attacker-chosen
/// jitter direction to unit norm, and the rest of the pipeline
/// would whiten that into a fabricated speaker-evidence vector
/// indistinguishable from a real embedding. Calibrating to the
/// data closes that window.
///
/// # Calibration limitation
///
/// Same caveat as [`RAW_EMBEDDING_MIN_NORM`]: the `0.1` threshold
/// is derived from a single 2-speaker conversational fixture.
/// Pyannote does not have an equivalent guard, so this is a
/// deliberate divergence — the trade-off was made because multiple
/// `mean1`-collapse attacks were documented where the L2-normalize
/// amplifies pure quantization or attacker-controlled jitter into a
/// finite `sqrt(128)`-normed PLDA output. The integration layer owns
/// the production health-check policy: telemetry, multi-corpus
/// validation, fallback, and (if needed) per-deployment threshold
/// tuning. The guard returns [`Result`] rather than panicking so the
/// integration layer can observe + skip rather than abort.
///
/// If the model weights or the embedder are ever changed, this
/// constant must be re-validated against fresh captured data —
/// see `tests/parity/python/capture_intermediates.py`.
pub(crate) const XVEC_CENTERED_MIN_NORM: f64 = 0.1;

/// Probabilistic Linear Discriminant Analysis transform. Two stages:
///
/// 1. [`xvec_transform`](Self::xvec_transform): center → L2-norm → LDA →
///    recenter → L2-norm → scale by `sqrt(D_out)`. Output `‖·‖ = sqrt(128)`.
/// 2. [`plda_transform`](Self::plda_transform): center → project onto
///    the descending-sorted generalized eigenvectors of `eigh(B, W)`.
///    Output is whitened (NOT L2-normed).
///
/// Mirrors `pyannote.audio.utils.vbx.vbx_setup` + `xvec_tf` + `plda_tf`
/// (`utils/vbx.py:181-218` in pyannote.audio 4.0.4). Validated
/// against the captured artifacts via `src/plda/parity_tests.rs`.
pub struct PldaTransform {
  // xvec_tf factors
  mean1: DVector<f64>,
  mean2: DVector<f64>,
  lda: DMatrix<f64>,
  sqrt_in_dim: f64,  // sqrt(EMBEDDING_DIMENSION)
  sqrt_out_dim: f64, // sqrt(PLDA_DIMENSION)

  // plda_tf factors (used by `plda_transform` and `phi()`).
  plda_mu: DVector<f64>,
  plda_eigenvectors_desc: DMatrix<f64>,
  phi: DVector<f64>,
}

impl PldaTransform {
  /// Construct from the compile-time-embedded weight blobs.
  ///
  /// Runs the generalized symmetric eigenvalue solve `eigh(B, W)`
  /// once at construction time:
  ///
  /// ```text
  /// W = inv(tr.T @ tr)              # within-class precision
  /// B = inv((tr.T / psi) @ tr)      # between-class precision
  /// (eigenvalues, eigenvectors) = generalized_eigh(B, W)  # ascending
  /// → reverse to descending → store
  /// ```
  ///
  /// Mirrors `pyannote/audio/utils/vbx.py:201-208`.
  pub fn new() -> Result<Self, Error> {
    let XvecWeights { mean1, mean2, lda } = load_xvec();
    let PldaWeights {
      mu,
      eigenvectors_desc,
      phi_desc,
    } = load_plda();

    // Eigenvectors are pre-computed offline via scipy's `eigh` on
    // `(B, W)` and shipped in `models/plda/eigenvectors_desc.bin`.
    // See `loader::EIGENVECTORS_DESC_BYTES` for the rationale —
    // LAPACK eigenvector signs are implementation-defined and
    // pinning them avoids a 38% DER divergence on fixture 04 due
    // to nalgebra/scipy disagreeing on 67 of 128 column signs.
    Ok(Self {
      mean1,
      mean2,
      lda,
      sqrt_in_dim: (EMBEDDING_DIMENSION as f64).sqrt(),
      sqrt_out_dim: (PLDA_DIMENSION as f64).sqrt(),
      plda_mu: mu,
      plda_eigenvectors_desc: eigenvectors_desc,
      phi: phi_desc,
    })
  }

  /// First PLDA stage. Mirrors `xvec_tf` in
  /// `pyannote/audio/utils/vbx.py:211-213`:
  ///
  /// ```text
  /// xvec_tf(x) = sqrt(D_out) *
  ///     l2_norm( lda.T @ (sqrt(D_in) * l2_norm(x - mean1)) - mean2 )
  /// ```
  ///
  /// Output norm is `sqrt(PLDA_DIMENSION)` — i.e. `sqrt(128) ≈ 11.31`,
  /// **not** 1.0. The outer scale-by-`sqrt(D_out)` is load-bearing
  /// for the downstream PLDA whitening; downstream consumers MUST
  /// not re-normalize this output.
  ///
  /// `input` is a [`RawEmbedding`] — a raw, **unnormalized** WeSpeaker
  /// vector — not [`diarization::embed::Embedding`](crate::embed::Embedding)
  /// (L2-normalized) which is the wrong distribution for PLDA.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] if a non-finite value appears in an
  ///   intermediate vector (the input is finite by `RawEmbedding`'s
  ///   construction-time invariant; this guards against arithmetic
  ///   overflows in the LDA projection).
  /// - [`Error::DegenerateInput`] if `‖input - mean1‖` is below the
  ///   data-calibrated `XVEC_CENTERED_MIN_NORM` threshold (`0.1`
  ///   — see the constant's source docs for the calibration), or if the
  ///   second-stage intermediate becomes degenerate. The first check
  ///   rejects both the `mean1.astype(f32)` collapse-to-mean attack
  ///   and the more sophisticated `mean1 + small_jitter` variants
  ///   that an earlier f32-quantization-noise-based threshold would
  ///   have admitted. (round 6).
  pub fn xvec_transform(&self, input: &RawEmbedding) -> Result<PostXvecEmbedding, Error> {
    // Input finite-ness is enforced by `RawEmbedding::from_raw_array`,
    // so we don't re-validate here. Intermediate-vector checks happen
    // inside `checked_l2_normalize_in_place` below.

    // 1. Promote f32 input to f64 and center: x = input - mean1.
    let mut x =
      DVector::<f64>::from_iterator(EMBEDDING_DIMENSION, input.0.iter().map(|v| *v as f64));
    x -= &self.mean1;

    // 2. L2-normalize, then scale by sqrt(D_in). Use the
    //    data-calibrated `XVEC_CENTERED_MIN_NORM` threshold here
    //    rather than the shared `NORM_EPSILON`. The threat model is
    //    a degraded or adversarial embedder returning `mean1 +
    //    jitter` for a small `jitter`: the centered f64 norm is
    //    `‖jitter‖`, the L2-normalize amplifies the (attacker-chosen)
    //    direction of `jitter` to unit norm, and the rest of the
    //    pipeline whitens that into a `sqrt(128)`-normed PLDA
    //    stage-1 vector indistinguishable from a real embedding.
    //    The threshold at `0.1` is calibrated against the captured
    //    real-input distribution (smallest observed centered norm
    //    1.36 across 654 raw embeddings); any below-threshold
    //    centered norm cannot be a real WeSpeaker output.
    checked_l2_normalize_in_place_with_min(&mut x, XVEC_CENTERED_MIN_NORM)?;
    x *= self.sqrt_in_dim;

    // 3. lda.T @ x  →  (PLDA_DIMENSION,)-shaped vector.
    //    nalgebra's `tr_mul` is matmul-with-transposed-lhs; avoids
    //    an explicit transpose copy.
    let mut y = self.lda.tr_mul(&x);

    // 4. Recenter: y -= mean2.
    y -= &self.mean2;

    // 5. L2-normalize, then scale by sqrt(D_out). Same validation
    //    as step 2 — guards against degenerate intermediates that
    //    could come from a corrupted upstream LDA matrix.
    checked_l2_normalize_in_place(&mut y)?;
    y *= self.sqrt_out_dim;

    let mut out = [0.0f64; PLDA_DIMENSION];
    for (slot, value) in out.iter_mut().zip(y.iter()) {
      *slot = *value;
    }
    // The algorithm guarantees `‖out‖ == sqrt(D_out)` by construction
    // — no need to re-validate via `from_pyannote_capture`.
    Ok(PostXvecEmbedding::from_xvec_output(out))
  }

  /// Second PLDA stage. Mirrors `plda_tf` in
  /// `pyannote/audio/utils/vbx.py:215-217`:
  ///
  /// ```text
  /// plda_tf(x0) = (x0 - plda_mu) @ plda_tr.T
  /// ```
  ///
  /// where `plda_tr = wccn.T[::-1]` (eigenvectors of the generalized
  /// problem as ROWS, in descending eigenvalue order). So
  /// `plda_tr.T = wccn[:, ::-1]` — eigenvectors as columns, descending.
  /// We store that directly in `plda_eigenvectors_desc` and matmul.
  ///
  /// Output is whitened (NOT L2-normed). The Rust port uses
  /// `eigenvectors.tr_mul(centered_x)` to express the row-vector
  /// matmul in column-vector form — the resulting ordering matches
  /// pyannote's row-major numpy result.
  ///
  /// `post_xvec` must be a [`PostXvecEmbedding`]. Distribution +
  /// finite-ness are enforced by that type — `plda_transform` itself
  /// does no validation. (stage-2 analogue of the
  /// `RawEmbedding` boundary).
  pub fn plda_transform(&self, post_xvec: &PostXvecEmbedding) -> [f64; PLDA_DIMENSION] {
    // 1. Center: x = post_xvec - plda_mu.
    let mut x = DVector::<f64>::from_iterator(PLDA_DIMENSION, post_xvec.0.iter().copied());
    x -= &self.plda_mu;

    // 2. Project onto descending eigenvectors. pyannote does
    // `(x - mu) @ eigenvectors_desc` (row vector × matrix). In
    // column-vector terms that's `eigenvectors_desc.T @ (x - mu)`.
    // `tr_mul(&x)` computes `self.transpose() * x` without an
    // explicit transpose copy.
    let y = self.plda_eigenvectors_desc.tr_mul(&x);

    let mut out = [0.0f64; PLDA_DIMENSION];
    for (slot, value) in out.iter_mut().zip(y.iter()) {
      *slot = *value;
    }
    out
  }

  /// Convenience: chain `xvec_transform` → `plda_transform`. Returns
  /// only the errors produced by stage 1 (`xvec_transform`); stage 2
  /// is now infallible because [`PostXvecEmbedding`] enforces its
  /// own preconditions.
  pub fn project(&self, input: &RawEmbedding) -> Result<[f64; PLDA_DIMENSION], Error> {
    let post_xvec = self.xvec_transform(input)?;
    Ok(self.plda_transform(&post_xvec))
  }

  /// Eigenvalue diagonal `phi` (descending) — `pyannote.audio.core.plda.PLDA.phi`.
  /// Consumed by VBx as the across-class covariance diagonal.
  pub fn phi(&self) -> &[f64] {
    self.phi.as_slice()
  }

  /// The clustering / PLDA stage identity for provenance stamping —
  /// the community-1 diarization pipeline. The version is fixed by the
  /// embedded weights ([`crate::provenance::DIARIZATION_PLDA_VERSION`]).
  pub fn identity(&self) -> crate::provenance::ModelIdentity {
    crate::provenance::ModelIdentity::new(
      crate::provenance::DIARIZATION_FAMILY,
      crate::provenance::DIARIZATION_PLDA_VERSION,
    )
  }
}

/// In-place L2 normalization with explicit error reporting. Returns
/// [`Error::NonFiniteInput`] if the norm is non-finite (input had
/// NaN/Inf that survived earlier checks; defense-in-depth) and
/// [`Error::DegenerateInput`] if the norm is below
/// `NORM_EPSILON` (dividing would amplify noise to dominate signal).
///
/// Used for the stage-2 (post-LDA) intermediate where the noise
/// floor is f64 quantization, well below `NORM_EPSILON`.
fn checked_l2_normalize_in_place(v: &mut DVector<f64>) -> Result<(), Error> {
  checked_l2_normalize_in_place_with_min(v, NORM_EPSILON as f64)
}

/// `checked_l2_normalize_in_place` with a caller-supplied minimum
/// norm. Used by `xvec_transform`'s first centering, where the
/// effective noise floor is `‖mean1.astype(f32) - mean1‖` (the
/// quantization noise of mean1 itself), ~3.5e-8 for the committed
/// weights — far above the shared `NORM_EPSILON = 1e-12`.
fn checked_l2_normalize_in_place_with_min(
  v: &mut DVector<f64>,
  min_norm: f64,
) -> Result<(), Error> {
  let n = v.norm();
  if !n.is_finite() {
    return Err(Error::NonFiniteInput);
  }
  if n < min_norm {
    return Err(Error::DegenerateInput);
  }
  *v /= n;
  Ok(())
}

#[cfg(test)]
mod helper_tests {
  use super::*;

  #[test]
  fn from_wespeaker_accepts_real_norm_and_exposes_array() {
    // A vector with norm well inside the captured raw range [0.536, 6.97].
    let mut arr = [0.0_f32; EMBEDDING_DIMENSION];
    arr[0] = 2.0; // norm 2.0 >> RAW_EMBEDDING_MIN_NORM (0.01)
    let raw = RawEmbedding::from_wespeaker(arr).expect("valid raw vector");
    assert_eq!(raw.as_array()[0], 2.0);
    assert_eq!(raw.as_array().len(), EMBEDDING_DIMENSION);
  }

  #[test]
  fn from_wespeaker_rejects_degenerate_and_nonfinite() {
    let zero = [0.0_f32; EMBEDDING_DIMENSION];
    assert!(matches!(
      RawEmbedding::from_wespeaker(zero),
      Err(Error::DegenerateInput)
    ));
    let mut nan = [0.0_f32; EMBEDDING_DIMENSION];
    nan[0] = 2.0;
    nan[1] = f32::NAN;
    assert!(matches!(
      RawEmbedding::from_wespeaker(nan),
      Err(Error::NonFiniteInput)
    ));
  }

  /// Direct test of the near-zero-norm guard. Constructed at the
  /// helper level rather than the public-API level because real f32
  /// inputs cannot produce a centered f64 norm below `NORM_EPSILON`
  /// after the f32→f64 promotion round-trip noise (see
  /// `src/plda/tests.rs` comment for the analysis).
  #[test]
  fn checked_l2_normalize_rejects_near_zero() {
    let mut v = DVector::<f64>::from_iterator(4, [1e-15, 1e-15, 1e-15, 1e-15]);
    let n = v.norm();
    assert!(
      n < NORM_EPSILON as f64,
      "test input norm {n} must be < epsilon"
    );
    let result = checked_l2_normalize_in_place(&mut v);
    assert!(
      matches!(result, Err(Error::DegenerateInput)),
      "got {result:?}"
    );
  }

  #[test]
  fn checked_l2_normalize_rejects_nan() {
    let mut v = DVector::<f64>::from_iterator(3, [1.0, f64::NAN, 1.0]);
    let result = checked_l2_normalize_in_place(&mut v);
    assert!(
      matches!(result, Err(Error::NonFiniteInput)),
      "got {result:?}"
    );
  }

  #[test]
  fn checked_l2_normalize_rejects_inf() {
    let mut v = DVector::<f64>::from_iterator(3, [1.0, f64::INFINITY, 1.0]);
    let result = checked_l2_normalize_in_place(&mut v);
    assert!(
      matches!(result, Err(Error::NonFiniteInput)),
      "got {result:?}"
    );
  }

  #[test]
  fn checked_l2_normalize_succeeds_on_unit_input() {
    let mut v = DVector::<f64>::from_iterator(3, [3.0, 4.0, 0.0]);
    checked_l2_normalize_in_place(&mut v).expect("non-degenerate, finite");
    let n = v.norm();
    assert!((n - 1.0).abs() < 1e-15, "norm after normalize = {n}");
  }
}
