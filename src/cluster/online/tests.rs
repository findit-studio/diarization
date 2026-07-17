//! Hermetic, model-free tests for `diarization::cluster::online`.
//!
//! Every cosine distance and every centroid value asserted below is
//! hand-computed from exactly-representable inputs (one-hot / small-fraction
//! vectors). The suite is written to be *mutation-proof*: each pinned case
//! names, in a comment, the source mutation it catches.

use crate::{
  cluster::online::{
    Assignment, OnlineClusterOptions, OnlineClusterer,
    algo::{cosine_distance, l2_normalize},
  },
  embed::{EMBEDDING_DIM, Embedding},
};

/// One-hot unit embedding along dimension `i` (exact: norm is `1.0`, so
/// `normalize_from` returns it unchanged).
fn basis(i: usize) -> Embedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = 1.0;
  Embedding::normalize_from(v).unwrap()
}

/// The antipode of [`basis`]`(i)`: `-eᵢ`, unit-norm, at cosine distance
/// exactly `2.0` from `basis(i)` (dot `−1`, clamped, `1 − (−1)`). The maximum
/// possible cosine distance — the value that makes the `speaker_threshold`
/// ceiling load-bearing.
fn neg_basis(i: usize) -> Embedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = -1.0;
  Embedding::normalize_from(v).unwrap()
}

/// Embedding with only dims 0 and 1 set, then L2-normalized.
fn emb2(a: f32, b: f32) -> Embedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[0] = a;
  v[1] = b;
  Embedding::normalize_from(v).unwrap()
}

/// Raw (un-normalized) `[f32; EMBEDDING_DIM]` with only dims 0 and 1 set — for
/// the direct helper tests that must exercise the non-unit division path.
fn raw2(a: f32, b: f32) -> [f32; EMBEDDING_DIM] {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[0] = a;
  v[1] = b;
  v
}

/// Options with wide-open thresholds (in `[0, 2]`) and no minimum duration, so
/// each test dials in exactly the gate it is probing.
fn opts(speaker: f32, embedding: f32, min_dur: f32) -> OnlineClusterOptions {
  OnlineClusterOptions::new()
    .with_speaker_threshold(speaker)
    .with_embedding_threshold(embedding)
    .with_min_speech_duration(min_dur)
}

// ── Basic assignment paths ───────────────────────────────────────────────

#[test]
fn first_sufficient_segment_seeds_speaker_one() {
  let mut c = OnlineClusterer::new(OnlineClusterOptions::default());
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.speaker_count(), 1);
  assert_eq!(c.speaker_ids().collect::<Vec<_>>(), vec![1]);
}

#[test]
fn short_segment_without_speakers_is_dropped() {
  // Default min_speech_duration = 1.0; 0.5 s with no speaker to match → drop.
  let mut c = OnlineClusterer::new(OnlineClusterOptions::default());
  assert_eq!(c.assign(&basis(0), 0.5), Assignment::Dropped);
  assert_eq!(c.speaker_count(), 0);
}

#[test]
fn duration_equal_to_min_creates_speaker() {
  // >= boundary (SpeakerManager.swift:164). Mutation `>=`→`>`: this drops.
  let mut c = OnlineClusterer::new(opts(0.65, 0.45, 1.0));
  assert_eq!(c.assign(&basis(0), 1.0), Assignment::New(1));
}

#[test]
fn identical_embedding_reuses_speaker() {
  let mut c = OnlineClusterer::new(OnlineClusterOptions::default());
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  // Distance 0 < speaker_threshold → reuse.
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::Existing(1));
  assert_eq!(c.speaker_count(), 1);
}

#[test]
fn orthogonal_embedding_spawns_new_speaker() {
  // Distance basis0↔basis1 is 1.0; default speaker_threshold 0.65 → new.
  let mut c = OnlineClusterer::new(OnlineClusterOptions::default());
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.assign(&basis(1), 2.0), Assignment::New(2));
  assert_eq!(c.speaker_count(), 2);
}

// ── Threshold boundary behavior (strict `<`) ─────────────────────────────

#[test]
fn assignment_boundary_is_strict_less_than() {
  // Distance basis0↔basis1 == 1.0 == speaker_threshold. Strict `<` (152) →
  // NOT assigned → a second speaker. Mutation `<`→`<=`: basis1 reuses spk1
  // and speaker_count would be 1.
  let mut c = OnlineClusterer::new(opts(1.0, 0.5, 0.0));
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.assign(&basis(1), 2.0), Assignment::New(2));
  assert_eq!(c.speaker_count(), 2);
}

#[test]
fn update_boundary_is_strict_less_than() {
  // basis1 is assigned to spk1 (dist 1.0 < 1.5) but the update gate is exactly
  // at the boundary: 1.0 < embedding_threshold(1.0) is FALSE → duration-only,
  // centroid untouched. Mutation `<`→`<=`: the centroid would EMA toward
  // basis1 and no longer equal basis0.
  let mut c = OnlineClusterer::new(opts(1.5, 1.0, 0.0));
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.assign(&basis(1), 2.0), Assignment::Existing(1));
  // Centroid is still exactly basis0.
  assert_eq!(c.centroid(1).unwrap(), basis(0).as_array());
  // But the duration accumulated (SpeakerManager.swift:457).
  assert_eq!(c.duration(1), Some(4.0));
}

// ── The centroid-update composite (recalc-then-EMA, NOT a plain EMA) ──────

#[test]
fn centroid_update_is_recalc_then_ema() {
  // Pins the exact composite (SpeakerTypes.swift:68-101,132-162):
  //   history = [e0, e1]; mean = [0.5, 0.5]; recalc → [1/√2, 1/√2];
  //   EMA(α=0.9) → normalize(0.9·[1/√2,1/√2] + 0.1·[0,1])
  //             = normalize([0.6363961, 0.7363961]) ≈ [0.653871, 0.756613].
  let mut c = OnlineClusterer::new(opts(1.5, 1.5, 0.0));
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.assign(&basis(1), 2.0), Assignment::Existing(1));

  let centroid = c.centroid(1).unwrap();
  assert!(
    (centroid[0] - 0.653871).abs() < 1e-4,
    "centroid[0] = {}",
    centroid[0]
  );
  assert!(
    (centroid[1] - 0.756613).abs() < 1e-4,
    "centroid[1] = {}",
    centroid[1]
  );
  assert_eq!(centroid[2], 0.0);
  // Distinguish the composite from a PURE EMA on the previous centroid, which
  // would give normalize(0.9·[1,0] + 0.1·[0,1]) ≈ [0.99388, 0.11043]. A plain
  // EMA keeps centroid[0] ≈ 0.994; the composite pulls it to ≈ 0.654.
  assert!(
    centroid[0] < 0.9,
    "recalc-then-EMA must move centroid[0] well below 0.9"
  );
  // Still unit-norm.
  let norm_sq: f32 = centroid.iter().map(|x| x * x).sum();
  assert!((norm_sq - 1.0).abs() < 1e-4, "norm_sq = {norm_sq}");
}

#[test]
fn ema_alpha_is_pinned_at_point_nine() {
  // The composite above is exquisitely sensitive to α. With α=0.8 the same
  // inputs give normalize(0.8·[1/√2,1/√2] + 0.2·[0,1]) ≈ [0.5942, 0.8043] —
  // centroid[0] ≈ 0.594, not 0.654. This asserts the 0.9 pin (SpeakerManager
  // .swift:452) tightly enough to catch any α perturbation ≥ ~0.02.
  assert_eq!(crate::cluster::online::EMA_ALPHA, 0.9);
  let mut c = OnlineClusterer::new(opts(1.5, 1.5, 0.0));
  c.assign(&basis(0), 2.0);
  c.assign(&basis(1), 2.0);
  let centroid = c.centroid(1).unwrap();
  // α=0.8 would land centroid[0] near 0.594; α=0.9 lands it near 0.654.
  assert!(
    centroid[0] > 0.64,
    "centroid[0] = {} (α mutated below 0.9?)",
    centroid[0]
  );
}

// ── Tie-breaking (deterministic, lowest id wins) ─────────────────────────

#[test]
fn equal_distance_ties_to_lowest_id() {
  // query = normalize([1,1]) is exactly equidistant from basis0 and basis1
  // (both dot products are the same f32). Strict-min + ascending-id scan →
  // the LOWER id wins. Mutation `<`→`<=` in findClosest: the higher id wins.
  let mut c = OnlineClusterer::new(opts(0.9, 0.0, 0.0));
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.assign(&basis(1), 2.0), Assignment::New(2));
  assert_eq!(c.assign(&emb2(1.0, 1.0), 2.0), Assignment::Existing(1));
}

// ── Determinism and order-dependence ─────────────────────────────────────

#[test]
fn same_sequence_is_deterministic() {
  let seq = [
    (basis(0), 2.0),
    (basis(0), 2.0),
    (basis(1), 2.0),
    (basis(1), 2.0),
    (basis(2), 0.2),
  ];
  let mut a = OnlineClusterer::new(OnlineClusterOptions::default());
  let mut b = OnlineClusterer::new(OnlineClusterOptions::default());
  let out_a = a.assign_all(seq.iter().copied());
  let out_b = b.assign_all(seq.iter().copied());
  assert_eq!(out_a, out_b);
  assert_eq!(a.speaker_count(), b.speaker_count());
  for id in a.speaker_ids() {
    assert_eq!(
      a.centroid(id),
      b.centroid(id),
      "centroid mismatch for speaker {id}"
    );
  }
}

#[test]
fn order_changes_assignment() {
  // A=[1,0], B≈[0.6,0.8], C=[0,1]. dist(A,B)=0.4, dist(B,C)=0.2, dist(A,C)=1.0.
  // speaker_threshold 0.5 (B matches A and C; A and C never match each other);
  // embedding_threshold 0.0 freezes centroids so the greediness is pure.
  let a = basis(0);
  let b = emb2(0.6, 0.8);
  let cc = basis(1);

  // [A, B, C]: B joins A's cluster (spk1); C starts spk2.
  let mut c1 = OnlineClusterer::new(opts(0.5, 0.0, 0.0));
  let out1 = c1.assign_all([(a, 2.0), (b, 2.0), (cc, 2.0)]);
  assert_eq!(
    out1,
    vec![
      Assignment::New(1),
      Assignment::Existing(1),
      Assignment::New(2)
    ]
  );

  // [A, C, B]: C starts spk2 first; B is now closer to C → joins spk2.
  let mut c2 = OnlineClusterer::new(opts(0.5, 0.0, 0.0));
  let out2 = c2.assign_all([(a, 2.0), (cc, 2.0), (b, 2.0)]);
  assert_eq!(
    out2,
    vec![
      Assignment::New(1),
      Assignment::New(2),
      Assignment::Existing(2)
    ]
  );

  // Same three embeddings, different order → B lands on a different speaker.
  assert_ne!(out1, out2);
}

// ── FIFO history cap (exercise the remove-oldest branch) ─────────────────

#[test]
fn many_identical_updates_stay_one_speaker() {
  // 60 identical embeddings > RAW_HISTORY_CAP (50), exercising the FIFO
  // remove-first branch. Identical inputs → mean == input → centroid stays at
  // basis0; still exactly one speaker.
  let mut c = OnlineClusterer::new(opts(1.5, 1.5, 0.0));
  for _ in 0..60 {
    c.assign(&basis(0), 1.0);
  }
  assert_eq!(c.speaker_count(), 1);
  let centroid = c.centroid(1).unwrap();
  assert!(
    (centroid[0] - 1.0).abs() < 1e-5,
    "centroid[0] = {}",
    centroid[0]
  );
}

#[test]
fn fifo_cap_evicts_oldest_at_fifty() {
  // Pin the FIFO-50 rule precisely (SpeakerTypes.swift:111-115). The prior
  // test feeds 60 IDENTICAL vectors, so the mean is basis0 no matter what the
  // cap does — it cannot see an eviction bug. Here we seed spk1 with basis(1),
  // then feed 50 × basis(0) into the SAME speaker (wide thresholds so every
  // basis(0) both matches and updates spk1). The 50th append finds the history
  // already at the cap and must evict the OLDEST entry — the lone basis(1) —
  // leaving history = [basis(0); 50], whose mean (hence centroid) is EXACTLY
  // basis(0).
  //
  // Mutation-proofs (each turns this red):
  //   • cap 50→51, or the off-by-one `>=`→`>`: basis(1) survives the 50th
  //     append, so centroid ≈ [0.99984, 0.01800, …] ≠ basis(0).
  //   • wrong-end eviction (remove newest): basis(1) is never evicted → same
  //     residue → centroid ≠ basis(0).
  let mut c = OnlineClusterer::new(opts(1.5, 1.5, 0.0));
  assert_eq!(c.assign(&basis(1), 1.0), Assignment::New(1));
  for _ in 0..50 {
    assert_eq!(c.assign(&basis(0), 1.0), Assignment::Existing(1));
  }
  assert_eq!(c.speaker_count(), 1);
  // Exactly basis(0): the basis(1) seed has been evicted from the FIFO.
  assert_eq!(c.centroid(1).unwrap(), basis(0).as_array());
}

// ── State management ─────────────────────────────────────────────────────

#[test]
fn reset_clears_speakers_and_id_counter() {
  let mut c = OnlineClusterer::new(OnlineClusterOptions::default());
  c.assign(&basis(0), 2.0);
  c.assign(&basis(1), 2.0);
  assert_eq!(c.speaker_count(), 2);
  c.reset();
  assert_eq!(c.speaker_count(), 0);
  // Id counter restarts at 1.
  assert_eq!(c.assign(&basis(2), 2.0), Assignment::New(1));
}

#[test]
fn assignment_speaker_id_accessor() {
  assert_eq!(Assignment::New(1).speaker_id(), Some(1));
  assert_eq!(Assignment::Existing(3).speaker_id(), Some(3));
  assert_eq!(Assignment::Dropped.speaker_id(), None);
}

// ── Direct hand-verification of the ported cosine / normalize math ───────

#[test]
fn cosine_distance_hand_values_unit_fast_path() {
  // Both unit → similarity IS the dot product (SpeakerOperations.swift:85-86).
  let b0 = *basis(0).as_array();
  let b1 = *basis(1).as_array();
  let neg0 = raw2(-1.0, 0.0);
  assert_eq!(cosine_distance(&b0, &b0), 0.0); // identical
  assert_eq!(cosine_distance(&b0, &b1), 1.0); // orthogonal
  assert_eq!(cosine_distance(&b0, &neg0), 2.0); // antipodal (clamp then 1-(-1))

  // [0.6,0.8]·[1,0] = 0.6 → distance 0.4.
  let known = *emb2(0.6, 0.8).as_array();
  assert!((cosine_distance(&known, &b0) - 0.4).abs() < 1e-6);
}

#[test]
fn cosine_distance_hand_values_division_path() {
  // Non-unit inputs (‖·‖=5) force the divide-by-magnitudes branch
  // (SpeakerOperations.swift:88-96): [3,4]·[4,-3] = 0 → distance 1.0.
  let a = raw2(3.0, 4.0);
  let b = raw2(4.0, -3.0);
  assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
  // [3,4]·[3,4] = 25, /(5·5) = 1 → distance 0.
  assert!(cosine_distance(&a, &a).abs() < 1e-6);
}

#[test]
fn cosine_distance_zero_magnitude_is_infinite() {
  // SpeakerOperations.swift:76-79 — a zero-magnitude vector → infinity.
  let zero = [0.0f32; EMBEDDING_DIM];
  assert_eq!(cosine_distance(&zero, &raw2(3.0, 4.0)), f32::INFINITY);
}

#[test]
fn l2_normalize_hand_values() {
  // [3,4] → [0.6,0.8] (norm 5).
  let n = l2_normalize(&raw2(3.0, 4.0));
  assert!((n[0] - 0.6).abs() < 1e-6, "n[0] = {}", n[0]);
  assert!((n[1] - 0.8).abs() < 1e-6, "n[1] = {}", n[1]);

  // Already-unit input is idempotent.
  let u = l2_normalize(basis(0).as_array());
  assert!((u[0] - 1.0).abs() < 1e-6);

  // Zero vector is CLAMPED, not rejected: norm floored at 1e-12, result all
  // zero, no NaN (VDSPOperations.swift:17 — distinct from normalize_from).
  let z = l2_normalize(&[0.0f32; EMBEDDING_DIM]);
  assert!(
    z.iter().all(|x| *x == 0.0),
    "clamped zero vector must stay zero, got {:?}",
    &z[..2]
  );
}

#[test]
fn cosine_distance_unit_fast_path_is_not_the_divide_path() {
  // Distinguish the unit fast path (similarity = raw dot) from the
  // always-divide branch (SpeakerOperations.swift:81-96). `a` has
  // sum-of-squares 1.0000960 — inside UNIT_TOLERANCE (1e-3) of 1 — so both
  // inputs are treated as unit and the similarity IS the dot product, 0.9,
  // giving distance 0.1. The divide branch would scale by 1/‖a‖ and land at
  // 0.1000432 instead. The fast-path result is within 1 ULP of 0.1; the divide
  // result is ~4.3e-5 away, so `< 1e-6` separates them.
  //
  // Mutation-proof: delete the `is_unit_a && is_unit_b` fast path (always
  // divide) → distance 0.1000432, |·−0.1| ≈ 4.3e-5 > 1e-6 → red.
  let a = raw2(0.9, 0.436); // ‖a‖² = 1.0000960, inside the fast-path window
  let b0 = *basis(0).as_array();
  let d = cosine_distance(&a, &b0);
  assert!((d - 0.1).abs() < 1e-6, "fast-path distance = {d}");
}

#[test]
fn cosine_distance_clamps_overshooting_dot_to_zero() {
  // emb2(2,3) normalizes to a vector whose f32 self-sum-of-squares rounds to
  // 1.0000001 (> 1.0). On the unit fast path the self-"similarity" is that
  // dot — 1.0000001 — which OVERSHOOTS 1.0; the clamp to [-1, 1]
  // (SpeakerOperations.swift:99) pins it at 1.0, so the self-distance is
  // exactly 0.0.
  //
  // Mutation-proof: drop the `.clamp(-1.0, 1.0)` → similarity 1.0000001,
  // distance 1 − 1.0000001 = −1.19e-7, a NEGATIVE distance ≠ 0.0 → red.
  let e = *emb2(2.0, 3.0).as_array();
  assert_eq!(cosine_distance(&e, &e), 0.0);
}

#[test]
fn l2_normalize_epsilon_clamp_is_1e_minus_12() {
  // Pin the norm floor in l2_normalize at exactly 1e-12
  // (VDSPOperations.swift:10,17). For v = [5e-13, 0, …] the true norm 5e-13 is
  // BELOW the floor, so the norm is clamped to 1e-12 and the scale is
  // 1/1e-12 = 1e12, mapping the single component to 5e-13 · 1e12 = 0.5 exactly.
  //
  // Mutation-proofs:
  //   • drop the `.max(L2_NORM_EPSILON)` clamp → scale by 1/5e-13, component
  //     → 1.0 ≠ 0.5 → red.
  //   • change the epsilon (e.g. 1e-13 < 5e-13) → the clamp never fires, scale
  //     by 1/5e-13, component → 1.0 ≠ 0.5 → red.
  // Such a vector can never arrive as an `Embedding` (normalize_from rejects
  // ‖·‖ < NORM_EPSILON); this pins the clamp on the intermediate centroids.
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[0] = 5e-13;
  let out = l2_normalize(&v);
  assert_eq!(out[0], 0.5);
  assert!(out[1..].iter().all(|&x| x == 0.0));
}

// ── Serde-bypass construction guard (L2) ─────────────────────────────────
//
// `OnlineClusterOptions` derives `Deserialize`; `#[serde(default)]` fills any
// missing field but the `with_*`/`set_*` range checks are bypassed, so a JSON
// blob can carry an out-of-range threshold straight into the struct.
// `OnlineClusterer::try_new` re-validates every field at the construction
// boundary (mirroring `Segmenter::try_new` / `cluster_offline`); `new` panics
// on the same violation. The `serde`-gated tests below are the only way to
// synthesize a bypassed `OnlineClusterOptions` in stable Rust — its fields are
// private and the setters panic — so they are gated on the feature that
// exercises the guard.

/// The exact failing history from the finding: `speaker_threshold` 2.1 sits
/// ABOVE the 2.0 cosine-distance ceiling. `try_new` must reject it.
///
/// Mutation-proof (delete the `speaker_threshold` guard in `try_new`):
/// construction returns `Ok`, and the `Ok` arm here reproduces the finding —
/// feeding `basis(0)` then its antipode reuses spk1 because the antipodal
/// distance `2.0 < 2.1`, an `Existing(1)` no validated config can produce.
/// That arm asserts `New(2)`, so the removed guard fails this test concretely.
#[cfg(feature = "serde")]
#[test]
fn try_new_rejects_serde_bypassed_speaker_threshold() {
  use crate::cluster::Error;
  let opts: OnlineClusterOptions = serde_json::from_str(
    r#"{"speaker_threshold":2.1,"embedding_threshold":0.0,"min_speech_duration":0.0}"#,
  )
  .expect("deserialize");
  match OnlineClusterer::try_new(opts) {
    Err(Error::InvalidOnlineOption { field, value, .. }) => {
      assert_eq!(field, "speaker_threshold");
      assert_eq!(value, 2.1);
    }
    Ok(mut c) => {
      assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
      assert_eq!(
        c.assign(&neg_basis(0), 2.0),
        Assignment::New(2),
        "guard removed: antipodal distance 2.0 < 2.1 reused spk1 (Existing(1)) \
         — the exact serde bypass this test pins"
      );
    }
    Err(other) => panic!("expected InvalidOnlineOption, got {other:?}"),
  }
}

/// `new` panics on the same bypassed config (mirrors `Segmenter::new`).
/// Mutation-proof: remove the guard → `new` no longer panics → `should_panic`
/// fails.
#[cfg(feature = "serde")]
#[test]
#[should_panic(expected = "invalid options")]
fn new_panics_on_serde_bypassed_speaker_threshold() {
  let opts: OnlineClusterOptions = serde_json::from_str(
    r#"{"speaker_threshold":2.1,"embedding_threshold":0.0,"min_speech_duration":0.0}"#,
  )
  .expect("deserialize");
  let _ = OnlineClusterer::new(opts);
}

/// The second guarded field: an out-of-range `embedding_threshold` (2.5 > 2.0)
/// is rejected and named. Mutation-proof: drop the `embedding_threshold` check
/// → `try_new` returns `Ok` → the `other` arm panics.
#[cfg(feature = "serde")]
#[test]
fn try_new_rejects_serde_bypassed_embedding_threshold() {
  use crate::cluster::Error;
  let opts: OnlineClusterOptions =
    serde_json::from_str(r#"{"embedding_threshold":2.5}"#).expect("deserialize");
  match OnlineClusterer::try_new(opts) {
    Err(Error::InvalidOnlineOption { field, value, .. }) => {
      assert_eq!(field, "embedding_threshold");
      assert_eq!(value, 2.5);
    }
    other => panic!("expected InvalidOnlineOption, got {other:?}"),
  }
}

/// The third guarded field: a negative `min_speech_duration` is rejected and
/// named. Mutation-proof: drop the `min_speech_duration` check → `try_new`
/// returns `Ok` → the `other` arm panics.
#[cfg(feature = "serde")]
#[test]
fn try_new_rejects_serde_bypassed_min_speech_duration() {
  use crate::cluster::Error;
  let opts: OnlineClusterOptions =
    serde_json::from_str(r#"{"min_speech_duration":-1.0}"#).expect("deserialize");
  match OnlineClusterer::try_new(opts) {
    Err(Error::InvalidOnlineOption { field, value, .. }) => {
      assert_eq!(field, "min_speech_duration");
      assert_eq!(value, -1.0);
    }
    other => panic!("expected InvalidOnlineOption, got {other:?}"),
  }
}

/// A valid JSON config still round-trips and constructs: `try_new` returns
/// `Ok` and the clusterer assigns normally. Guards against an over-eager guard
/// that rejects in-range configs (which would make `.expect(...)` panic here).
#[cfg(feature = "serde")]
#[test]
fn try_new_accepts_valid_serde_roundtrip() {
  let opts: OnlineClusterOptions = serde_json::from_str(
    r#"{"speaker_threshold":0.65,"embedding_threshold":0.45,"min_speech_duration":1.0}"#,
  )
  .expect("deserialize");
  let mut c = OnlineClusterer::try_new(opts).expect("valid config must construct");
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::Existing(1));
}

/// The boundary the finding pivots on, provable without serde: the *maximum
/// valid* `speaker_threshold` (2.0) still SPAWNS a new speaker for an antipode,
/// because the assignment gate is strict `<` and the antipodal distance is
/// exactly 2.0 (`2.0 < 2.0` is false). No validated configuration can turn the
/// antipode into `Existing`, which is exactly why the serde-bypassed 2.1 is a
/// defect. Mutation-proof: assignment gate `<`→`<=` → the antipode reuses spk1
/// (`Existing(1)`) → this test's `New(2)` fails.
#[test]
fn max_valid_speaker_threshold_spawns_on_antipode() {
  let mut c = OnlineClusterer::new(opts(2.0, 0.0, 0.0));
  assert_eq!(c.assign(&basis(0), 2.0), Assignment::New(1));
  assert_eq!(c.assign(&neg_basis(0), 2.0), Assignment::New(2));
  assert_eq!(c.speaker_count(), 2);
}
