//! The greedy online centroid matcher — a faithful port of FluidAudio's
//! `SpeakerManager.assignSpeaker` and the `Speaker` centroid machinery it
//! drives.
//!
//! Citations are to the `FluidAudio` repo, paths under
//! `Sources/FluidAudio/Diarizer/`. See the module doc ([`super`]) for the
//! algorithm-class caveats (order-dependent, not pyannote-parity).

use crate::{
  cluster::Error,
  embed::{EMBEDDING_DIM, Embedding},
};

use super::options::{OnlineClusterOptions, duration_in_range, threshold_in_range};

/// EMA blend weight for the centroid update — the `alpha` hard-wired at
/// FluidAudio's update call site (`Clustering/SpeakerManager.swift:452`) and
/// the `updateMainEmbedding` default (`Clustering/SpeakerTypes.swift:72`). The
/// new embedding contributes `1 - EMA_ALPHA`.
pub const EMA_ALPHA: f32 = 0.9;

/// FIFO cap on a speaker's retained raw embeddings. At capacity the oldest is
/// dropped before the newest is appended (`Clustering/SpeakerTypes.swift:111`).
pub const RAW_HISTORY_CAP: usize = 50;

/// Minimum sum-of-squares for an embedding to be allowed to update a centroid
/// — FluidAudio's degenerate-vector guard
/// (`Clustering/SpeakerManager.swift:447`, `Clustering/SpeakerTypes.swift:77`).
/// For a unit-norm [`Embedding`] this is always satisfied (`≈ 1.0`); it is
/// ported for fidelity, not because it can fire here.
const EMBEDDING_UPDATE_MIN_SUMSQ: f32 = 0.01;

/// Clamp epsilon in FluidAudio's `l2Normalize`: the norm is floored at this
/// value rather than the vector being rejected
/// (`Offline/Utils/VDSPOperations.swift:10,17`). Distinct from dia's
/// [`Embedding::normalize_from`], which *rejects* below `NORM_EPSILON`; the
/// centroid intermediates here are not [`Embedding`]s, so the faithful clamp
/// applies.
const L2_NORM_EPSILON: f32 = 1e-12;

/// Tolerance for the "already unit norm" fast path in FluidAudio's
/// `cosineDistance` (`Clustering/SpeakerOperations.swift:11,81-82`): if both
/// squared norms are within this of `1.0`, the similarity is the raw dot
/// product; otherwise it is divided by the magnitudes.
const UNIT_TOLERANCE: f32 = 1e-3;

/// The outcome of one [`OnlineClusterer::assign`] step.
///
/// FluidAudio collapses all three into a `Speaker?` return
/// (`Clustering/SpeakerManager.swift:135-177`, `nil` for a drop). This port
/// keeps them distinct: the branch taken is itself information (new speaker vs.
/// reused speaker vs. skipped segment), and none of the three is an *error*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assignment {
  /// Assigned to an existing speaker: the nearest centroid was strictly
  /// closer than `speaker_threshold` (`Clustering/SpeakerManager.swift:152`).
  /// Carries that speaker's id.
  Existing(u64),
  /// Spawned a new speaker: no centroid within `speaker_threshold` AND
  /// `speech_duration >= min_speech_duration`
  /// (`Clustering/SpeakerManager.swift:163-172`). Carries the new id.
  New(u64),
  /// Dropped: no centroid within `speaker_threshold` AND `speech_duration <
  /// min_speech_duration` (`Clustering/SpeakerManager.swift:174-176`). No
  /// speaker is recorded.
  Dropped,
}

impl Assignment {
  /// The id of the speaker this segment landed on, or `None` for
  /// [`Dropped`](Self::Dropped). [`New`](Self::New) and
  /// [`Existing`](Self::Existing) both yield an id.
  pub fn speaker_id(&self) -> Option<u64> {
    match self {
      Self::Existing(id) | Self::New(id) => Some(*id),
      Self::Dropped => None,
    }
  }
}

/// One tracked speaker: its running centroid plus the raw-embedding history
/// that centroid is (re)derived from.
#[derive(Debug, Clone)]
struct OnlineSpeaker {
  id: u64,
  /// Running centroid, maintained ~unit-norm. Mirrors
  /// `Speaker.currentEmbedding` (`Clustering/SpeakerTypes.swift:12`).
  centroid: [f32; EMBEDDING_DIM],
  /// Retained normalized raw embeddings (FIFO, cap [`RAW_HISTORY_CAP`]).
  /// Mirrors `Speaker.rawEmbeddings` (`Clustering/SpeakerTypes.swift:22`); the
  /// centroid recalculation averages exactly these.
  raw_history: Vec<[f32; EMBEDDING_DIM]>,
  /// Accumulated assigned speech (seconds). Mirrors `Speaker.duration`
  /// (`Clustering/SpeakerTypes.swift:14`). Tracked for inspection/parity; no
  /// assignment decision depends on a speaker's accumulated duration.
  duration: f32,
}

/// FluidAudio's streaming speaker database (`SpeakerManager`) as a sans-I/O,
/// deterministic Rust state machine.
///
/// Feed embeddings one at a time with [`assign`](Self::assign); each call
/// either reuses the nearest existing speaker, spawns a new one, or drops the
/// segment. **The result depends on the order embeddings are fed** — this is
/// the defining property of the greedy online algorithm, not a defect (see
/// [`super`]). Determinism *given a fixed order* is total: no RNG, no
/// hash-map iteration, pure `f32` arithmetic.
#[derive(Debug, Clone)]
pub struct OnlineClusterer {
  options: OnlineClusterOptions,
  /// Speakers in creation order; the index order IS ascending-id order. This
  /// pins the tie-break that FluidAudio leaves to nondeterministic
  /// `Dictionary` iteration (`Clustering/SpeakerManager.swift:421`): see
  /// [`Self::assign`].
  speakers: Vec<OnlineSpeaker>,
  /// Id for the next spawned speaker. Starts at `1`, matching FluidAudio's
  /// `nextSpeakerId` (`Clustering/SpeakerManager.swift:16`, ids rendered as
  /// `String(nextSpeakerId)` at `:470`).
  next_id: u64,
}

impl Default for OnlineClusterer {
  fn default() -> Self {
    Self::new(OnlineClusterOptions::default())
  }
}

impl OnlineClusterer {
  /// A fresh clusterer with no speakers and the given options.
  ///
  /// # Panics
  /// Panics if any [`OnlineClusterOptions`] field is out of range: either
  /// threshold non-finite or outside `[0.0, 2.0]`, or `min_speech_duration`
  /// non-finite or negative. Defense-in-depth: the option setters already
  /// enforce these invariants on the builder path, so this trip only fires
  /// when the options were constructed without them — most realistically a
  /// serde-deserialized config (`#[serde(default)]` fields are never validated
  /// by the setters). Use [`Self::try_new`] to surface these preconditions as
  /// a [`cluster::Error`](crate::cluster::Error) instead.
  pub fn new(options: OnlineClusterOptions) -> Self {
    Self::try_new(options).expect("OnlineClusterer::new: invalid options; use try_new to handle")
  }

  /// Fallible variant of [`Self::new`]. Returns a
  /// [`cluster::Error`](crate::cluster::Error) (`InvalidOnlineOption`) for any
  /// of the range violations described on [`Self::new`]; otherwise identical
  /// output.
  ///
  /// Validation happens once, at the construction boundary — the same
  /// serde-bypass defense `Segmenter::try_new` and `cluster_offline` apply.
  /// `OnlineClusterOptions` derives `Deserialize` with `#[serde(default)]`
  /// fields, so a JSON/TOML config can carry a threshold above the `2.0`
  /// cosine-distance ceiling (or a negative duration) that the `with_*` /
  /// `set_*` setters would have rejected; left unchecked those values reach
  /// [`assign`](Self::assign) and flip its strict-`<` / `>=` gates.
  pub fn try_new(options: OnlineClusterOptions) -> Result<Self, Error> {
    // Mirror the setter predicates exactly (the shared `threshold_in_range` /
    // `duration_in_range`) so a serde-bypassed config is gated by the same
    // rule the builder path enforces.
    if !threshold_in_range(options.speaker_threshold()) {
      return Err(Error::InvalidOnlineOption {
        field: "speaker_threshold",
        value: options.speaker_threshold(),
        constraint: "finite cosine distance in [0.0, 2.0]",
      });
    }
    if !threshold_in_range(options.embedding_threshold()) {
      return Err(Error::InvalidOnlineOption {
        field: "embedding_threshold",
        value: options.embedding_threshold(),
        constraint: "finite cosine distance in [0.0, 2.0]",
      });
    }
    if !duration_in_range(options.min_speech_duration()) {
      return Err(Error::InvalidOnlineOption {
        field: "min_speech_duration",
        value: options.min_speech_duration(),
        constraint: "finite and >= 0.0 seconds",
      });
    }
    Ok(Self {
      options,
      speakers: Vec::new(),
      next_id: 1,
    })
  }

  /// The clusterer's options.
  pub fn options(&self) -> &OnlineClusterOptions {
    &self.options
  }

  /// Number of speakers currently tracked.
  pub fn speaker_count(&self) -> usize {
    self.speakers.len()
  }

  /// Tracked speaker ids, ascending (creation order).
  pub fn speaker_ids(&self) -> impl Iterator<Item = u64> + '_ {
    self.speakers.iter().map(|s| s.id)
  }

  /// The current centroid of speaker `id`, or `None` if no such speaker.
  ///
  /// The vector is ~unit-norm. Exposed for inspection and for the
  /// Swift-trace parity oracle (see [`super`]).
  pub fn centroid(&self, id: u64) -> Option<&[f32; EMBEDDING_DIM]> {
    self
      .speakers
      .iter()
      .find(|s| s.id == id)
      .map(|s| &s.centroid)
  }

  /// Accumulated assigned speech duration (seconds) for speaker `id`, or
  /// `None` if no such speaker.
  pub fn duration(&self, id: u64) -> Option<f32> {
    self
      .speakers
      .iter()
      .find(|s| s.id == id)
      .map(|s| s.duration)
  }

  /// Assign one embedding, returning the [`Assignment`] taken.
  ///
  /// Faithful port of `SpeakerManager.assignSpeaker`
  /// (`Clustering/SpeakerManager.swift:135-177`):
  ///
  /// 1. `embedding` is the caller's L2-normalized vector, standing in for the
  ///    `l2Normalize(rawEmbedding)` FluidAudio computes at `:147`. Two
  ///    faithfulness caveats apply, both deliberate:
  ///    - **Normalization is not bit-identical to Accelerate.** dia's
  ///      [`Embedding`] comes from [`Embedding::normalize_from`] (sum of
  ///      squares in `f64`, then a per-component divide); FluidAudio's
  ///      `l2Normalize` sums in `f32` via `vDSP_dotpr` and multiplies by the
  ///      reciprocal. They agree to ~1 ULP but not to the bit, and vDSP's SIMD
  ///      reduction order is *not* reproducible in scalar `f32` at all — even
  ///      the exact scalar `l2Normalize` op-sequence still differs from Swift's
  ///      committed vDSP trace by ≈3.4e-8 (~1 ULP). A cosine distance sitting
  ///      exactly on `speaker_threshold` can therefore have its strict-`<`
  ///      outcome differ from Accelerate by that last ULP. This is a bound of
  ///      the scalar port, not a bug: re-normalizing here does not close it (it
  ///      only injects more scalar-vs-vDSP noise — measured against the oracle
  ///      trace), so the arithmetic is left as the faithful op-mirror it
  ///      already is.
  ///    - **Degenerate vectors are excluded by domain, not rejected the Swift
  ///      way.** An [`Embedding`] exists only for `||raw|| >= NORM_EPSILON`, so a
  ///      zero/degenerate raw never reaches `assign`. FluidAudio does *not*
  ///      reject it: `l2Normalize` floors the norm at `1e-12`, so a zero vector
  ///      normalizes to all-zeros and STILL creates a speaker (with a zero
  ///      centroid — only its history entry is dropped by the `sumSquares >
  ///      0.01` guard), and a sub-`1e-12` vector is *amplified*
  ///      (`[5e-13, …] → [0.5, …]` at `:147`) and STILL creates a speaker. But
  ///      the create path normalizes a SECOND time (`createNewSpeaker`,
  ///      `:469`), re-scaling that `[0.5, …]` to the unit `[1.0, …]`, so the
  ///      stored centroid is `[1.0, …]`, not the amplified `[0.5, …]`. (The
  ///      later re-normalizes preserve the now-unit vector: `Speaker.init`,
  ///      `RawEmbedding.init`, `recalculateMainEmbedding` —
  ///      `Clustering/SpeakerTypes.swift:48,215,159`.) dia's typed API forbids
  ///      those states rather than
  ///      reproducing them; this is a restriction of the port's input domain,
  ///      not equivalent behavior.
  /// 2. Find the nearest existing centroid by cosine distance, strict-min
  ///    (`:417-430`). Ties break to the LOWEST id: speakers are scanned in
  ///    ascending-id order and the strict `<` keeps the first of an equal
  ///    run. (FluidAudio scans a `Dictionary`, whose order is undefined; this
  ///    port fixes it.)
  /// 3. If that nearest distance is `< speaker_threshold` (`:152`), assign to
  ///    it and update it ([`Self::update_existing`]).
  /// 4. Otherwise, if `speech_duration >= min_speech_duration` (`:164`),
  ///    spawn a new speaker.
  /// 5. Otherwise drop the segment (`:174-176`).
  pub fn assign(&mut self, embedding: &Embedding, speech_duration: f32) -> Assignment {
    let e = *embedding.as_array();

    // Step 2: nearest centroid, strict-min over ascending-id speakers.
    let mut min_distance = f32::INFINITY;
    let mut closest: Option<usize> = None;
    for (i, sp) in self.speakers.iter().enumerate() {
      let d = cosine_distance(&e, &sp.centroid);
      if d < min_distance {
        min_distance = d;
        closest = Some(i);
      }
    }

    // Step 3: assign to the nearest speaker if strictly within threshold.
    if let Some(i) = closest
      && min_distance < self.options.speaker_threshold()
    {
      self.update_existing(i, &e, speech_duration, min_distance);
      return Assignment::Existing(self.speakers[i].id);
    }

    // Step 4: spawn a new speaker if the segment is long enough.
    if speech_duration >= self.options.min_speech_duration() {
      let id = self.create_new(&e, speech_duration);
      return Assignment::New(id);
    }

    // Step 5: too short to seed a speaker.
    Assignment::Dropped
  }

  /// Assign a whole sequence in order, collecting the per-item outcomes.
  ///
  /// Pure convenience over [`assign`](Self::assign) in a loop; the ordering of
  /// `items` is the order-dependence knob (see [`super`]).
  pub fn assign_all(
    &mut self,
    items: impl IntoIterator<Item = (Embedding, f32)>,
  ) -> Vec<Assignment> {
    items.into_iter().map(|(e, d)| self.assign(&e, d)).collect()
  }

  /// Clear all speakers and reset the id counter to `1`. Mirrors
  /// `SpeakerManager.reset` (`Clustering/SpeakerManager.swift:610-628`, the
  /// non-permanent branch — this port has no permanent-speaker concept).
  pub fn reset(&mut self) {
    self.speakers.clear();
    self.next_id = 1;
  }

  /// `updateExistingSpeaker` (`Clustering/SpeakerManager.swift:432-460`).
  ///
  /// If `distance` is strictly under `embedding_threshold` (`:444`), and the
  /// embedding clears the `sumSquares > 0.01` quality guard (`:445-447`), the
  /// centroid is updated ([`OnlineSpeaker::update_main_embedding`]). Otherwise
  /// only the accumulated duration grows (`:457`) — the centroid is untouched.
  fn update_existing(&mut self, i: usize, e: &[f32; EMBEDDING_DIM], duration: f32, distance: f32) {
    let sp = &mut self.speakers[i];
    if distance < self.options.embedding_threshold() {
      if sum_squares(e) > EMBEDDING_UPDATE_MIN_SUMSQ {
        sp.update_main_embedding(e, duration);
      }
      // sumSquares <= 0.01: FluidAudio does nothing here (no `else`), so
      // neither does this port. Unreachable for a unit-norm Embedding.
    } else {
      sp.duration += duration;
    }
  }

  /// `createNewSpeaker` (`Clustering/SpeakerManager.swift:462-492`). The new
  /// speaker's centroid is the (already normalized) embedding, and its raw
  /// history is that single embedding — exactly what `Speaker.init` +
  /// `addRawEmbedding` → `recalculateMainEmbedding` produce for one raw
  /// (`Clustering/SpeakerTypes.swift:36-55,104-117,132-162`). No EMA runs on
  /// creation.
  fn create_new(&mut self, e: &[f32; EMBEDDING_DIM], duration: f32) -> u64 {
    let id = self.next_id;
    self.next_id += 1;
    self.speakers.push(OnlineSpeaker {
      id,
      centroid: *e,
      raw_history: vec![*e],
      duration,
    });
    id
  }
}

impl OnlineSpeaker {
  /// `Speaker.updateMainEmbedding` (`Clustering/SpeakerTypes.swift:68-101`) —
  /// the composite centroid update, NOT a plain EMA:
  ///
  /// 1. Append `e` to the raw history, FIFO-capped at [`RAW_HISTORY_CAP`]
  ///    (`:104-117`).
  /// 2. `recalculateMainEmbedding` (`:132-162`): centroid ←
  ///    `l2_normalize(mean(raw_history))` — the mean over ALL retained raws,
  ///    including the one just appended.
  /// 3. EMA (`:90-94`): centroid ←
  ///    `l2_normalize(EMA_ALPHA · centroid + (1 - EMA_ALPHA) · e)`.
  ///
  /// The redundant `sumSquares > 0.01` guard at `:75-77` is applied by the
  /// caller before this runs; it is not re-checked here.
  fn update_main_embedding(&mut self, e: &[f32; EMBEDDING_DIM], duration: f32) {
    // 1. FIFO append (SpeakerTypes.swift:111-115). FluidAudio re-normalizes on
    //    the way in (RawEmbedding.init -> l2Normalize, SpeakerTypes.swift:215);
    //    dia stores `*e` directly. That renorm is redundant here -- `e` is an
    //    Embedding, already unit-norm by construction -- and re-adding it only
    //    widens the scalar-vs-vDSP gap against Swift's oracle trace (measured),
    //    so it is deliberately omitted.
    if self.raw_history.len() >= RAW_HISTORY_CAP {
      self.raw_history.remove(0);
    }
    self.raw_history.push(*e);

    // 2. Recalculate centroid as the normalized mean of the raw history
    //    (SpeakerTypes.swift:132-162).
    let mut mean = [0.0f32; EMBEDDING_DIM];
    for raw in &self.raw_history {
      for (m, r) in mean.iter_mut().zip(raw.iter()) {
        *m += *r;
      }
    }
    let count = self.raw_history.len() as f32;
    for m in &mut mean {
      *m /= count;
    }
    self.centroid = l2_normalize(&mean);

    // 3. EMA blend of the recalculated centroid with the new embedding
    //    (SpeakerTypes.swift:90-94).
    let mut blended = [0.0f32; EMBEDDING_DIM];
    for (b, (c, ei)) in blended.iter_mut().zip(self.centroid.iter().zip(e.iter())) {
      *b = EMA_ALPHA * *c + (1.0 - EMA_ALPHA) * *ei;
    }
    self.centroid = l2_normalize(&blended);

    // Metadata: accumulate duration (SpeakerTypes.swift:98).
    self.duration += duration;
  }
}

/// Sum of squares of a vector — FluidAudio's `vDSP_svesq`
/// (`Clustering/SpeakerManager.swift:446`).
pub(crate) fn sum_squares(v: &[f32; EMBEDDING_DIM]) -> f32 {
  v.iter().map(|x| x * x).sum()
}

/// L2-normalize with FluidAudio's *clamp* semantics
/// (`Offline/Utils/VDSPOperations.swift:12-23`): the norm is floored at
/// [`L2_NORM_EPSILON`] (`1e-12`) and the vector is scaled by its reciprocal,
/// rather than the vector being rejected. At the floor the two degenerate
/// cases diverge, mirroring Swift: a true zero vector stays zero
/// (`0 · 1e12 = 0`), but a nonzero vector whose norm is below the floor is
/// *amplified* by `1 / 1e-12` — e.g. `[5e-13, …] → [0.5, …]` — NOT returned
/// unchanged.
pub(crate) fn l2_normalize(v: &[f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
  let norm = sum_squares(v).sqrt().max(L2_NORM_EPSILON);
  let scale = 1.0 / norm;
  let mut out = [0.0f32; EMBEDDING_DIM];
  for (o, x) in out.iter_mut().zip(v.iter()) {
    *o = *x * scale;
  }
  out
}

/// Cosine *distance* — a faithful port of `SpeakerUtilities.cosineDistance`
/// (`Clustering/SpeakerOperations.swift:62-101`). Range `[0.0, 2.0]` (`0`
/// identical, `2` antipodal); `INFINITY` if either vector has zero magnitude.
///
/// The unit-norm fast path ([`UNIT_TOLERANCE`]) takes the raw dot product when
/// both squared norms are `≈ 1`, else divides by the magnitudes; the result is
/// clamped to `[-1, 1]` before `1 - similarity`. Reproduced rather than
/// delegated to [`Embedding::similarity`] because that method assumes unit
/// inputs and omits the clamp — this must match the Swift arithmetic for the
/// out-of-tree parity oracle.
pub(crate) fn cosine_distance(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
  let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
  let ssa = sum_squares(a);
  let ssb = sum_squares(b);
  if !(ssa > 0.0 && ssb > 0.0) {
    return f32::INFINITY;
  }
  let is_unit_a = (ssa - 1.0).abs() <= UNIT_TOLERANCE;
  let is_unit_b = (ssb - 1.0).abs() <= UNIT_TOLERANCE;
  let similarity = if is_unit_a && is_unit_b {
    dot
  } else {
    dot / (ssa.sqrt() * ssb.sqrt())
  };
  1.0 - similarity.clamp(-1.0, 1.0)
}
