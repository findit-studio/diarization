//! Options for [`OnlineClusterer`](super::OnlineClusterer) — the tunable
//! surface of FluidAudio's `SpeakerManager` greedy online matcher.
//!
//! Every default and every gate is cited to the FluidAudio source it ports
//! (repo `FluidAudio`, paths under `Sources/FluidAudio/Diarizer/`). Port from
//! source, not summaries: the two thresholds gate *different* decisions (see
//! each field), and the production `DiarizerManager` derives both from ONE
//! base `clusteringThreshold` (`Core/DiarizerManager.swift:29,32`).

/// Default assignment threshold — the `speakerThreshold` a bare
/// `SpeakerManager()` is constructed with
/// (`Clustering/SpeakerManager.swift:46`).
///
/// A cosine *distance* (`0` identical … `2` antipodal), NOT a similarity.
pub const DEFAULT_SPEAKER_THRESHOLD: f32 = 0.65;

/// Default embedding-update threshold — the `embeddingThreshold` a bare
/// `SpeakerManager()` is constructed with
/// (`Clustering/SpeakerManager.swift:47`). A cosine distance.
pub const DEFAULT_EMBEDDING_THRESHOLD: f32 = 0.45;

/// Default minimum speech duration (seconds) to spawn a new speaker — the
/// `minSpeechDuration` a bare `SpeakerManager()` is constructed with
/// (`Clustering/SpeakerManager.swift:48`).
pub const DEFAULT_MIN_SPEECH_DURATION: f32 = 1.0;

/// Upper bound for a cosine-distance threshold. `cosine_distance` returns
/// `1 - clamp(cos, -1, 1)` (`Clustering/SpeakerOperations.swift:99-100`), so
/// its codomain is `[0.0, 2.0]`; a threshold outside that can never sit on a
/// real decision boundary.
const MAX_THRESHOLD_DISTANCE: f32 = 2.0;

/// Predicate for a valid cosine-distance threshold: finite and within
/// `[0.0, MAX_THRESHOLD_DISTANCE]`. Shared so the panicking setters and the
/// fallible [`OnlineClusterer::try_new`](super::OnlineClusterer::try_new)
/// construction guard apply the *identical* rule — a serde-deserialized config
/// that bypasses the setters is still checked against the same bound, with no
/// second predicate to drift out of sync.
#[inline]
pub(crate) fn threshold_in_range(v: f32) -> bool {
  v.is_finite() && (0.0..=MAX_THRESHOLD_DISTANCE).contains(&v)
}

/// Predicate for a valid minimum speech duration: finite and non-negative.
/// Shared with [`OnlineClusterer::try_new`](super::OnlineClusterer::try_new)
/// for the same reason as [`threshold_in_range`].
#[inline]
pub(crate) fn duration_in_range(v: f32) -> bool {
  v.is_finite() && v >= 0.0
}

/// Range check shared by both threshold setters. Mirrors
/// [`crate::cluster`]'s `OfflineClusterOptions` builder-panic idiom (an
/// out-of-range knob is a caller bug, surfaced eagerly).
#[inline]
fn validate_threshold(v: f32) {
  assert!(
    threshold_in_range(v),
    "threshold must be a finite cosine distance in [0.0, 2.0]; got {v}"
  );
}

/// Range check for the duration setter: a finite, non-negative number of
/// seconds.
#[inline]
fn validate_duration(v: f32) {
  assert!(
    duration_in_range(v),
    "min_speech_duration must be finite and >= 0.0 seconds; got {v}"
  );
}

/// Configuration for [`OnlineClusterer`](super::OnlineClusterer).
///
/// Ports the three fields of `SpeakerManager` that its assignment path
/// actually consults (`Clustering/SpeakerManager.swift:135-177`). The struct's
/// fourth field, `minEmbeddingUpdateDuration`, is **deliberately not ported**:
/// it is stored (`SpeakerManager.swift:43`) but never read anywhere in the
/// assign / update path — the embedding-update decision is gated purely by
/// distance and vector quality, not by duration (see
/// [`Self::embedding_threshold`]). Porting it would add a knob that changes
/// nothing.
///
/// # The two thresholds gate different things
/// - [`speaker_threshold`](Self::speaker_threshold): *assignment* — reuse the
///   nearest existing speaker vs. spawn a new one.
/// - [`embedding_threshold`](Self::embedding_threshold): *centroid update* —
///   whether an already-assigned segment is close enough to fold into the
///   speaker's running centroid.
///
/// Because the update threshold is (by default) the smaller of the two, there
/// is a band `[embedding_threshold, speaker_threshold)` where a segment is
/// assigned to a speaker but does NOT move its centroid.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OnlineClusterOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_speaker_threshold"))]
  speaker_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_embedding_threshold"))]
  embedding_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_min_speech_duration"))]
  min_speech_duration: f32,
}

#[cfg(feature = "serde")]
const fn default_speaker_threshold() -> f32 {
  DEFAULT_SPEAKER_THRESHOLD
}
#[cfg(feature = "serde")]
const fn default_embedding_threshold() -> f32 {
  DEFAULT_EMBEDDING_THRESHOLD
}
#[cfg(feature = "serde")]
const fn default_min_speech_duration() -> f32 {
  DEFAULT_MIN_SPEECH_DURATION
}

impl Default for OnlineClusterOptions {
  /// The bare `SpeakerManager()` defaults
  /// (`Clustering/SpeakerManager.swift:45-55`): `0.65` / `0.45` / `1.0`.
  ///
  /// Note this is NOT the production `DiarizerManager` wiring, which derives
  /// the thresholds from `clusteringThreshold = 0.7`
  /// (`Core/DiarizerTypes.swift:9`) as `0.7 × 1.2 = 0.84` and `0.7 × 0.8 =
  /// 0.56` — reproduce that with [`Self::from_clustering_threshold`].
  fn default() -> Self {
    Self {
      speaker_threshold: DEFAULT_SPEAKER_THRESHOLD,
      embedding_threshold: DEFAULT_EMBEDDING_THRESHOLD,
      min_speech_duration: DEFAULT_MIN_SPEECH_DURATION,
    }
  }
}

impl OnlineClusterOptions {
  /// Construct with the bare-`SpeakerManager()` defaults (see [`Default`]).
  pub fn new() -> Self {
    Self::default()
  }

  /// Construct the way production `DiarizerManager` does
  /// (`Core/DiarizerManager.swift:23-35`): from a single base
  /// `clusteringThreshold`, deriving
  /// - `speaker_threshold = base × 1.2` (`DiarizerManager.swift:29` — the
  ///   inline comment there says "0.9x" but the code multiplies by `1.2`; the
  ///   code is truth), and
  /// - `embedding_threshold = base × 0.8` (`DiarizerManager.swift:32`).
  ///
  /// `min_speech_duration` keeps its default; production passes
  /// `config.minSpeechDuration`, whose own default is also `1.0`
  /// (`Core/DiarizerTypes.swift:12`). Passing `base = 0.7` reproduces the
  /// shipping FluidAudio diarizer's thresholds (`0.84` / `0.56`).
  ///
  /// # Panics
  /// Panics if either derived threshold is non-finite or outside `[0.0, 2.0]`
  /// (e.g. `base > 1.666…` overflows `speaker_threshold` past `2.0`).
  pub fn from_clustering_threshold(base: f32) -> Self {
    Self::default()
      .with_speaker_threshold(base * 1.2)
      .with_embedding_threshold(base * 0.8)
  }

  // ── Accessors ──────────────────────────────────────────────────────────

  /// Maximum cosine distance to the nearest existing speaker for a segment to
  /// be *assigned* to it rather than spawning a new speaker. The comparison
  /// is strict `<` (`Clustering/SpeakerManager.swift:152`).
  pub fn speaker_threshold(&self) -> f32 {
    self.speaker_threshold
  }

  /// Maximum cosine distance (to the matched speaker's centroid) for an
  /// assigned segment to be folded into that centroid. Strictly closer than
  /// this → the running centroid is updated; at or beyond it → only the
  /// speaker's accumulated duration changes
  /// (`Clustering/SpeakerManager.swift:444-459`). Strict `<`.
  pub fn embedding_threshold(&self) -> f32 {
    self.embedding_threshold
  }

  /// Minimum speech duration (seconds) for a non-matching segment to spawn a
  /// new speaker; shorter non-matching segments are dropped. The comparison
  /// is `>=` (`Clustering/SpeakerManager.swift:164`).
  pub fn min_speech_duration(&self) -> f32 {
    self.min_speech_duration
  }

  // ── Builders (consuming with_*) ────────────────────────────────────────

  /// Set the assignment threshold (builder).
  ///
  /// # Panics
  /// Panics if `t` is non-finite or outside `[0.0, 2.0]`.
  pub fn with_speaker_threshold(mut self, t: f32) -> Self {
    validate_threshold(t);
    self.speaker_threshold = t;
    self
  }

  /// Set the embedding-update threshold (builder).
  ///
  /// # Panics
  /// Panics if `t` is non-finite or outside `[0.0, 2.0]`.
  pub fn with_embedding_threshold(mut self, t: f32) -> Self {
    validate_threshold(t);
    self.embedding_threshold = t;
    self
  }

  /// Set the minimum new-speaker speech duration in seconds (builder).
  ///
  /// # Panics
  /// Panics if `d` is non-finite or negative.
  pub fn with_min_speech_duration(mut self, d: f32) -> Self {
    validate_duration(d);
    self.min_speech_duration = d;
    self
  }

  // ── Mutators (in-place set_*) ──────────────────────────────────────────

  /// Set the assignment threshold (in-place).
  ///
  /// # Panics
  /// Panics if `t` is non-finite or outside `[0.0, 2.0]`.
  pub fn set_speaker_threshold(&mut self, t: f32) -> &mut Self {
    validate_threshold(t);
    self.speaker_threshold = t;
    self
  }

  /// Set the embedding-update threshold (in-place).
  ///
  /// # Panics
  /// Panics if `t` is non-finite or outside `[0.0, 2.0]`.
  pub fn set_embedding_threshold(&mut self, t: f32) -> &mut Self {
    validate_threshold(t);
    self.embedding_threshold = t;
    self
  }

  /// Set the minimum new-speaker speech duration in seconds (in-place).
  ///
  /// # Panics
  /// Panics if `d` is non-finite or negative.
  pub fn set_min_speech_duration(&mut self, d: f32) -> &mut Self {
    validate_duration(d);
    self.min_speech_duration = d;
    self
  }
}

#[cfg(test)]
mod validation_tests {
  use super::*;

  #[test]
  fn defaults_match_bare_speaker_manager() {
    let o = OnlineClusterOptions::new();
    assert_eq!(o.speaker_threshold(), 0.65);
    assert_eq!(o.embedding_threshold(), 0.45);
    assert_eq!(o.min_speech_duration(), 1.0);
  }

  #[test]
  fn from_clustering_threshold_applies_diarizer_manager_ratios() {
    // DiarizerManager.swift:29,32 — speaker = base×1.2, embedding = base×0.8.
    let o = OnlineClusterOptions::from_clustering_threshold(0.7);
    assert!(
      (o.speaker_threshold() - 0.84).abs() < 1e-6,
      "{}",
      o.speaker_threshold()
    );
    assert!(
      (o.embedding_threshold() - 0.56).abs() < 1e-6,
      "{}",
      o.embedding_threshold()
    );
    // min_speech_duration keeps its default (production also passes 1.0).
    assert_eq!(o.min_speech_duration(), 1.0);
  }

  #[test]
  #[should_panic(expected = "threshold must be a finite cosine distance")]
  fn speaker_threshold_nan_panics() {
    let _ = OnlineClusterOptions::new().with_speaker_threshold(f32::NAN);
  }

  #[test]
  #[should_panic(expected = "threshold must be a finite cosine distance")]
  fn speaker_threshold_above_two_panics() {
    let _ = OnlineClusterOptions::new().with_speaker_threshold(2.5);
  }

  #[test]
  #[should_panic(expected = "threshold must be a finite cosine distance")]
  fn embedding_threshold_negative_panics() {
    let _ = OnlineClusterOptions::new().with_embedding_threshold(-0.1);
  }

  #[test]
  #[should_panic(expected = "threshold must be a finite cosine distance")]
  fn from_clustering_threshold_overflow_panics() {
    // 2.0 × 1.2 = 2.4 > 2.0 → the derived speaker_threshold is rejected.
    let _ = OnlineClusterOptions::from_clustering_threshold(2.0);
  }

  #[test]
  #[should_panic(expected = "min_speech_duration must be finite and >= 0.0")]
  fn min_speech_duration_negative_panics() {
    let _ = OnlineClusterOptions::new().with_min_speech_duration(-1.0);
  }

  #[test]
  #[should_panic(expected = "min_speech_duration must be finite and >= 0.0")]
  fn min_speech_duration_infinite_panics() {
    let _ = OnlineClusterOptions::new().with_min_speech_duration(f32::INFINITY);
  }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
  use super::*;

  #[test]
  fn default_roundtrips_through_json() {
    let opts = OnlineClusterOptions::new();
    let json = serde_json::to_string(&opts).expect("serialize");
    let back: OnlineClusterOptions = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(opts, back);
  }

  #[test]
  fn partial_json_fills_defaults() {
    // Only one field present; the other two come from the serde defaults.
    let json = r#"{"speaker_threshold": 0.9}"#;
    let opts: OnlineClusterOptions = serde_json::from_str(json).expect("deserialize");
    assert_eq!(opts.speaker_threshold(), 0.9);
    assert_eq!(opts.embedding_threshold(), DEFAULT_EMBEDDING_THRESHOLD);
    assert_eq!(opts.min_speech_duration(), DEFAULT_MIN_SPEECH_DURATION);
  }
}
