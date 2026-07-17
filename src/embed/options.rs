//! ort model options for `diarization::embed`, plus the re-exported
//! pyannote fbank / embedding geometry constants.

// The pyannote fbank / embedding constants (window sizes, embedding dim,
// mel bins, frame counts, normalization epsilon, sample rate) are defined
// in `diaric::embed`; re-exported here so the ort/tch model runners in this
// crate keep referencing `crate::embed::options::*` unchanged.
pub use diaric::embed::{
  EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS, HOP_SAMPLES, MIN_CLIP_SAMPLES,
  NORM_EPSILON, SAMPLE_RATE_HZ,
};

// ── EmbedModelOptions ─────────────────────────────────────────────────────

#[cfg(feature = "ort")]
use ort::ep::ExecutionProviderDispatch;
#[cfg(feature = "ort")]
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

/// Builder for [`EmbedModel`](crate::embed::EmbedModel) runtime configuration.
///
/// Mirrors [`SegmentModelOptions`](crate::segment::SegmentModelOptions): the
/// same four ort knobs (graph optimization level, execution providers,
/// intra/inter-op thread counts), with both consuming `with_*` and
/// in-place `set_*` builders.
///
/// Default: ort defaults for optimization level and threading, no
/// execution providers configured beyond ort's default search.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbedModelOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_optimization_level",
      with = "crate::ort_serde::graph_optimization_level"
    )
  )]
  optimization_level: GraphOptimizationLevel,
  #[cfg_attr(feature = "serde", serde(skip, default))]
  providers: Vec<ExecutionProviderDispatch>,
  #[cfg_attr(feature = "serde", serde(default = "default_threads"))]
  intra_threads: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_threads"))]
  inter_threads: usize,
}

#[cfg(feature = "ort")]
const fn default_optimization_level() -> GraphOptimizationLevel {
  GraphOptimizationLevel::Disable
}

#[cfg(feature = "ort")]
const fn default_threads() -> usize {
  1
}

#[cfg(feature = "ort")]
impl Default for EmbedModelOptions {
  fn default() -> Self {
    Self {
      optimization_level: default_optimization_level(),
      providers: Vec::new(),
      intra_threads: default_threads(),
      inter_threads: default_threads(),
    }
  }
}

#[cfg(feature = "ort")]
impl EmbedModelOptions {
  /// Construct with all-default options.
  pub fn new() -> Self {
    Self::default()
  }

  // ── Builder (consuming with_*) ───────────────────────────────────────

  /// Override the graph optimization level.
  pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = level;
    self
  }

  /// Configure execution providers in priority order. Default: ort's
  /// default execution-provider selection (typically CPU).
  ///
  /// **Caveat:** non-CPU providers may degrade WeSpeaker ResNet34 numerics
  /// and break the byte-determinism guarantees in spec §11.9. Do not enable
  /// without measuring against the pyannote parity harness (Task 46).
  pub fn with_providers(mut self, providers: Vec<ExecutionProviderDispatch>) -> Self {
    self.providers = providers;
    self
  }

  /// Override `intra_threads`. Default is `1` for bit-exact
  /// reproducibility across runs (parallel reductions are not
  /// deterministic).
  pub fn with_intra_threads(mut self, n: usize) -> Self {
    self.intra_threads = n;
    self
  }

  /// Override `inter_threads`. Default is `1`.
  pub fn with_inter_threads(mut self, n: usize) -> Self {
    self.inter_threads = n;
    self
  }

  // ── Mutators (in-place set_*) ────────────────────────────────────────

  /// Set the graph optimization level (in-place).
  pub fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = level;
    self
  }

  /// Set the execution providers (in-place).
  pub fn set_providers(&mut self, providers: Vec<ExecutionProviderDispatch>) -> &mut Self {
    self.providers = providers;
    self
  }

  /// Set `intra_threads` (in-place).
  pub fn set_intra_threads(&mut self, n: usize) -> &mut Self {
    self.intra_threads = n;
    self
  }

  /// Set `inter_threads` (in-place).
  pub fn set_inter_threads(&mut self, n: usize) -> &mut Self {
    self.inter_threads = n;
    self
  }

  // ── Internal apply ───────────────────────────────────────────────────

  /// Apply the option set to a `SessionBuilder`. Used internally by
  /// [`EmbedModel`](crate::embed::EmbedModel).
  pub(crate) fn apply(
    self,
    mut builder: SessionBuilder,
  ) -> Result<SessionBuilder, crate::embed::Error> {
    builder = builder
      .with_optimization_level(self.optimization_level)
      .map_err(ort::Error::from)?;
    builder = builder
      .with_intra_threads(self.intra_threads)
      .map_err(ort::Error::from)?;
    builder = builder
      .with_inter_threads(self.inter_threads)
      .map_err(ort::Error::from)?;
    if !self.providers.is_empty() {
      builder = builder
        .with_execution_providers(self.providers)
        .map_err(ort::Error::from)?;
    }
    Ok(builder)
  }
}
