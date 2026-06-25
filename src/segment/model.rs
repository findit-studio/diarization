//! ONNX Runtime wrapper for pyannote/segmentation-3.0 plus Layer-2
//! streaming convenience methods on [`Segmenter`].

use std::path::Path;

use ort::{
  ep::ExecutionProviderDispatch,
  session::{
    Session as OrtSession,
    builder::{GraphOptimizationLevel, SessionBuilder},
  },
  value::TensorRef,
};

use crate::segment::{
  error::Error,
  options::{FRAMES_PER_WINDOW, POWERSET_CLASSES, WINDOW_SAMPLES},
  segmenter::Segmenter,
  types::{Action, Event},
};

/// Builder for [`SegmentModel`] runtime configuration.
///
/// Default: optimization level [`GraphOptimizationLevel::Disable`]
/// (matches silero's choice — stable across ort versions), thread
/// counts left to ort's defaults, no execution providers beyond
/// ort's default search.
///
/// `serde` (feature-gated): `optimization_level` is bridged through a
/// snake-case wrapper enum because the foreign `GraphOptimizationLevel`
/// has no `Serialize`/`Deserialize` impl. `providers` is `serde(skip)`d
/// — execution-provider configuration is runtime-specific (CUDA /
/// CoreML / etc.) and not naturally serializable.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentModelOptions {
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

const fn default_optimization_level() -> GraphOptimizationLevel {
  GraphOptimizationLevel::Disable
}

const fn default_threads() -> usize {
  1
}

impl Default for SegmentModelOptions {
  fn default() -> Self {
    Self {
      optimization_level: default_optimization_level(),
      providers: Vec::new(),
      intra_threads: default_threads(),
      inter_threads: default_threads(),
    }
  }
}

impl SegmentModelOptions {
  /// Construct with all-default options.
  pub fn new() -> Self {
    Self::default()
  }

  /// Override the graph optimization level.
  pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = level;
    self
  }

  /// Configure execution providers in priority order. Default: ort's
  /// default execution-provider selection (typically CPU).
  ///
  /// **Caveat:** CoreML on macOS is known to degrade pyannote/segmentation-3.0
  /// numerics (see the design spec). Do not enable without measuring.
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

  /// Apply the option set to a `SessionBuilder`.
  fn apply(self, mut builder: SessionBuilder) -> Result<SessionBuilder, Error> {
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

/// Thin ort wrapper for one segmentation model session.
///
/// Owns one `ort::Session` plus reusable input scratch. Auto-derives
/// `Send`; does NOT auto-derive `Sync` because `ort::Session` is `!Sync`.
/// Use one per worker thread. Matches `silero::Session` exactly
/// (silero/src/session.rs line 61: "Send but not Sync").
///
/// **Shape validation:** v0.1.0 validates the model's output shape on first
/// inference (returns [`Error::InferenceShapeMismatch`] if `[589, 7]` is
/// violated). Load-time dimension verification (`Error::IncompatibleModel`)
/// is reserved for a future revision once a stable ort metadata API is
/// available.
pub struct SegmentModel {
  inner: OrtSession,
  input_scratch: Vec<f32>,
  /// Provenance source: the bundled artifact tag, the loaded-file
  /// basename, or `None` for an in-memory model. Feeds the dynamic
  /// version of [`Self::identity`].
  source: Option<String>,
}

impl SegmentModel {
  /// Build the [`SegmentModelOptions`] used by the no-arg constructors.
  ///
  /// Equivalent to [`SegmentModelOptions::default`] but additionally
  /// registers any execution providers compiled into the binary via the
  /// per-EP cargo features (CoreML, CUDA, TensorRT, DirectML, ROCm,
  /// OpenVINO, …). When no per-EP feature is enabled,
  /// [`crate::ep::auto_providers`] returns an empty list and behavior
  /// matches `SegmentModelOptions::default` exactly — the default
  /// build dispatches to ORT's CPU EP unchanged.
  ///
  /// Callers who want to override or disable provider auto-registration
  /// should construct an options struct explicitly and pass it through
  /// [`Self::from_file_with_options`] / [`Self::bundled_with_options`].
  fn default_options_with_auto_providers() -> SegmentModelOptions {
    SegmentModelOptions::default().with_providers(crate::ep::auto_providers())
  }

  /// Load the model from disk with default options.
  ///
  /// When a per-EP cargo feature (e.g. `coreml`, `cuda`) is enabled
  /// the matching execution provider is auto-registered at session
  /// creation; with no per-EP feature on, this is identical to
  /// `from_file_with_options(path, SegmentModelOptions::default())`.
  pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    Self::from_file_with_options(path, Self::default_options_with_auto_providers())
  }

  /// Load the model from disk with custom options.
  ///
  /// Bypasses provider auto-registration — the caller's `opts` (and
  /// thus the providers explicitly set via
  /// [`SegmentModelOptions::with_providers`]) are honored as-is.
  pub fn from_file_with_options<P: AsRef<Path>>(
    path: P,
    opts: SegmentModelOptions,
  ) -> Result<Self, Error> {
    let path = path.as_ref();
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    let basename = path.file_name().map(|n| n.to_string_lossy().into_owned());
    Ok(Self::new_from_session(session, basename))
  }

  /// Load the model from an in-memory ONNX byte buffer with default options.
  ///
  /// `bytes` is **copied** into ort's session; the buffer can be dropped
  /// immediately after this call returns.
  ///
  /// Default options auto-register per-EP-compiled execution providers.
  /// See [`Self::from_file`] for details.
  pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
    Self::from_memory_with_options(bytes, Self::default_options_with_auto_providers())
  }

  /// Load the model from an in-memory ONNX byte buffer with custom options.
  pub fn from_memory_with_options(bytes: &[u8], opts: SegmentModelOptions) -> Result<Self, Error> {
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder.commit_from_memory(bytes)?;
    Ok(Self::new_from_session(session, None))
  }

  /// Load the bundled `pyannote/segmentation-3.0` ONNX with default options.
  ///
  /// The model bytes are embedded into the compiled artifact via
  /// `include_bytes!` (gated on the `bundled-segmentation` cargo feature,
  /// which is on by default). No filesystem path or env var needed.
  ///
  /// Default options auto-register any execution providers compiled in
  /// via the per-EP cargo features (CoreML, CUDA, TensorRT, DirectML,
  /// ROCm, OpenVINO, …). See [`Self::from_file`] for the auto-register
  /// contract. With no per-EP feature on, dispatch is ORT-CPU as
  /// before.
  ///
  /// # Asymmetric default with embedding
  ///
  /// Segmentation's auto-register default is paired with an
  /// **explicit** default for embedding:
  /// [`crate::embed::EmbedModel::from_file`] does NOT auto-register
  /// EPs even when per-EP features are on. The reason is empirical:
  /// ORT's CoreML EP mistranslates the WeSpeaker ResNet34-LM graph
  /// and emits NaN/Inf on most inputs, while it handles the
  /// segmentation graph correctly. The asymmetry preserves the
  /// segmentation speedup without breaking the embedding pipeline.
  ///
  /// Source: `pyannote/segmentation-3.0` on HuggingFace, MIT-licensed —
  /// see `NOTICE` for attribution requirements.
  #[cfg(feature = "bundled-segmentation")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled-segmentation")))]
  pub fn bundled() -> Result<Self, Error> {
    Self::bundled_with_options(Self::default_options_with_auto_providers())
  }

  /// Load the bundled segmentation model with custom options.
  #[cfg(feature = "bundled-segmentation")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled-segmentation")))]
  pub fn bundled_with_options(opts: SegmentModelOptions) -> Result<Self, Error> {
    const BUNDLED_BYTES: &[u8] = include_bytes!("../../models/segmentation-3.0.onnx");
    let mut model = Self::from_memory_with_options(BUNDLED_BYTES, opts)?;
    model.source = Some(crate::provenance::BUNDLED_SEGMENTATION_VERSION.to_string());
    Ok(model)
  }

  fn new_from_session(session: OrtSession, source: Option<String>) -> Self {
    Self {
      inner: session,
      input_scratch: Vec::with_capacity(WINDOW_SAMPLES as usize),
      source,
    }
  }

  /// The segmentation identity for provenance stamping. Family is
  /// [`crate::provenance::SEGMENTATION_FAMILY`]; version is the bundled
  /// tag, the loaded basename, or the family name for an in-memory model.
  pub fn identity(&self) -> crate::provenance::ModelIdentity {
    let version = self
      .source
      .clone()
      .unwrap_or_else(|| crate::provenance::SEGMENTATION_FAMILY.to_string());
    crate::provenance::ModelIdentity::new(crate::provenance::SEGMENTATION_FAMILY, version)
  }

  /// Run inference on one 160 000-sample window. Returns the flattened
  /// `[FRAMES_PER_WINDOW * POWERSET_CLASSES] = [4123]` logits.
  ///
  /// Exposed for advanced callers who want to combine Layer 1's state
  /// machine with their own batching or scheduling.
  pub fn infer(&mut self, samples: &[f32]) -> Result<Vec<f32>, Error> {
    debug_assert_eq!(samples.len(), WINDOW_SAMPLES as usize);

    // Reject non-finite input at the boundary. The owned and streaming
    // offline paths feed `infer`'s output directly into `softmax_row` /
    // hard powerset argmax (bypassing the streaming `Segmenter`'s
    // `NonFiniteScores` guard), so a NaN sample propagating through
    // ORT into NaN logits would silently produce wrong diarization.
    if samples.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }

    self.input_scratch.clear();
    self.input_scratch.extend_from_slice(samples);

    // Use the first input and first output by position. pyannote/segmentation-3.0
    // is a single-input, single-output model; this avoids needing to know the
    // name and is robust to exporter-version naming differences.
    let outputs = self.inner.run(ort::inputs![TensorRef::from_array_view((
      [1usize, 1usize, WINDOW_SAMPLES as usize],
      self.input_scratch.as_slice()
    ),)?,])?;

    // Guard against zero-output sessions before positional indexing.
    // `outputs[0]` panics at the FFI boundary (ort's Index<usize>
    // panics for OOB), which would turn a malformed-model error into
    // a library-caller panic. A graceful typed error is the right
    // contract.
    let first_output = outputs
      .values()
      .next()
      .ok_or(Error::MissingInferenceOutput)?;
    let (shape, data) = first_output.try_extract_tensor::<f32>()?;
    // Validate the trailing two dims (frames, classes) BEFORE relying on
    // the row-major flattening. A model returning the same element count
    // in a different layout — e.g. `[1, POWERSET_CLASSES, FRAMES_PER_WINDOW]`
    // (axes swapped) or `[FRAMES_PER_WINDOW * POWERSET_CLASSES]` (rank
    // 1) — would otherwise pass the count check, and `push_inference`
    // would softmax groups of 7 values that are not class logits for
    // one frame, silently corrupting all speaker probabilities.
    let dims: &[i64] = shape.as_ref();
    let n_frames = FRAMES_PER_WINDOW as i64;
    let n_classes = POWERSET_CLASSES as i64;
    // Required canonical layout: `[*, FRAMES_PER_WINDOW, POWERSET_CLASSES]`
    // where `*` is one or more leading batch / channel dims.
    let layout_ok = if dims.len() >= 2 {
      dims[dims.len() - 2] == n_frames && dims[dims.len() - 1] == n_classes
    } else {
      false
    };
    if !layout_ok {
      return Err(Error::IncompatibleModel {
        tensor: "output",
        // `-1` matches the existing dynamic-batch convention used by
        // `Error::IncompatibleModel`.
        expected: &[-1, FRAMES_PER_WINDOW as i64, POWERSET_CLASSES as i64],
        got: dims.to_vec(),
      });
    }
    let expected = FRAMES_PER_WINDOW * POWERSET_CLASSES;
    if data.len() != expected {
      return Err(Error::InferenceShapeMismatch {
        expected,
        got: data.len(),
      });
    }
    // Reject non-finite logits before returning to the caller. The
    // owned and streaming offline paths immediately softmax these
    // values; a NaN here would propagate to a NaN-vs-NaN argmax and
    // produce arbitrary hard powerset labels (i.e. silent wrong
    // diarization). The streaming `Segmenter::push_inference` path
    // has its own `NonFiniteScores` guard, but `infer` is also a
    // public direct entrypoint and must enforce the contract itself.
    if data.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteOutput);
    }
    Ok(data.to_vec())
  }
}

impl Segmenter {
  /// Push samples and drive the state machine to a quiescent state by
  /// fulfilling each `NeedsInference` via `model.infer`. `emit` is called
  /// for every emitted [`Event`].
  ///
  /// This is the streaming entry point that mirrors
  /// `silero::Session::process_stream`.
  ///
  /// **Retry contract** (): if a previous call left a
  /// stashed inference (a transient `model.infer` failure or
  /// `Error::NonFiniteScores` from `push_inference`), this call
  /// retries the stash BEFORE pushing new audio. On a stash retry
  /// failure, the new `samples` are NOT appended — the caller can
  /// safely re-pass the same chunk without double-counting it. Mirror
  /// of the diarizer-level retry boundary.
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn process_samples<F>(
    &mut self,
    model: &mut SegmentModel,
    samples: &[f32],
    mut emit: F,
  ) -> Result<(), Error>
  where
    F: FnMut(Event),
  {
    if self.pending_inference.is_some() {
      self.drain(model, &mut emit)?;
    }
    self.push_samples(samples);
    self.drain(model, &mut emit)
  }

  /// Equivalent to `finish` followed by draining all remaining actions
  /// (running inference for any unprocessed window).
  ///
  /// Retries any stashed inference before calling `finish()` so that
  /// the segmenter is not left half-finished if the stash retry fails.
  /// `finish()` is idempotent, so re-driving `finish_stream` after a
  /// retryable error is safe.
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn finish_stream<F>(&mut self, model: &mut SegmentModel, mut emit: F) -> Result<(), Error>
  where
    F: FnMut(Event),
  {
    if self.pending_inference.is_some() {
      self.drain(model, &mut emit)?;
    }
    self.finish();
    self.drain(model, &mut emit)
  }

  fn drain<F>(&mut self, model: &mut SegmentModel, emit: &mut F) -> Result<(), Error>
  where
    F: FnMut(Event),
  {
    // Retry any stashed inference from a prior failed drain BEFORE
    // polling new actions. Without this, an `infer`/`push_inference`
    // failure popped `Action::NeedsInference`, returned Err, and lost
    // the in-flight `(WindowId, samples)` pair forever — `WindowId`
    // stayed in `pending` and finalization could stall.
    //
    // Two retryable failure modes share the stash, mirroring the
    // diarizer's `pending_seg_inference` semantics:
    //   1. `model.infer` returns Err   → transient backend failure.
    //   2. `model.infer` returns Ok but `push_inference` rejects the
    //      logits (e.g. `Error::NonFiniteScores`)
    //      → segmenter intentionally leaves `id` pending so the caller
    //         can retry with valid scores from a re-run.
    if let Some((id, samples)) = self.pending_inference.take() {
      match model.infer(&samples) {
        Ok(scores) => match self.push_inference(id, &scores) {
          Ok(()) => {}
          Err(e @ Error::NonFiniteScores { .. }) => {
            self.pending_inference = Some((id, samples));
            return Err(e);
          }
          Err(e) => return Err(e),
        },
        Err(e) => {
          self.pending_inference = Some((id, samples));
          return Err(e);
        }
      }
    }

    while let Some(action) = self.poll() {
      match action {
        Action::NeedsInference { id, samples } => {
          // Stash before invoking the model so a transient failure
          // (or non-finite logits) doesn't lose the action handle.
          //
          match model.infer(&samples) {
            Ok(scores) => match self.push_inference(id, &scores) {
              Ok(()) => {}
              Err(e @ Error::NonFiniteScores { .. }) => {
                self.pending_inference = Some((id, samples));
                return Err(e);
              }
              Err(e) => return Err(e),
            },
            Err(e) => {
              self.pending_inference = Some((id, samples));
              return Err(e);
            }
          }
        }
        Action::Activity(a) => emit(Event::Activity(a)),
        Action::VoiceSpan(r) => emit(Event::VoiceSpan(r)),
        // Layer 2 hides per-frame raw probabilities from the caller, the
        // same way it hides `NeedsInference`. Diarizer-grade callers that
        // need `SpeakerScores` use the Layer-1 `poll` API directly.
        Action::SpeakerScores { .. } => {}
      }
    }
    Ok(())
  }
}
