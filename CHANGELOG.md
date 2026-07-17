# 0.2.0

Raw-embedding hand-off: a desktop `segment+embed` node can now produce
raw WeSpeaker vectors plus the segmentation/count structure clustering
needs, and a separate `cluster` node clusters them — preserving the
raw-not-normalized PLDA invariant and reproducing the bundled
`StreamingOfflineDiarizer` output bit-for-bit (parity-gated).

ADDED

- `streaming::StreamingEmbedder` — public memory-fused segment+embed
  node; `push(abs_start, samples) -> streaming::RangeEmbeddings`,
  `finish()`. Shares the fused per-range body with
  `StreamingOfflineDiarizer::push_voice_range`; emits RAW, unnormalized
  WeSpeaker vectors (never L2-normalized).
- `streaming::RangeEmbeddings` — public per-VAD-range carrier (raw
  WeSpeaker vectors + segmentation activity + count + timing + absolute
  start) for a split segment+embed → cluster topology, with a validated
  constructor (`RangeShapeError`) and accessors.
- `streaming::cluster_ranges(&[RangeEmbeddings], &PldaTransform, &StreamingOfflineOptions)`
  — public reduce-at-close clustering entry point; runs the single
  global pyannote `cluster_vbx` pass and reproduces the bundled
  `StreamingOfflineDiarizer` output exactly (shared
  `cluster_ranges_inner`, RNG-free offline path).
- `plda::RawEmbedding::from_wespeaker` + `as_array` — public,
  constructible raw (unnormalized) PLDA-input carrier; reuses the
  existing finite + min-norm validation.
- `provenance` module — `ModelIdentity` value type + family/version
  constants; `PldaTransform::identity`, `EmbedModel::identity` (+
  `source_basename`), `SegmentModel::identity`, so a downstream pipeline
  reads the loaded model's identity instead of hardcoding it.

CHANGED

- `StreamingOfflineDiarizer` internally accumulates `RangeEmbeddings`
  carriers and `finalize` delegates to the shared clustering core (pure
  relocation — no behavior change; streaming + offline unit suites and
  the 01_dialogue e2e parity remain green).


# UNRELEASED

BREAKING (pre-1.0):

- `diarization::embed::Error` is now `#[non_exhaustive]`. Callers
  with exhaustive `match` arms must add a `_ =>` wildcard. The
  attribute is forward-looking — variants in this enum represent
  low-level numerical / boundary conditions whose set evolves as
  new failure modes are surfaced or as internal kernels stop
  emitting one. The attribute lets future variant additions /
  retirements stay non-breaking after this point.
- `diarization::embed::Error::Fbank(String)` variant removed. The
  variant was tied to the previous `kaldi-native-fbank` C++ backend,
  which has been replaced by an in-tree torchaudio-compliance fbank
  port (no `Result<_, String>` boundary to wrap). Code that matched
  the variant directly will not compile.
- `silero-vad` Cargo feature removed. The `silero` crate was only
  consumed by `examples/run_streaming_pipeline.rs` and is now a
  `[dev-dependencies]` entry — no feature gate needed. Build the
  streaming example with just `--features ort,bundled-segmentation`.
  Bumped `silero` from 0.3 → 0.4 (its segmenter API changed:
  `process_samples`/`finish_stream` no longer take a closure;
  example updated to drain the pending-segment queue instead).


The pyannote-community-1 offline + streaming-offline pipelines now
ship in full: VBx clustering, PLDA, AHC, centroid + Hungarian
assignment, reconstruction, RTTM emission. The crate exposes both
the offline pipeline (one-shot batch) and the streaming-offline
variant (push voice ranges, finalize once at the end). End-to-end
DER vs pyannote 4.0.4 on the in-repo fixture suite is ≤ 0.4% on the
worst clip and bit-exact on the rest.

PUBLIC SURFACE

- **`diarization::offline`** — `OfflineInput` / `diarize_offline`:
  caller-supplied segmentation + embedding tensors → diarization +
  RTTM spans. No ORT inference inside; pair with `OwnedDiarizationPipeline`
  (under `feature = "ort"`) for the full audio entrypoint.
- **`diarization::streaming::StreamingOfflineDiarizer`** — push
  voice ranges incrementally via `push_voice_range(&mut seg, &mut emb,
  ...)`, call `finalize(&plda)` once to produce RTTM spans. Same
  numerics as `diarize_offline` modulo plumbing.
- **`diarization::segment`** — `SegmentModel::bundled()` /
  `from_file` / `from_memory` (default + `_with_options` variants);
  segmentation-3.0 ONNX is embedded via `include_bytes!` under the
  default `bundled-segmentation` feature.
- **`diarization::embed`** — `EmbedModel::from_file` /
  `from_memory` (and `from_torchscript_file` under `feature = "tch"`).
  WeSpeaker ResNet34-LM is BYO; fetch it from
  `FinDIT-Studio/dia-models` on HuggingFace. The single-file packed
  ONNX is the canonical form.
- **`diarization::plda`** — `PldaTransform::new()` (no args), re-exported
  from `diaric`, which embeds the weights via `include_bytes!`; CC-BY-4.0
  with attribution preserved in `diaric`'s `NOTICE` and
  `models/plda/SOURCE.md`.
- **`diarization::cluster`** — `ahc`, `vbx`, `centroid`, `hungarian`
  submodules expose the algorithmic primitives directly for callers
  who want to wire their own pipeline.
- **`diarization::pipeline::assign_embeddings`** — the AHC + VBx +
  centroid + Hungarian core, callable on already-projected
  post-PLDA features.
- **`diarization::reconstruct`** — discrete grid + RTTM span emission
  + `try_discrete_to_spans` (fallible variant for direct callers).
- **`diarization::aggregate::count_pyannote`** — overlap-add per-frame
  speaker-count tensor, bit-exact with pyannote.
- **`diarization::ep`** — opt-in ORT execution providers (CoreML,
  CUDA, TensorRT, DirectML, ROCm, OpenVINO, WebGPU, …) gated by
  per-EP cargo features and the `gpu` meta-feature. `auto_providers()`
  helper picks compiled-in EPs at runtime.
- **`diarization::spill`** — `SpillOptions` + `SpillBytes` /
  `SpillBytesMut` for file-backed mmap fallback above the
  configurable threshold; protects multi-hour inputs from
  OOM-aborting the pipeline.

ASYMMETRIC EP DEFAULT

- `SegmentModel::bundled()` / `::from_file()` auto-register
  `dia::ep::auto_providers()` so any compiled-in per-EP feature
  accelerates segmentation with no caller code change.
- `EmbedModel::from_file()` does **NOT** auto-register EPs.
  Empirically, ORT's CoreML EP miscompiles the WeSpeaker
  ResNet34-LM graph and emits NaN/Inf on most realistic inputs
  across every CoreML compute unit / model format / static-shape
  knob; auto-on would crash the embed pipeline. Callers on a vetted
  EP host opt in via `EmbedModelOptions::default().with_providers(...)`
  and `EmbedModel::from_file_with_options(path, opts)`. See
  `crate::ep` and `crate::embed::EmbedModel::from_file` docs.

CORRECTNESS GUARANTEES

- **Bit-exact pyannote 4.0.4 parity** on the in-repo fixture suite
  (01_dialogue, 02_pyannote_sample, 03_dual_speaker,
  04_three_speaker, 05_four_speaker — DER 0.0000–0.0037; 06_long_recording
  is `#[ignore]`d at the strict bit-level due to GEMM-roundoff drift
  past T=1004 but the per-frame coverage at DER 0.0019 is the
  release-blocking metric).
- **SpillBytes / SpillBytesMut** are `Send + Sync`; the runtime EP
  registration is per-session.
- **Cross-platform** spill: `posix_fallocate` on Linux,
  `F_PREALLOCATE` on macOS, `SetFileValidData`/`SetEndOfFile` on
  Windows; reservations happen before any mapped writes so we
  never `SIGBUS` on `ENOSPC` mid-run.

TESTING

- 495 lib unit tests pass on default features; full DER suite
  (in-repo + speakrs clips) at the bit-exact baseline.
- Parity tests under `src/*/parity_tests.rs` skip cleanly via
  `parity_fixtures_or_skip!` when `tests/parity/fixtures/` is
  absent (the published crate tarball excludes the fixtures to
  stay under the 10 MiB crates.io limit).
- `tests/parity/run.sh` is a manual harness for end-to-end DER
  validation against pyannote-on-disk; provide your own clip path
  if running outside a workspace checkout.

BUILD

- Rust edition 2024, MSRV 1.95.
- `nalgebra 0.34`, `kodama 0.3` (AHC linkage), `kaldi-native-fbank 0.1`,
  `pathfinding 4.15` (Hungarian), `mediatime`, `thiserror`,
  `memmapix 0.9` + `bytemuck 1` + `tempfile 3` + `fs4 1` for the
  spill backend, `rustix` (Linux/Android only, for `O_TMPFILE`).
- Optional features: `serde`, `tch`, `silero-vad`, plus 16 per-EP
  features (`coreml`, `cuda`, `tensorrt`, `directml`, `rocm`,
  `migraphx`, `openvino`, `webgpu`, `xnnpack`, `onednn`, `cann`,
  `acl`, `qnn`, `nnapi`, `tvm`, `azure`) and a `gpu` meta-feature.

KNOWN LIMITATIONS / DEFERRED

- WeSpeaker embed model (~26 MiB) exceeds the crates.io 10 MiB
  hard limit; not bundled. Fetch from
  `FinDIT-Studio/dia-models` on HuggingFace at the pinned revision
  documented in `scripts/download-embed-model.sh`, or set
  `DIA_EMBED_MODEL_PATH` if you keep the model elsewhere.
- ORT CoreML EP cannot run the WeSpeaker graph correctly; the
  asymmetric default (seg-auto, embed-CPU) ships as the workaround.
- FP16 / INT8 ONNX variants and TensorRT / OpenVINO IR / CoreML
  `.mlpackage` formats are not provided; the canonical FP32
  single-file ONNX runs on every ORT EP that doesn't have the
  WeSpeaker miscompile.
- 06_long_recording (T=1004) hits a GEMM-roundoff partition drift
  at the strict bit-exact level; tolerant per-frame coverage is in
  `reconstruct::parity_tests::reconstruct_within_tolerance_06_long_recording`.

# 0.1.0 (2026-04-26)

Initial release. Ships the `diarization::segment` module — Sans-I/O speaker
segmentation backed by `pyannote/segmentation-3.0` ONNX.

FEATURES

- **Sans-I/O state machine** (`diarization::segment::Segmenter`) with no `ort`
  dependency. Caller pumps audio in via `push_samples`, drains `Action`s
  via `poll`, runs ONNX inference externally, and pushes scores back via
  `push_inference`. The state machine is exercisable in unit tests with
  synthetic scores — no model file required.
- **Layer 2 streaming driver** (`Segmenter::process_samples` and
  `finish_stream`) gated on the default `ort` feature. Mirrors silero's
  `Session::process_stream` callback idiom.
- **`SegmentModel`** wraps `ort::Session` for `pyannote/segmentation-3.0`
  with `from_file`, `from_memory`, and `*_with_options` constructors.
- **`SegmentModelOptions`** builder for `GraphOptimizationLevel`,
  `ExecutionProviderDispatch`, intra/inter thread counts. Both `ort`
  types are re-exported from `diarization::segment`.
- **`mediatime`-based time types** (`TimeRange`, `Timestamp`, `Duration`)
  for every sample range and duration crossing the public API.
- **Sliding-window scheduling** with configurable step (default 2.5 s)
  and tail-anchored window for end-of-stream coverage.
- **Powerset decoding** (7-class → 3-speaker additive marginals + voice
  probability), **per-frame voice-timeline stitching** (overlap-add mean,
  ~1.7 MB/hour storage), **streaming hysteresis** with onset/offset
  thresholds, **window-local `SpeakerActivity`** records, and
  **`voice_merge_gap`** post-processing.

CORRECTNESS GUARANTEES

- **Generation-counter `WindowId`** (process-wide `AtomicU64`): stale
  inference results from before a `clear()` and cross-`Segmenter` ID
  collisions both reject as `Error::UnknownWindow`.
- **Pending-aware finalization boundary**: out-of-order `push_inference`
  cannot prematurely finalize frames whose other contributing windows
  haven't yet reported.
- **Tail-window activity clamping** to `total_samples`.
- **Frame-to-sample conversion** uses integer-rounded division
  (`frame_to_sample`) bit-for-bit equivalent to Python's
  `int(round(...))`. **Sample-to-frame conversion** uses floor
  (`frame_index_of`) for boundary safety.

OBSERVABILITY

- `Segmenter::pending_inferences()` and `Segmenter::buffered_samples()`
  introspection for backpressure detection.
- Compile-time `Send + Sync` assertion on `Segmenter`; compile-time `Send`
  assertion on `SegmentModel` (which is `!Sync` because `ort::Session` is).

EXAMPLES, TESTS, BENCHES

- `examples/stream_layer1.rs`: Sans-I/O usage with synthetic inferencer
  (no model file, no `ort` feature).
- `examples/stream_from_wav.rs`: full Layer-2 pipeline streaming a 16 kHz
  mono WAV file in 100 ms chunks.
- `tests/integration_segment.rs`: gated `#[ignore]` smoke test against a
  real downloaded model.
- `benches/segment.rs`: Layer-1 throughput on one minute of audio.
- 54 unit tests covering options, powerset, hysteresis, RLE, sliding-window
  planning, per-frame stitching, segmenter end-to-end, out-of-order
  `push_inference`, cross-`Segmenter` ID collision, stale-id rejection,
  empty-stream handling, tail-window activity clamping.

BUILD

- Edition 2024, Rust 1.95.
- Default features `["std", "ort"]`. `--no-default-features --features
  std` builds without `ort` and exposes only Layer 1.
- Lints aligned with sibling crates (silero, soundevents, scenesdetect,
  mediatime).

KNOWN LIMITATIONS

- **No load-time ONNX shape verification.** The `ort` 2.0.0-rc.12 metadata
  API doesn't expose dimensions in a way matching the spec's assumption;
  shape mismatches surface on first inference as
  `Error::InferenceShapeMismatch`. The `Error::IncompatibleModel` variant
  is reserved for the eventual load-time check. Matches silero's pragmatic
  stance.
- **Sample-rate is the caller's responsibility.** `push_samples` accepts
  `&[f32]` without validating that the input is 16 kHz mono. Feeding the
  wrong rate produces silently corrupted output.
- **No bundled model.** Run `scripts/download-model.sh` to fetch
  `pyannote/segmentation-3.0` from Hugging Face.

DEFERRED FOR v0.2

- `diarization::embed` module (speaker embedding via WeSpeaker ResNet34).
- `infer_batch` for cross-stream batching, `IoBinding`-based
  reusable-output-buffer fast path, `Arc<[f32]>` in `Action::NeedsInference`.
- `serde` derives on output types.
- `step_samples` typed as `Duration`.
- Soft-cap `try_push_samples` for backpressure enforcement.
- Bundled model behind a Cargo feature.
- F1 numerical-parity tests vs `pyannote.audio`.
