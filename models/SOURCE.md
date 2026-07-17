# Bundled model files

`dia` compiles the `pyannote/segmentation-3.0` model into its binary via
`include_bytes!` (below). The PLDA whitening weights are embedded through
the `diaric` dependency, not this crate's own tree. Downstream
redistributors must reproduce the attributions in `NOTICE` (MIT for the
bundled segmentation model) and in `diaric`'s `NOTICE` (CC-BY-4.0 for the
PLDA weights).

## `segmentation-3.0.onnx`

The 16 kHz 7-class powerset speaker-segmentation network from
`pyannote/segmentation-3.0`. Embedded by
`SegmentModel::bundled()` when the `bundled-segmentation` cargo
feature is enabled (default-on). Off-switch: callers who BYO a
fine-tuned variant turn off `default-features` and use
`SegmentModel::from_file` / `from_memory`.

- **License:** MIT (CNRS / Hervé Bredin)
- **Source:** <https://huggingface.co/pyannote/segmentation-3.0>
- **Original layout:** `pytorch_model.onnx` in the HF repo (renamed
  on download).
- **SHA-256:** `057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25`
- **Size:** 5 986 908 bytes (~5.99 MiB), gzip ~5.28 MiB.

`scripts/download-model.sh` mirrors the upstream snapshot for callers
who disable bundling. Refreshing the bundled file: re-run the script
into `models/segmentation-3.0.onnx`, update the SHA-256 above, and
re-run `cargo test`.

## PLDA weights (embedded via the `diaric` dependency)

The PLDA whitening weights from
`pyannote/speaker-diarization-community-1` are no longer part of this
crate: `diaric` owns `models/plda/` and embeds the blobs via
`include_bytes!` in `diaric::plda`. See diaric's `models/plda/SOURCE.md`
for the full provenance + refresh procedure.

- **License:** CC-BY-4.0 (BUT Speech@FIT; pyannote integration by
  Jiangyu Han and Petr Pálka). Attribution required — see `diaric`'s
  `NOTICE`.

## NOT bundled — `wespeaker_resnet34_lm.onnx`

The 27 MB WeSpeaker ResNet34-LM export exceeds the crates.io 10 MB
crate-tarball limit (the float32 weights are mostly incompressible —
gzip recovers ~7 %). Callers fetch it via
`scripts/download-embed-model.sh` (or set `DIA_EMBED_MODEL_PATH`).
The expected SHA-256 lives in that script.

### Single-file vs external-data layout

The shipped form is the **single-file** ONNX (~25.5 MiB, all weights
inlined). The single-file form sidesteps a *load-time* failure on
ORT's CoreML EP — Apple's optimizer fails to relocate external
initializers when the model uses the alternative external-data
layout (a small `.onnx` header next to a large `.onnx.data` sidecar)
and aborts session creation with `model_path must not be empty`.

**This is purely about *loading*, not *running*.** Even with the
single-file form, ORT's CoreML EP **mistranslates the WeSpeaker
ResNet34-LM compute graph** and emits NaN/Inf on most realistic
inputs (verified across every `ComputeUnits` setting, both
`NeuralNetwork` and `MLProgram` model formats, and the
static-shape knob; only short clips with simple acoustic content
happen to evade it). The `EmbedModel` finite-output validator
then aborts the pipeline. **dia therefore does NOT auto-register
EPs on `EmbedModel::from_file` even when `--features coreml` is
on**; the asymmetric default that *does* auto-register on
`SegmentModel::bundled()` is documented at
[`crate::segment::SegmentModel::bundled`].

If you bring your own model in external-data form (e.g. from an
upstream pyannote or HuggingFace mirror), repack it before use via:

```python
import onnx
m = onnx.load("wespeaker_resnet34_lm.onnx", load_external_data=True)
onnx.save(m, "wespeaker_resnet34_lm.onnx", save_as_external_data=False)
```

— same f32 weights, no quantization, no graph transform; the only
change is that ORT no longer follows an external pointer. This
makes the model loadable on CoreML; **it does NOT make it correct
to run on CoreML**.

### `.pt` TorchScript variant

A separate dev-only file used by the optional `tch` feature. Out-of-tree.
