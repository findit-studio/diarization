<div align="center">
<h1>diarization</h1>
</div>
<div align="center">

Sans-I/O speaker diarization with pyannote-equivalent accuracy.

[<img alt="github" src="https://img.shields.io/badge/github-findit--studio/diarization-8da0cb?style=for-the-badge&logo=GitHub" height="22">][GitHub-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fdiarization" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-studio/diarization/ci.yml?logo=GitHub-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-studio/diarization?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-diarization-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/diarization?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/diarization?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>

## Quick start

The segmentation model ships inside this crate, and the PLDA weights ship
inside its [`diaric`](https://crates.io/crates/diaric) dependency (a published
crate) — both embed into your binary automatically, so only the WeSpeaker
ResNet34-LM embedding ONNX is BYO (~26 MB; above the crates.io 10 MB hard
limit, so it cannot be bundled). Fetch it from the
[FinDIT-Studio/dia-models](https://huggingface.co/FinDIT-Studio/dia-models)
HuggingFace bundle. Both commands below pin a specific HF commit and
verify SHA-256 before installing — a republished or truncated upstream
model surfaces as a hard failure rather than silently altering
diarization output.

```sh
# Pinned upstream revision + expected SHA-256 of the FP32 single-file ONNX.
DIA_EMBED_MODEL_REV="6eef479c954ec180e79cee316af2f16d5f7720bd"
DIA_EMBED_MODEL_SHA256="f23f04aa9d0f6b8b0a28de016d226dcbe92d7461a6e58045401acfbed623838a"
mkdir -p models
TMP="$(mktemp "${TMPDIR:-/tmp}/wespeaker_resnet34_lm.XXXXXXXXXX")"
```

```sh
# Option A: huggingface_hub CLI (handles caching, retries, optional auth).
hf download \
  --revision "$DIA_EMBED_MODEL_REV" \
  --local-dir "$(dirname "$TMP")" \
  --local-dir-use-symlinks False \
  FinDIT-Studio/dia-models wespeaker_resnet34_lm.onnx
mv "$(dirname "$TMP")/wespeaker_resnet34_lm.onnx" "$TMP"
```

```sh
# Option B: plain curl, no extra tools.
curl --fail --location \
  --output "$TMP" \
  "https://huggingface.co/FinDIT-Studio/dia-models/resolve/${DIA_EMBED_MODEL_REV}/wespeaker_resnet34_lm.onnx"
```

```sh
# Then verify and install:
ACTUAL="$(shasum -a 256 "$TMP" | awk '{print $1}')"
if [ "$ACTUAL" != "$DIA_EMBED_MODEL_SHA256" ]; then
  echo "SHA-256 mismatch: expected $DIA_EMBED_MODEL_SHA256, got $ACTUAL" >&2
  rm -f "$TMP"; exit 1
fi
mv "$TMP" models/wespeaker_resnet34_lm.onnx
```

(Workspace developers can also run `./scripts/download-embed-model.sh`,
which wraps the same revision + SHA. The script is omitted from the
published crate tarball, so the inline commands above are the source
of truth for crates.io users.)

Then run an end-to-end example. The simplest needs only the `ort`
feature:

```sh
cargo run --release --features ort --example run_owned_pipeline -- \
  path/to/clip_16k.wav > hyp.rttm
```

For the streaming pipeline (uses the sister `silero` crate to detect
voice ranges on the fly), the same `ort` feature is enough — `silero`
itself is a dev-dependency, always available to examples:

```sh
cargo run --release --features ort --example run_streaming_pipeline -- \
  path/to/clip.wav
```

`DIA_EMBED_MODEL_PATH` overrides the default `models/wespeaker_resnet34_lm.onnx`
location if you keep the model elsewhere.

## Cargo features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `ort` | yes | The ONNX-runtime-backed `SegmentModel` and `EmbedModel` types. |
| `bundled-segmentation` | yes | Embeds `models/segmentation-3.0.onnx` (~6 MB) into the binary. Exposes `SegmentModel::bundled()`. Implies `ort`. Disable to ship a fine-tuned segmentation model separately. |
| `tch` | no | TorchScript embedding backend (libtorch ≈600 MB). Bit-exact pyannote on heavy-overlap fixtures where ONNX→ORT diverges. |

`silero` is tracked as a dev-dependency (only `examples/run_streaming_pipeline.rs`
consumes it). No feature gate — examples have access to dev-deps.

The PLDA parity suite moved to the [`diaric`](https://crates.io/crates/diaric)
dependency along with the backend-free core; it lives in `diaric`'s
`src/plda/parity_tests.rs` (API docs at [docs.rs/diaric](https://docs.rs/diaric)).
`diaric` is a published crate, but running its parity suite still needs a source
checkout: Cargo does not run a dependency's unit tests through re-exports (so the
filter matches 0 tests here), and the `.npz` fixtures those tests read are
checked into the `diaric` repo rather than shipped in the published crate tarball
(they would blow the crates.io 10 MB limit). Run it from a `diaric` checkout:

```bash
# from a checkout of https://github.com/findit-studio/diaric
cargo test plda::parity_tests
```

It auto-skips when `tests/parity/fixtures/01_dialogue/*.npz` is absent
(checked in to `diaric`, but a fresh checkout from a model-only mirror
would have to regenerate them via the Phase-0 capture script).

## License

`diarization` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.
Bundled third-party model attributions and source licenses are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright (c) 2026 FinDIT studio authors.

### Bundled-model attributions propagate to downstream binaries

`diarization` embeds one third-party model artifact of its own into every
compiled binary via `include_bytes!`:

| File | License | Source |
|---|---|---|
| `models/segmentation-3.0.onnx` (bundled when `bundled-segmentation` feature is on, default) | **MIT** | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |

This crate's own SPDX expression is therefore
`(MIT OR Apache-2.0) AND MIT`. It also depends on
[`diaric`](https://github.com/findit-studio/diaric), which embeds the
CC-BY-4.0 PLDA weights — and vendors the SciPy / torchaudio / FluidAudio
source ports — into any linking binary; those obligations are declared by
`diaric`'s own SPDX expression and `NOTICE`.

When you redistribute a binary that depends on `diarization`, reproduce the
attributions from this crate's
[NOTICE](https://github.com/findit-studio/diarization/blob/main/NOTICE) **and**
from [`diaric`'s NOTICE](https://github.com/findit-studio/diaric/blob/main/NOTICE)
somewhere a recipient can find — for instance, in your application's
"About" or third-party-licenses page. Full provenance:
[models/SOURCE.md](https://github.com/findit-studio/diarization/blob/main/models/SOURCE.md)
(segmentation) and diaric's
[models/plda/SOURCE.md](https://github.com/findit-studio/diaric/blob/main/models/plda/SOURCE.md)
(PLDA).

To opt out of the segmentation bundling (e.g. to ship a fine-tuned
variant), disable default features: `diarization = { version = "...",
default-features = false, features = ["ort"] }`. You then load via
`SegmentModel::from_file` / `from_memory` directly.

[GitHub-url]: https://github.com/findit-studio/diarization
[CI-url]: https://github.com/findit-studio/diarization/actions/workflows/ci.yml
[codecov-url]: https://app.codecov.io/gh/findit-studio/diarization/
[doc-url]: https://docs.rs/diarization
[crates-url]: https://crates.io/crates/diarization
