#!/usr/bin/env bash
# Pyannote parity harness.
#
# Requires:
# - models/segmentation-3.0.onnx and models/wespeaker_resnet34_lm.onnx
# - uv (https://docs.astral.sh/uv/)
# - the clip path; defaults to the canonical 2-speaker fixture
#
# Behavior:
# - If <fixture-dir>/manifest.json is missing, runs intermediate capture first.
# - Then runs dia and pyannote, computes DER.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/../.."

DEFAULT_CLIP="$SCRIPT_DIR/fixtures/01_dialogue/clip_16k.wav"
CLIP="${1:-$DEFAULT_CLIP}"
# `clip_16k.wav` files under `fixtures/*/` are gitignored (the upstream
# reference clips are sourced separately and not tracked). On a clean
# checkout the default path will not exist; surface a helpful error
# instead of letting `realpath` fail under `set -e` with no context.
if [ ! -f "$CLIP" ]; then
  if [ "$CLIP" = "$DEFAULT_CLIP" ]; then
    echo "[run.sh] error: default fixture clip not found at:" >&2
    echo "          $DEFAULT_CLIP" >&2
    echo "        That path is gitignored on purpose (upstream-sourced" >&2
    echo "        audio). Either:" >&2
    echo "          - pass an explicit clip:  ./tests/parity/run.sh path/to/clip_16k.wav" >&2
    echo "          - or provision the fixture by running" >&2
    echo "            tests/parity/python/capture_intermediates.py" >&2
    echo "            against your own 16 kHz mono WAV." >&2
  else
    echo "[run.sh] error: clip not found: $CLIP" >&2
  fi
  exit 1
fi
ABS_CLIP="$(cd "$ROOT" && realpath "$CLIP")"
SNAPSHOT_DIR="$(dirname "$ABS_CLIP")"
MANIFEST="$SNAPSHOT_DIR/manifest.json"

cd "$SCRIPT_DIR/python"
if [ ! -d .venv ]; then
  uv venv
fi
uv pip install -e . > /dev/null

if [ ! -f "$MANIFEST" ]; then
  echo "[run.sh] no manifest at $MANIFEST; running capture..."
  uv run python capture_intermediates.py "$ABS_CLIP"
else
  echo "[run.sh] reusing existing snapshot at $SNAPSHOT_DIR"
fi

# Reuse the captured RTTM as the reference (no need to rerun pyannote).
REF_RTTM="$SNAPSHOT_DIR/reference.rttm"

cd "$ROOT"
cargo run --release --manifest-path tests/parity/Cargo.toml -- "$CLIP" \
  > "$SCRIPT_DIR/hyp.rttm"

cd "$SCRIPT_DIR/python"
uv run python score.py "$REF_RTTM" "$SCRIPT_DIR/hyp.rttm"
