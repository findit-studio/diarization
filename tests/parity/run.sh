#!/usr/bin/env bash
# Pyannote parity harness.
#
# Requires:
# - models/segmentation-3.0.onnx and models/wespeaker_resnet34_lm.onnx
# - models/plda/xvec_transform.npz and models/plda/plda.npz
# - uv (https://docs.astral.sh/uv/)
# - the clip path; defaults to the canonical 2-speaker fixture
#
# Behavior:
# - If <fixture-dir>/manifest.json is missing, runs intermediate capture first.
# - Then runs dia and pyannote, computes DER.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/../.."

# Audio fixtures live in the sister `audio-fixtures` repo
# (https://github.com/Findit-AI/audio-fixtures). Pass an explicit path,
# or set `DIA_AUDIO_FIXTURES` to point at a checkout — defaults to
# `<dia>/../audio-fixtures` (sibling layout).
AUDIO_FIXTURES="${DIA_AUDIO_FIXTURES:-$ROOT/../audio-fixtures}"
DEFAULT_CLIP="$AUDIO_FIXTURES/pcm_f32le/01_dialogue.wav"
CLIP="${1:-$DEFAULT_CLIP}"
if [ ! -f "$CLIP" ]; then
  if [ "$CLIP" = "$DEFAULT_CLIP" ]; then
    echo "[run.sh] error: default fixture clip not found at:" >&2
    echo "          $DEFAULT_CLIP" >&2
    echo "        Audio fixtures live in the sister 'audio-fixtures' repo." >&2
    echo "        Either:" >&2
    echo "          - check it out as a sibling of dia:" >&2
    echo "            git clone https://github.com/Findit-AI/audio-fixtures.git $ROOT/../audio-fixtures" >&2
    echo "          - or set DIA_AUDIO_FIXTURES to its location" >&2
    echo "          - or pass an explicit clip:" >&2
    echo "            ./tests/parity/run.sh path/to/clip_16k.wav" >&2
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
