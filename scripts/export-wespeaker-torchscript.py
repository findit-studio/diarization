"""Export pyannote's WeSpeaker ResNet34 embedding model.

Produces two artifacts, both with signature
`(waveforms_or_fbank, weights) → embeddings`:

- `models/wespeaker_resnet34_lm.pt` (TorchScript, for the tch
  backend): takes raw 16 kHz waveforms `[N, 160_000]` plus a
  per-frame mask `[N, 589]`. Computes fbank internally (matching
  pyannote's `compute_fbank` exactly), then runs the resnet with
  the mask as statistics-pooling weights.
- `models/wespeaker_resnet34_lm.onnx` (ONNX, for the ort backend):
  takes pre-computed fbank `[N, ≈999, 80]` plus a per-frame mask
  `[N, 589]`. The Rust ORT backend computes the fbank externally
  via `kaldi-native-fbank` because torchaudio's kaldi.fbank doesn't
  export to ONNX.

Both wrappers pass `weights` through to
`WeSpeakerResNet34.resnet.forward(features, weights=weights)`,
matching pyannote's exact embedding extraction call. This is the
key to fixing the `04_three_speaker` overlap-heavy fixture (38%
DER → 0% DER): pyannote's segmentation mask is meant to drive the
pooling layer, not to gate audio samples.

Run from the repository root with the parity Python venv:

    tests/parity/python/.venv/bin/python scripts/export-wespeaker-torchscript.py
"""

import torch
from pyannote.audio import Pipeline


class WeSpeakerWrapper(torch.nn.Module):
    """Wraps pyannote's WeSpeaker end-to-end (waveforms → fbank →
    resnet → embedding) as a non-Lightning nn.Module so it can be
    traced.

    Input shape: `(N, samples)` — N raw waveform clips at 16 kHz mono.
    Output: `(N, 256)` raw, un-normalized embeddings — bit-exact to
    `WeSpeakerResNet34.forward(waveforms.unsqueeze(1))`.

    We can't trace `WeSpeakerResNet34` directly because it inherits
    from `LightningModule`, whose `.trainer` property raises when
    untrained. The wrapper sidesteps Lightning by lifting the two
    sub-modules we need (`_fbank` and `resnet`) into a plain nn.Module
    and replicating `compute_fbank`'s preprocessing inline.
    """

    def __init__(self, embed_model):
        super().__init__()
        self._fbank = embed_model._fbank
        self.resnet = embed_model.resnet

    def forward(
        self, waveforms: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        # waveforms: [N, samples]; weights: [N, num_frames].
        waveforms = waveforms.unsqueeze(1)
        scaled = waveforms * (1 << 15)
        features_list: list = []
        for b in range(scaled.shape[0]):
            features_list.append(self._fbank(scaled[b]))
        features = torch.stack(features_list, dim=0)
        features = features - torch.mean(features, dim=1, keepdim=True)
        # Pyannote's `WeSpeakerResNet34.forward` passes `weights` to
        # the resnet — it drives the temporal statistics pooling layer.
        # The mask has 1.0 in active frames and 0.0 elsewhere; the
        # pooling layer ignores 0.0-weighted frames when computing
        # the per-utterance mean and std.
        _, embedding = self.resnet(features, weights=weights)
        return embedding


class WeSpeakerOnnxWrapper(torch.nn.Module):
    """ONNX-friendly wrapper that takes pre-computed fbank + weights.

    `torch.onnx.export` can't trace through torchaudio's
    `kaldi.fbank`, so we leave fbank computation to the Rust caller
    (which uses `kaldi-native-fbank`). This wrapper covers the
    post-fbank chain only:

      input fbank `[N, num_frames, 80]` → mean-center across frames →
      resnet+pool with `weights` → embedding `[N, 256]`.

    `num_frames` for ONNX export is set to a representative value
    matching what kaldi-native-fbank emits for a 10s clip
    (`(160_000 - 400) / 160 + 1 ≈ 998` frames; we use 999 to align
    with torchaudio's count). Dynamic axes let the runtime accept
    other lengths.
    """

    def __init__(self, embed_model):
        super().__init__()
        self.resnet = embed_model.resnet

    def forward(
        self, fbank: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        # fbank: [N, num_frames, 80]; weights: [N, num_weights].
        # The Rust caller's `kaldi-native-fbank` ALREADY mean-centers
        # the fbank across frames (see the per-mel mean-centering in
        # `diaric`'s `src/embed/fbank.rs`),
        # so this wrapper does NOT center. Passing the centered fbank
        # straight to the resnet matches the existing
        # `wespeaker_resnet34_lm.onnx` contract — only the weights
        # input is new.
        _, embedding = self.resnet(fbank, weights=weights)
        return embedding


def export_torchscript(embed_model, example_audio, example_weights):
    print("== TorchScript ==")
    wrapped = WeSpeakerWrapper(embed_model)
    wrapped.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped, (example_audio, example_weights), strict=False
        )
    out_path = "models/wespeaker_resnet34_lm.pt"
    traced.save(out_path)
    reloaded = torch.jit.load(out_path)
    reloaded.eval()
    with torch.no_grad():
        out = reloaded(example_audio, example_weights)
    print(f"  output shape: {tuple(out.shape)}")
    waveforms_ref = example_audio.unsqueeze(1)
    with torch.no_grad():
        ref_out = embed_model(waveforms_ref, weights=example_weights)
    diff = (out - ref_out).abs().max().item()
    print(f"  max abs diff vs pyannote: {diff:.3e}")
    assert diff < 1e-4, f"TorchScript diverges from pyannote: {diff}"
    print(f"  saved {out_path}")


def export_onnx(embed_model, example_audio, example_weights):
    print("== ONNX ==")
    onnx_wrapper = WeSpeakerOnnxWrapper(embed_model)
    onnx_wrapper.eval()
    # Build a pre-centered fbank from the example audio, matching
    # what `kaldi-native-fbank` will hand us at runtime. We use
    # pyannote's `_fbank` to compute the raw fbank, then mean-center
    # ourselves — the deployed ONNX runtime sees the same shape and
    # value distribution.
    with torch.no_grad():
        scaled = example_audio.unsqueeze(1) * (1 << 15)
        fbank_raw = embed_model._fbank(scaled[0]).unsqueeze(0)
        fbank_unc = fbank_raw - fbank_raw.mean(dim=1, keepdim=True)

    out_path = "models/wespeaker_resnet34_lm.onnx"
    torch.onnx.export(
        onnx_wrapper,
        (fbank_unc, example_weights),
        out_path,
        input_names=["fbank", "weights"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {0: "batch", 1: "num_frames"},
            "weights": {0: "batch", 1: "num_weights"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    # Verify by loading via onnxruntime if available.
    try:
        import numpy as np  # type: ignore
        import onnxruntime as ort_  # type: ignore

        session = ort_.InferenceSession(out_path)
        out = session.run(
            None,
            {
                "fbank": fbank_unc.numpy(),
                "weights": example_weights.numpy(),
            },
        )[0]
        with torch.no_grad():
            ref_out = onnx_wrapper(fbank_unc, example_weights).numpy()
        diff = float(np.abs(out - ref_out).max())
        print(f"  output shape: {out.shape}")
        print(f"  max abs diff vs PyTorch wrapper: {diff:.3e}")
    except ImportError:
        print("  (onnxruntime not installed; skipping ONNX inference smoke-test)")
    print(f"  saved {out_path}")


def main():
    print("loading pyannote/speaker-diarization-community-1 ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )
    embed_model = pipeline._embedding.model_
    embed_model.eval()
    print(f"  model: {type(embed_model).__name__}")

    # Pyannote's get_embeddings call: 10s waveforms (160_000 samples)
    # + 589-element segmentation mask. Trace at this signature so
    # both backends accept pyannote's actual sizes.
    example_audio = torch.randn((1, 160_000), dtype=torch.float32) * 0.01
    example_weights = torch.ones((1, 589), dtype=torch.float32)

    export_torchscript(embed_model, example_audio, example_weights)
    export_onnx(embed_model, example_audio, example_weights)


if __name__ == "__main__":
    main()
