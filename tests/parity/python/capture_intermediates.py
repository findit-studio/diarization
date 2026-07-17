"""Capture pyannote/speaker-diarization-community-1 intermediate artifacts.

Outputs (under tests/parity/fixtures/<clip-stem>/):
  - raw_embeddings.npz    (num_chunks, num_slots, 256) pre-PLDA WeSpeaker
  - plda_embeddings.npz   post_xvec + post_plda (num_train, 128) + train indices
  - segmentations.npz     pyannote per-chunk per-frame speaker probs
  - ahc_init_labels.npy   (num_train,) AHC init labels
  - ahc_state.npz         threshold
  - reconstruction.npz    count + discrete_diarization (Phase 5b)
  - vbx_state.npz         qinit, q_final, sp_final, elbo_trajectory
  - clustering.npz        soft_clusters, hard_clusters, centroids
  - reference.rttm        final RTTM
  - manifest.json         sha256 + pyannote/numpy versions

Strategy:
  - hook callback for raw embeddings + final discrete diarization (public API).
  - Replace pipeline.clustering with CapturingVBxClustering subclass whose
    __call__ body mirrors pyannote 4.0.4's VBxClustering.__call__ verbatim
    with capture statements interleaved.

Usage:
  uv run python capture_intermediates.py <clip.wav>
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyannote.audio
from einops import rearrange
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.clustering import VBxClustering
from pyannote.audio.utils.vbx import VBx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from scipy.special import softmax as scipy_softmax
from sklearn.cluster import KMeans

PIPELINE_NAME = "pyannote/speaker-diarization-community-1"
# `cluster_vbx` default (pyannote.audio.utils.vbx:140); the smoothing
# factor applied to the one-hot ahc_clusters before passing as VBx
# initial responsibilities.
VBX_INIT_SMOOTHING = 7.0


@dataclass
class CaptureBuffer:
    # via hook callback
    segmentation: np.ndarray | None = None
    speaker_counting: np.ndarray | None = None
    raw_embeddings: np.ndarray | None = None
    discrete_diarization: np.ndarray | None = None
    chunk_start: float | None = None
    chunk_duration: float | None = None
    chunk_step: float | None = None
    frame_start: float | None = None
    frame_duration: float | None = None
    frame_step: float | None = None

    # via CapturingVBxClustering
    train_embeddings: np.ndarray | None = None
    train_chunk_idx: np.ndarray | None = None
    train_speaker_idx: np.ndarray | None = None
    post_xvec: np.ndarray | None = None
    post_plda: np.ndarray | None = None
    ahc_clusters: np.ndarray | None = None
    qinit: np.ndarray | None = None
    q_final: np.ndarray | None = None
    sp_final: np.ndarray | None = None
    elbo_trajectory: list[float] = field(default_factory=list)
    soft_clusters: np.ndarray | None = None
    hard_clusters: np.ndarray | None = None
    centroids: np.ndarray | None = None


class CapturingVBxClustering(VBxClustering):
    """Records every intermediate of VBxClustering.__call__ to `self._buf`.

    The body of __call__ is a verbatim copy of
    pyannote.audio.pipelines.clustering.VBxClustering.__call__ from
    pyannote.audio==4.0.4 (clustering.py:572-668), with capture
    statements interleaved. If the upstream version is bumped, this
    body must be re-synced against the new source.
    """

    def __init__(self, *args, capture_buf: CaptureBuffer, **kwargs):
        super().__init__(*args, **kwargs)
        self._buf = capture_buf

    def __call__(
        self,
        embeddings,
        segmentations=None,
        num_clusters=None,
        min_clusters=None,
        max_clusters=None,
        **kwargs,
    ):
        buf = self._buf
        constrained_assignment = self.constrained_assignment

        train_embeddings, chunk_idx, speaker_idx = self.filter_embeddings(
            embeddings, segmentations=segmentations
        )
        buf.train_embeddings = train_embeddings.copy()
        buf.train_chunk_idx = np.asarray(chunk_idx).copy()
        buf.train_speaker_idx = np.asarray(speaker_idx).copy()

        if train_embeddings.shape[0] < 2:
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            buf.hard_clusters = hard_clusters.copy()
            buf.soft_clusters = soft_clusters.copy()
            buf.centroids = centroids.copy()
            return hard_clusters, soft_clusters, centroids

        # AHC (clustering.py:597-603)
        train_embeddings_normed = train_embeddings / np.linalg.norm(
            train_embeddings, axis=1, keepdims=True
        )
        dendrogram = linkage(
            train_embeddings_normed, method="centroid", metric="euclidean"
        )
        ahc_clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)
        buf.ahc_clusters = ahc_clusters.copy()

        # PLDA — capture xvec/plda stages separately by invoking the lambdas
        # directly. self.plda(x) is _plda_tf(_xvec_tf(x), lda_dim=...).
        post_xvec = self.plda._xvec_tf(train_embeddings)
        buf.post_xvec = post_xvec.copy()
        fea = self.plda._plda_tf(post_xvec, lda_dim=self.plda.lda_dimension)
        buf.post_plda = fea.copy()

        # VBx — replicate cluster_vbx() inline so we can capture qinit and
        # the ELBO trajectory `Li` (cluster_vbx discards them).
        qinit = np.zeros((len(ahc_clusters), int(ahc_clusters.max()) + 1))
        qinit[range(len(ahc_clusters)), ahc_clusters.astype(int)] = 1.0
        qinit = scipy_softmax(qinit * VBX_INIT_SMOOTHING, axis=1)
        buf.qinit = qinit.copy()

        gamma, pi, Li, _, _ = VBx(
            fea,
            self.plda.phi,
            Fa=self.Fa,
            Fb=self.Fb,
            pi=qinit.shape[1],
            gamma=qinit,
            maxIters=20,
            return_model=True,
        )
        buf.q_final = gamma.copy()
        buf.sp_final = pi.copy()
        buf.elbo_trajectory = [float(np.asarray(li).item()) for li in Li]

        # Centroids (clustering.py:617-620)
        num_chunks, num_speakers, dimension = embeddings.shape
        W = gamma[:, pi > 1e-7]
        centroids = (
            W.T @ train_embeddings.reshape(-1, dimension)
        ) / W.sum(0, keepdims=True).T

        # KMeans branch (clustering.py:625-643)
        auto_num_clusters, _ = centroids.shape
        if min_clusters is not None and auto_num_clusters < min_clusters:
            num_clusters = min_clusters
        elif max_clusters is not None and auto_num_clusters > max_clusters:
            num_clusters = max_clusters
        if num_clusters and num_clusters != auto_num_clusters:
            constrained_assignment = False
            kmeans_clusters = KMeans(
                n_clusters=num_clusters, n_init=3, random_state=42, copy_x=False
            ).fit_predict(train_embeddings_normed)
            centroids = np.vstack(
                [
                    np.mean(train_embeddings[kmeans_clusters == k], axis=0)
                    for k in range(num_clusters)
                ]
            )

        # e2k distances (clustering.py:646-655)
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = 2 - e2k_distance

        # Constrained Hungarian (clustering.py:658-662)
        if constrained_assignment:
            const = soft_clusters.min() - 1.0
            soft_clusters[segmentations.data.sum(1) == 0] = const
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        hard_clusters = hard_clusters.reshape(num_chunks, num_speakers)
        buf.soft_clusters = soft_clusters.copy()
        buf.hard_clusters = hard_clusters.copy()
        buf.centroids = centroids.copy()
        return hard_clusters, soft_clusters, centroids


def make_hook(buf: CaptureBuffer):
    """Build the pyannote-style hook callback.

    `pipeline(file, hook=...)` calls
    `hook(name, artefact, file=..., total=..., completed=...)`. Progress
    callbacks pass `total` + `completed`; only milestone calls have
    artefact set. We record artefacts at four named milestones.
    """

    def hook(name, artifact, file=None, total=None, completed=None, **kw):
        if total is not None or completed is not None:
            return
        if name == "segmentation":
            buf.segmentation = np.asarray(artifact.data).copy()
            # Capture sliding-window timing metadata for Phase 5b
            # reconstruction port: pyannote's `Inference.aggregate`
            # uses these to map chunk indices to output-frame indices.
            sw = artifact.sliding_window
            buf.chunk_start = float(sw.start)
            buf.chunk_duration = float(sw.duration)
            buf.chunk_step = float(sw.step)
        elif name == "speaker_counting":
            buf.speaker_counting = np.asarray(artifact.data).copy()
            sw = artifact.sliding_window
            buf.frame_start = float(sw.start)
            buf.frame_duration = float(sw.duration)
            buf.frame_step = float(sw.step)
        elif name == "embeddings":
            buf.raw_embeddings = np.asarray(artifact).copy()
        elif name == "discrete_diarization":
            buf.discrete_diarization = np.asarray(artifact.data).copy()

    return hook


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: capture_intermediates.py <clip.wav>")
    clip = Path(sys.argv[1]).resolve()
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    out_dir = clip.parent
    print(f"[capture] clip: {clip}")
    print(f"[capture] out:  {out_dir}")

    pipeline = Pipeline.from_pretrained(PIPELINE_NAME)
    buf = CaptureBuffer()

    # Replace pipeline.clustering (a VBxClustering instance) with our
    # capturing subclass for one run. Restored in the finally block so the
    # pipeline object is left as-is even if the run errors.
    original_clustering = pipeline.clustering
    cap = CapturingVBxClustering(
        plda=pipeline._plda,
        metric=original_clustering.metric,
        constrained_assignment=original_clustering.constrained_assignment,
        capture_buf=buf,
    )
    cap.threshold = original_clustering.threshold
    cap.Fa = original_clustering.Fa
    cap.Fb = original_clustering.Fb
    pipeline.clustering = cap
    try:
        result = pipeline(str(clip), hook=make_hook(buf))
    finally:
        pipeline.clustering = original_clustering

    diarization = (
        result.speaker_diarization
        if hasattr(result, "speaker_diarization")
        else result
    )

    # Persist artifacts
    np.savez_compressed(
        out_dir / "raw_embeddings.npz",
        embeddings=buf.raw_embeddings,
    )
    # `segmentation` is the per-chunk per-frame per-speaker probability
    # tensor that drives both `filter_embeddings` (active-frame ratio)
    # and the constrained_assignment masking inside `cluster_vbx` (zero-
    # activity speakers get a low-cost sentinel). Captured for Phase 5a.
    np.savez_compressed(
        out_dir / "segmentations.npz",
        segmentations=buf.segmentation,
    )
    # Reconstruction stage 8 fixtures — Phase 5b. `count` is the
    # per-frame instantaneous-active-speaker count derived by
    # pyannote's `binarize+sum` over the aggregated segmentations
    # (used as top-K cap when binarizing the clustered output).
    # `discrete_diarization` is the final per-frame discrete labels.
    # min_duration_off feeds Phase 5c's Binarize port. Pyannote
    # community-1's segmentation block hardcodes this from config.yaml.
    seg_min_duration_off = float(pipeline.segmentation.min_duration_off)
    np.savez_compressed(
        out_dir / "reconstruction.npz",
        count=buf.speaker_counting,
        discrete_diarization=buf.discrete_diarization,
        # Sliding-window timing — needed by Phase 5b's overlap-add
        # aggregation port. Without these, the chunk-to-output-frame
        # mapping is implicit and would have to be reverse-engineered
        # from numpy shape alone.
        chunk_start=np.float64(buf.chunk_start),
        chunk_duration=np.float64(buf.chunk_duration),
        chunk_step=np.float64(buf.chunk_step),
        frame_start=np.float64(buf.frame_start),
        frame_duration=np.float64(buf.frame_duration),
        frame_step=np.float64(buf.frame_step),
        min_duration_off=np.float64(seg_min_duration_off),
    )
    np.savez_compressed(
        out_dir / "plda_embeddings.npz",
        post_xvec=buf.post_xvec,
        post_plda=buf.post_plda,
        train_chunk_idx=buf.train_chunk_idx,
        train_speaker_idx=buf.train_speaker_idx,
        # `phi` is the PLDA eigenvalue diagonal that VBx consumes
        # independently of the projected feature matrix. Captured
        # here so the Rust port's `phi()` can be parity-checked
        # numerically; structural (descending + length) checks
        # alone would let a regression returning raw `psi` or
        # mis-scaled eigenvalues silently break VBx posterior
        # updates downstream. Codex review MEDIUM (round 8).
        phi=pipeline._plda.phi,
    )
    np.save(out_dir / "ahc_init_labels.npy", buf.ahc_clusters)
    np.savez_compressed(
        out_dir / "ahc_state.npz",
        # `threshold` is the AHC linkage cutoff (config.yaml; community-1
        # default is 0.6). Captured alongside the labels so a future
        # config retune surfaces as a parity failure instead of silent
        # hardcoded-constant drift. (Phase 4, Task 0.)
        threshold=np.float64(cap.threshold),
    )
    np.savez_compressed(
        out_dir / "vbx_state.npz",
        qinit=buf.qinit,
        q_final=buf.q_final,
        sp_final=buf.sp_final,
        elbo_trajectory=np.array(buf.elbo_trajectory, dtype=np.float64),
        # `fa`, `fb`, `max_iters` are inputs to VBx — pinned in the
        # pipeline's config.yaml (community-1 uses Fa=0.07, Fb=0.8;
        # `cluster_vbx`'s call site at clustering.py:613 overrides
        # maxIters=20). Capturing the inputs alongside the outputs
        # (q_final, sp_final, elbo_trajectory) keeps the parity test
        # self-contained: a future model upgrade surfaces as a
        # parity failure rather than a silent hardcoded-constant
        # drift. (Phase 2 plan, Task 0.)
        fa=np.float64(cap.Fa),
        fb=np.float64(cap.Fb),
        max_iters=np.int64(20),
    )
    np.savez_compressed(
        out_dir / "clustering.npz",
        soft_clusters=buf.soft_clusters,
        hard_clusters=buf.hard_clusters,
        centroids=buf.centroids,
    )

    rttm_path = out_dir / "reference.rttm"
    with rttm_path.open("w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {clip.stem} 1 {turn.start:.3f} {turn.duration:.3f}"
                f" <NA> <NA> {speaker} <NA> <NA>\n"
            )

    artifact_files = [
        "raw_embeddings.npz",
        "segmentations.npz",
        "plda_embeddings.npz",
        "ahc_init_labels.npy",
        "ahc_state.npz",
        "vbx_state.npz",
        "clustering.npz",
        "reconstruction.npz",
        "reference.rttm",
    ]
    manifest = {
        "pyannote_audio_version": pyannote.audio.__version__,
        "numpy_version": np.__version__,
        "clip_path": str(clip),
        "clip_sha256": _file_sha256(clip),
        "artifacts": {f: _file_sha256(out_dir / f) for f in artifact_files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # PLDA weight refresh moved to `diaric` (scripts/export-plda-weights.py):
    # the `models/plda/` blobs are compiled into the `diaric` dependency, not
    # this crate, so this capture harness no longer exports them.

    # Summary
    print(f"[capture] raw_embeddings: {buf.raw_embeddings.shape}")
    print(f"[capture] post_xvec:      {buf.post_xvec.shape}")
    print(f"[capture] post_plda:      {buf.post_plda.shape}")
    ahc_unique = sorted(set(buf.ahc_clusters.tolist()))
    print(f"[capture] ahc_clusters:   {buf.ahc_clusters.shape}, unique={ahc_unique}")
    print(f"[capture] q_final:        {buf.q_final.shape}")
    print(f"[capture] sp_final:       {buf.sp_final}")
    print(f"[capture] elbo iters:     {len(buf.elbo_trajectory)}")
    hard_unique = sorted(set(buf.hard_clusters.flatten().tolist()))
    print(f"[capture] hard_clusters:  {buf.hard_clusters.shape}, unique={hard_unique}")
    print("[capture] done")


if __name__ == "__main__":
    main()
