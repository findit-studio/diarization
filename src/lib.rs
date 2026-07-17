#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

// The backend-free diarization core — clustering, PLDA, pipeline
// assembly, reconstruction/RTTM, frame aggregation, provenance, and the
// spill-backed buffer types — lives in the `diaric` crate. `diarization`
// is the ONNX/Torch pyannote port: it layers the segmentation + embedding
// model runners (`segment`, `embed`), the execution-provider surface
// (`ep`), and the streaming service (`streaming`) on top, and re-exports
// the `diaric` core so existing `diarization::{cluster, plda, …}` paths
// keep resolving unchanged. The public `spill` module (spill-buffer
// configuration reachable from `OwnedPipelineOptions::with_spill_options`
// / `StreamingOfflineOptions::with_spill_options`) is `diaric::spill`.
pub use diaric::{aggregate, cluster, pipeline, plda, provenance, reconstruct, spill};

pub mod embed;
pub mod segment;

/// Opt-in ONNX Runtime execution providers (CoreML, CUDA, TensorRT,
/// DirectML, ROCm, OpenVINO, WebGPU, …) for hardware-accelerated
/// segmentation + embedding inference. See [`crate::ep`] for the full
/// list, the per-EP cargo features that toggle each one, and an
/// `auto_providers()` helper that picks the right EP at runtime.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub mod ep;

#[cfg(all(feature = "ort", feature = "serde"))]
mod ort_serde;

pub mod offline;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub mod streaming;
