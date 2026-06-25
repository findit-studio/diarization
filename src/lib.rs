#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

pub mod cluster;
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

#[cfg(test)]
pub(crate) mod test_util;

// Numerical primitives shared across the algorithm modules. Three-tier
// backend layout (scalar/arch/dispatch) modeled on the colconv crate.
// Crate-private — algorithm modules call into `ops::*`; downstream
// callers don't see this layer. `_bench` flips it to `pub` so external
// benches in `benches/ops.rs` can A/B scalar vs SIMD on the primitives
// directly.
#[cfg_attr(feature = "_bench", doc(hidden))]
#[cfg(feature = "_bench")]
pub mod ops;
#[cfg(not(feature = "_bench"))]
pub(crate) mod ops;

/// Spill-buffer configuration types reachable from public API
/// surfaces (e.g. `OwnedPipelineOptions::with_spill_options`,
/// `StreamingOfflineOptions::with_spill_options`).
///
/// The implementation lives in the crate-private `ops::spill`
/// module; this module is the public re-export so downstream
/// callers can name and construct the types they need.
///
/// Production deployments where `/tmp` is `tmpfs` (Docker default)
/// **must** override [`SpillOptions::with_spill_dir`](crate::spill::SpillOptions::with_spill_dir)
/// to a real-disk path — without it, "spill to disk" reduces to
/// "spill to RAM" and
/// the OOM concern that motivates this whole subsystem is
/// unaddressed. That override is only possible because these types
/// are exposed here.
pub mod spill {
  pub use crate::ops::spill::{SpillBytes, SpillBytesMut, SpillError, SpillOptions};
}

pub mod plda;

pub mod provenance;

pub mod pipeline;

pub mod reconstruct;

pub mod aggregate;

pub mod offline;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub mod streaming;
