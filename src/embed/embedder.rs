//! Sliding-window mean aggregation for variable-length clips.
//! Spec §5.1 (unweighted) / §5.2 (voice-probability-weighted).
//!
//! These helpers are the bridge between the raw `EmbedModel::embed_features`
//! API (single fixed-length window) and the public `embed{,_weighted,_masked}`
//! methods on `EmbedModel` (variable-length clips). They are `pub(crate)`
//! because the public surface lives on `EmbedModel` itself.

use diaric::{axpy_f32, embed::Error as EmbedCore};

use crate::embed::{
  EmbedModel, Error,
  options::{EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, HOP_SAMPLES, MIN_CLIP_SAMPLES, NORM_EPSILON},
};

/// Plan window starts for a clip of `len` samples (spec §5.1).
///
/// Algorithm:
/// - `len <= EMBED_WINDOW_SAMPLES`: single window at start `0`. Caller is
///   expected to zero-pad the clip up to `EMBED_WINDOW_SAMPLES` before
///   passing to `compute_fbank`.
/// - `len > EMBED_WINDOW_SAMPLES`: regular grid `[0, HOP, 2*HOP, …, k_max*HOP]`
///   with `k_max = (len - WINDOW) / HOP`, plus a tail anchor at
///   `len - WINDOW` (so the last window ends exactly at `len`).
///   The result is sorted + deduped — when the regular grid ends at
///   `len - WINDOW` (multiples align), the tail is collapsed.
///
/// Caller invariant: `len >= MIN_CLIP_SAMPLES` (verified by `embed_unweighted`).
pub(crate) fn plan_starts(len: usize) -> Vec<usize> {
  if len <= EMBED_WINDOW_SAMPLES as usize {
    return vec![0];
  }
  let win = EMBED_WINDOW_SAMPLES as usize;
  let hop = HOP_SAMPLES as usize;
  let k_max = (len - win) / hop;
  let mut starts: Vec<usize> = (0..=k_max).map(|k| k * hop).collect();
  starts.push(len - win);
  starts.sort_unstable();
  starts.dedup();
  starts
}

/// Run inference on one full clip via the unweighted sliding-window-mean
/// algorithm (spec §5.1).
///
/// - `len < MIN_CLIP_SAMPLES`: returns [`EmbedCore::InvalidClip`].
/// - `MIN_CLIP_SAMPLES <= len <= EMBED_WINDOW_SAMPLES`: single inference
///   on the zero-padded clip, returns `(raw, 1)`.
/// - `len > EMBED_WINDOW_SAMPLES`: sums per-window raw outputs across the
///   sliding-window plan, returns `(sum, num_windows)`.
///
/// Returns the **unnormalized** sum. Caller L2-normalizes via
/// [`Embedding::normalize_from`](crate::embed::Embedding::normalize_from)
/// (which surfaces [`EmbedCore::DegenerateEmbedding`] on zero-norm).
pub(crate) fn embed_unweighted(
  model: &mut EmbedModel,
  samples: &[f32],
) -> Result<([f32; EMBEDDING_DIM], u32), Error> {
  if samples.len() < MIN_CLIP_SAMPLES as usize {
    return Err(
      EmbedCore::InvalidClip {
        len: samples.len(),
        min: MIN_CLIP_SAMPLES as usize,
      }
      .into(),
    );
  }
  // Backend-independent finite-input guard. ORT routes through
  // `compute_full_fbank` which rejects non-finite samples upfront,
  // but the tch backend feeds `samples` directly into a TorchScript
  // `Tensor::from_slice` and may return a corrupted-but-finite
  // embedding that passes the post-output check. Mirrors the guard
  // already in `EmbedModel::embed_chunk_with_frame_mask`.
  if samples.iter().any(|v| !v.is_finite()) {
    return Err(EmbedCore::NonFiniteInput.into());
  }
  let mut sum = [0.0f32; EMBEDDING_DIM];

  if samples.len() <= EMBED_WINDOW_SAMPLES as usize {
    // Zero-pad to EMBED_WINDOW_SAMPLES (kaldi-fbank's frame budget).
    let mut padded = vec![0.0f32; EMBED_WINDOW_SAMPLES as usize];
    padded[..samples.len()].copy_from_slice(samples);
    let raw = model.embed_audio_clip(&padded)?;
    return Ok((raw, 1));
  }

  let starts = plan_starts(samples.len());
  let win = EMBED_WINDOW_SAMPLES as usize;
  let clips: Vec<&[f32]> = starts.iter().map(|&s| &samples[s..s + win]).collect();
  let raws = model.embed_audio_clips_batch(&clips)?;
  // SIMD-routable per-window aggregation. `ops::axpy_f32` with
  // `alpha = 1.0` is `y += x`; the f32 mul_add loop autovectorizes
  // to NEON `vfmaq_f32` / AVX2 `_mm256_fmadd_ps` over 256-element
  // strides. Using mul_add (vs scalar `+=`) shifts the rounding
  // boundary by at most 1 ULP per element relative to a literal
  // `*s += r` chain, which doesn't propagate visibly through
  // L2-normalize / cosine clustering.
  for raw in &raws {
    axpy_f32(&mut sum, 1.0, raw.as_slice());
  }
  Ok((sum, starts.len() as u32))
}

/// Sliding-window mean WEIGHTED by per-sample voice probabilities (spec §5.2).
///
/// Same window plan as [`embed_unweighted`]. Per-window weight = mean of
/// `voice_probs` over that window's samples. The returned sum is the
/// per-window weighted sum; caller divides by `total_weight` (or, more
/// simply, L2-normalizes — for a unit-vector output the normalization
/// step is equivalent).
///
/// Errors:
/// - [`EmbedCore::WeightShapeMismatch`] if `voice_probs.len() != samples.len()`.
/// - [`EmbedCore::InvalidClip`] if `samples.len() < MIN_CLIP_SAMPLES`.
/// - [`EmbedCore::AllSilent`] if the sum of per-window weights is below
///   [`NORM_EPSILON`] (no signal to aggregate).
///
/// Returns `(weighted_sum, num_windows, total_weight)`.
pub(crate) fn embed_weighted_inner(
  model: &mut EmbedModel,
  samples: &[f32],
  voice_probs: &[f32],
) -> Result<([f32; EMBEDDING_DIM], u32, f32), Error> {
  if samples.len() != voice_probs.len() {
    return Err(
      EmbedCore::WeightShapeMismatch {
        samples_len: samples.len(),
        weights_len: voice_probs.len(),
      }
      .into(),
    );
  }
  if samples.len() < MIN_CLIP_SAMPLES as usize {
    return Err(
      EmbedCore::InvalidClip {
        len: samples.len(),
        min: MIN_CLIP_SAMPLES as usize,
      }
      .into(),
    );
  }
  // Backend-independent finite-input guard on samples (mirrors
  // `embed_unweighted`). tch backend forwards samples directly to
  // TorchScript without an upstream finite check.
  if samples.iter().any(|v| !v.is_finite()) {
    return Err(EmbedCore::NonFiniteInput.into());
  }
  // Voice-probability weights must be finite AND in [0, 1]. NaN
  // weights bypass the `total_weight < NORM_EPSILON` check (every
  // comparison with NaN is false) and propagate into the per-window
  // mul_add, poisoning the aggregated sum. Out-of-range finite
  // weights (negative, > 1) produce a signed mixture that no longer
  // represents a probability-weighted mean — the caller's contract
  // is "voice probabilities", not arbitrary weights.
  if voice_probs
    .iter()
    .any(|w| !w.is_finite() || *w < 0.0 || *w > 1.0)
  {
    return Err(EmbedCore::InvalidVoiceProbs.into());
  }

  let mut sum = [0.0f32; EMBEDDING_DIM];
  let win = EMBED_WINDOW_SAMPLES as usize;

  if samples.len() <= win {
    // Zero-pad path. Weight = mean of voice_probs over the (un-padded) range.
    let mut padded = vec![0.0f32; win];
    padded[..samples.len()].copy_from_slice(samples);
    let raw = model.embed_audio_clip(&padded)?;
    let w: f32 = voice_probs.iter().sum::<f32>() / voice_probs.len() as f32;
    if w < NORM_EPSILON {
      return Err(EmbedCore::AllSilent.into());
    }
    axpy_f32(&mut sum, w, raw.as_slice());
    return Ok((sum, 1, w));
  }

  let starts = plan_starts(samples.len());
  let clips: Vec<&[f32]> = starts.iter().map(|&s| &samples[s..s + win]).collect();
  let raws = model.embed_audio_clips_batch(&clips)?;
  let mut total_weight = 0.0f32;
  for (i, &start) in starts.iter().enumerate() {
    let weights = &voice_probs[start..start + win];
    let w: f32 = weights.iter().sum::<f32>() / win as f32;
    axpy_f32(&mut sum, w, raws[i].as_slice());
    total_weight += w;
  }
  if total_weight < NORM_EPSILON {
    return Err(EmbedCore::AllSilent.into());
  }
  Ok((sum, starts.len() as u32, total_weight))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn plan_starts_for_2s_clip() {
    // EMBED_WINDOW_SAMPLES = 32_000. Single-window path → [0].
    let starts = plan_starts(EMBED_WINDOW_SAMPLES as usize);
    assert_eq!(starts, vec![0]);
  }

  #[test]
  fn plan_starts_for_3s_clip() {
    // 48_000 samples; win = 32_000, hop = 16_000.
    // k_max = (48_000 - 32_000) / 16_000 = 1.
    // Regular grid: [0, 16_000]. Tail anchor: 48_000 - 32_000 = 16_000.
    // After dedup: [0, 16_000].
    let starts = plan_starts(48_000);
    assert_eq!(starts, vec![0, 16_000]);
  }

  #[test]
  fn plan_starts_for_3_5s_clip() {
    // 56_000 samples. k_max = (56_000 - 32_000) / 16_000 = 1.
    // Regular: [0, 16_000]. Tail: 56_000 - 32_000 = 24_000.
    // Dedup → [0, 16_000, 24_000] (3 distinct windows; tail not aligned).
    let starts = plan_starts(56_000);
    assert_eq!(starts, vec![0, 16_000, 24_000]);
  }

  #[test]
  fn plan_starts_for_4s_clip() {
    // 64_000 samples. k_max = (64_000 - 32_000) / 16_000 = 2.
    // Regular: [0, 16_000, 32_000]. Tail: 32_000. Dedup → [0, 16_000, 32_000].
    let starts = plan_starts(64_000);
    assert_eq!(starts, vec![0, 16_000, 32_000]);
  }

  #[test]
  fn plan_starts_skips_dedup_when_tail_misaligned() {
    // 50_000 samples. k_max = (50_000 - 32_000) / 16_000 = 1.
    // Regular: [0, 16_000]. Tail: 50_000 - 32_000 = 18_000.
    // After sort/dedup: [0, 16_000, 18_000].
    let starts = plan_starts(50_000);
    assert_eq!(starts, vec![0, 16_000, 18_000]);
  }

  #[test]
  fn plan_starts_for_min_clip_returns_single() {
    // Below window length → single window at 0.
    let starts = plan_starts(MIN_CLIP_SAMPLES as usize);
    assert_eq!(starts, vec![0]);
  }
}
