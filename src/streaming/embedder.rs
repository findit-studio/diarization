//! `StreamingEmbedder`: the public memory-fused segment+embed node.

use crate::{
  embed::EmbedModel,
  segment::SegmentModel,
  streaming::{RangeEmbeddings, StreamingError, StreamingOfflineOptions},
};

/// The public memory-fused `segment+embed` node.
///
/// Drives pyannote segmentation + WeSpeaker embedding on one VAD voice
/// range at a time and emits a [`RangeEmbeddings`] carrier per range —
/// the unit a downstream `cluster` node consumes. Audio is never
/// retained across [`push`](Self::push) calls; each call drops its PCM
/// after producing the carrier, so memory is bounded by one range plus
/// the small derived tensors the caller chooses to keep.
///
/// This is the public counterpart to
/// [`StreamingOfflineDiarizer::push_voice_range`](crate::streaming::StreamingOfflineDiarizer::push_voice_range):
/// identical fused segmentation+embedding work, but the raw output
/// crosses the API boundary instead of accumulating internally. Use this
/// when the clustering stage runs as a separate service (see
/// [`crate::streaming::cluster_ranges`]); use
/// [`StreamingOfflineDiarizer`](crate::streaming::StreamingOfflineDiarizer)
/// when one object owns the whole pipeline.
///
/// Both model types are `!Sync` (single ORT session per thread), so the
/// caller owns them and passes `&mut` references per push.
pub struct StreamingEmbedder {
  options: StreamingOfflineOptions,
}

impl StreamingEmbedder {
  /// Construct with the given options (carry `community-1` defaults by
  /// passing [`StreamingOfflineOptions::new`]).
  pub fn new(options: StreamingOfflineOptions) -> Self {
    Self { options }
  }

  /// Borrow the options.
  pub fn options(&self) -> &StreamingOfflineOptions {
    &self.options
  }

  /// Run fused segmentation + masked embedding on one voice range and
  /// return its [`RangeEmbeddings`] carrier (raw, unnormalized WeSpeaker
  /// vectors + segmentation activity + count + timing).
  ///
  /// `abs_start_sample` is the absolute sample index where this range
  /// begins in the original stream; it is recorded on the carrier so
  /// the cluster stage can re-anchor output spans.
  ///
  /// # Errors
  /// [`StreamingError::Shape`] for empty/misconfigured ranges,
  /// [`StreamingError::Segment`] / [`StreamingError::Embed`] for ONNX
  /// inference failures — same contract as
  /// [`StreamingOfflineDiarizer::push_voice_range`](crate::streaming::StreamingOfflineDiarizer::push_voice_range).
  pub fn push(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    abs_start_sample: u64,
    samples: &[f32],
  ) -> Result<RangeEmbeddings, StreamingError> {
    crate::streaming::offline_diarizer::build_range(
      &self.options,
      seg_model,
      embed_model,
      abs_start_sample,
      samples,
    )
  }

  /// End-of-stream marker. The embedder holds no cross-range state
  /// (each [`push`](Self::push) is self-contained), so this is a no-op
  /// provided for caller symmetry with stateful streaming APIs.
  pub const fn finish(&self) {}
}

#[cfg(all(test, feature = "ort", feature = "bundled-segmentation"))]
mod tests {
  use super::*;
  use crate::embed::EMBEDDING_DIM;
  use std::path::{Path, PathBuf};

  fn embed_model_or_skip() -> Option<EmbedModel> {
    let p = std::env::var_os("DIA_EMBED_MODEL_PATH")
      .map(PathBuf::from)
      .unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx")
      });
    if !p.exists() {
      eprintln!(
        "[skip] WeSpeaker model not at {}; set DIA_EMBED_MODEL_PATH",
        p.display()
      );
      return None;
    }
    Some(EmbedModel::from_file(&p).expect("load embed"))
  }

  fn load_wav_16k_mono(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.channels, 1);
    match (spec.sample_format, spec.bits_per_sample) {
      (hound::SampleFormat::Int, 16) => reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect(),
      (hound::SampleFormat::Float, 32) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
      (fmt, bps) => panic!("unsupported wav: {fmt:?} {bps}-bit"),
    }
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn push_emits_raw_not_normalized_embeddings() {
    let Some(mut emb) = embed_model_or_skip() else {
      return;
    };
    // Real conversational speech reliably drives segmentation activity
    // (a synthetic tone does not, leaving every slot inactive). Use the
    // first ~30 s of the 01_dialogue fixture — enough chunks to embed at
    // least one active speaker slot.
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("tests/parity/fixtures/01_dialogue/clip_16k.wav");
    if !wav.exists() {
      eprintln!("[skip] {} not present", wav.display());
      return;
    }
    let full = load_wav_16k_mono(&wav);
    let take = full.len().min(30 * 16_000);
    let samples = &full[..take];

    let mut seg = SegmentModel::bundled().expect("seg");
    let mut embedder = StreamingEmbedder::new(StreamingOfflineOptions::new());
    let range = embedder
      .push(&mut seg, &mut emb, 0, samples)
      .expect("push ok");
    assert!(range.num_chunks() >= 1);
    assert_eq!(
      range.raw_embeddings().len(),
      range.num_chunks() * 3 * EMBEDDING_DIM
    );
    // The invariant: at least one slot's raw embedding has a norm
    // materially different from 1.0 (a normalized vector would be ~1.0).
    let any_unnormalized = range
      .raw_embeddings()
      .chunks_exact(EMBEDDING_DIM)
      .any(|slot| {
        let norm_sq: f64 = slot.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let norm = norm_sq.sqrt();
        norm > 1e-6 && (norm - 1.0).abs() > 0.05
      });
    assert!(
      any_unnormalized,
      "StreamingEmbedder must emit RAW (unnormalized) WeSpeaker output"
    );
    embedder.finish();
  }
}
