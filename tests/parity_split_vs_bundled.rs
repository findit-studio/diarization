//! R4 parity gate: the decomposed `StreamingEmbedder` + `cluster_ranges`
//! path must reproduce the bundled `StreamingOfflineDiarizer` output
//! bit-for-bit on the same audio. Both share `build_range` +
//! `cluster_ranges_inner` and the offline clustering path has no RNG, so
//! equality is exact. Splitting segment+embed from cluster must NOT change
//! diarization results.
//!
//! `#[ignore]`-gated: loads the bundled segmentation model + the BYO
//! WeSpeaker ONNX. Run:
//! ```
//! DIA_EMBED_MODEL_PATH=models/wespeaker_resnet34_lm.onnx \
//!   cargo test --release --test parity_split_vs_bundled \
//!   --features ort,bundled-segmentation -- --ignored --nocapture
//! ```

#![cfg(all(feature = "ort", feature = "bundled-segmentation"))]

use diarization::{
  embed::EmbedModel,
  plda::PldaTransform,
  segment::SegmentModel,
  streaming::{
    StreamingEmbedder, StreamingOfflineDiarizer, StreamingOfflineOptions, cluster_ranges,
  },
};
use std::path::PathBuf;

fn load_wav(path: &PathBuf) -> Vec<f32> {
  let mut reader = hound::WavReader::open(path).expect("open wav");
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 16_000);
  assert_eq!(spec.channels, 1);
  match (spec.sample_format, spec.bits_per_sample) {
    (hound::SampleFormat::Int, 16) => reader
      .samples::<i16>()
      .map(|s| s.expect("read") as f32 / i16::MAX as f32)
      .collect(),
    (hound::SampleFormat::Float, 32) => reader.samples::<f32>().map(|s| s.expect("read")).collect(),
    _ => panic!("unsupported wav"),
  }
}

fn embed_path() -> PathBuf {
  std::env::var_os("DIA_EMBED_MODEL_PATH")
    .map(PathBuf::from)
    .unwrap_or_else(|| {
      PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx")
    })
}

/// Carve the clip into VAD-like ranges on fixed boundaries (the parity
/// invariant must hold for ANY identical range partition fed to both
/// paths — we use a deterministic partition so both sides see the same
/// ranges). Each range is one >=10 s window so segmentation has a chunk.
fn ranges_of(samples: &[f32]) -> Vec<(u64, Vec<f32>)> {
  const WIN: usize = 160_000; // 10 s
  let mut out = Vec::new();
  let mut start = 0usize;
  while start < samples.len() {
    let end = (start + WIN).min(samples.len());
    if end - start < 16_000 {
      break; // drop a <1 s tail; both paths drop it identically
    }
    out.push((start as u64, samples[start..end].to_vec()));
    start = end;
  }
  out
}

#[test]
#[ignore = "loads ONNX + wav (slow); run with --ignored"]
fn split_reproduces_bundled_diarizer() {
  let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/parity/fixtures/01_dialogue/clip_16k.wav");
  if !wav.exists() {
    eprintln!("[skip] {} not present", wav.display());
    return;
  }
  let emb_path = embed_path();
  if !emb_path.exists() {
    eprintln!(
      "[skip] {} not present; run scripts/download-embed-model.sh",
      emb_path.display()
    );
    return;
  }

  let samples = load_wav(&wav);
  let ranges = ranges_of(&samples);
  assert!(!ranges.is_empty(), "fixture must yield at least one range");

  let opts = StreamingOfflineOptions::new();

  // ── Path A: bundled StreamingOfflineDiarizer ──────────────────────
  let bundled_spans = {
    let mut seg = SegmentModel::bundled().expect("seg");
    let mut emb = EmbedModel::from_file(&emb_path).expect("embed");
    let plda = PldaTransform::new().expect("plda");
    let mut diarizer = StreamingOfflineDiarizer::new(opts.clone());
    for (start, pcm) in &ranges {
      diarizer
        .push_voice_range(&mut seg, &mut emb, *start, pcm)
        .expect("push");
    }
    diarizer.finalize(&plda).expect("finalize")
  };

  // ── Path B: StreamingEmbedder + cluster_ranges ────────────────────
  let split_spans = {
    let mut seg = SegmentModel::bundled().expect("seg");
    let mut emb = EmbedModel::from_file(&emb_path).expect("embed");
    let plda = PldaTransform::new().expect("plda");
    let mut embedder = StreamingEmbedder::new(opts.clone());
    let mut carriers = Vec::new();
    for (start, pcm) in &ranges {
      carriers.push(
        embedder
          .push(&mut seg, &mut emb, *start, pcm)
          .expect("push"),
      );
    }
    embedder.finish();
    cluster_ranges(&carriers, &plda, &opts).expect("cluster")
  };

  // ── Exact parity ─────────────────────────────────────────────────
  assert_eq!(
    bundled_spans.len(),
    split_spans.len(),
    "span count differs: bundled={} split={}",
    bundled_spans.len(),
    split_spans.len()
  );
  for (i, (a, b)) in bundled_spans.iter().zip(split_spans.iter()).enumerate() {
    assert_eq!(a.start_sample(), b.start_sample(), "span {i} start differs");
    assert_eq!(a.end_sample(), b.end_sample(), "span {i} end differs");
    assert_eq!(a.speaker_id(), b.speaker_id(), "span {i} speaker differs");
  }
}
