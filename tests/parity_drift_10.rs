//! Diagnostic: measure segmentation + embedding drift between dia
//! (ORT) and pyannote (PyTorch) on the captured 10_mrbeast_clean_water
//! audio. Reports cell-level differences so we can pick the right
//! follow-up fix path (tch backend for embed, tch backend for
//! segmentation, or torchaudio fbank port).
//!
//! Run with:
//!   cargo test --test parity_drift_10 --features ort,bundled-segmentation -- --ignored --nocapture

#![cfg(all(feature = "ort", feature = "bundled-segmentation"))]

use diarization::{
  embed::EmbedModel,
  segment::{FRAMES_PER_WINDOW, POWERSET_CLASSES, SegmentModel, powerset},
};
use std::{fs::File, io::BufReader, path::PathBuf};

/// Captured pyannote intermediates (`segmentations.npz`,
/// `raw_embeddings.npz`, …) live under `tests/parity/fixtures/<name>/`
/// in dia's repo. Audio + reference RTTM live in the sister
/// `audio-fixtures` repo (resolved via `DIA_AUDIO_FIXTURES` or
/// `../audio-fixtures` sibling default).
fn fixture_dir() -> PathBuf {
  let f =
    std::env::var("DIA_DRIFT_FIXTURE").unwrap_or_else(|_| "10_mrbeast_clean_water".to_string());
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/parity/fixtures/{f}"))
}

fn audio_fixtures_root() -> PathBuf {
  if let Some(env) = std::env::var_os("DIA_AUDIO_FIXTURES") {
    return PathBuf::from(env);
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .map(|p| p.join("audio-fixtures"))
    .unwrap_or_else(|| PathBuf::from("../audio-fixtures"))
}

fn fixture_wav_path(name: &str) -> Option<PathBuf> {
  let root = audio_fixtures_root();
  for codec in ["pcm_s16le", "pcm_f32le"] {
    let p = root.join(codec).join(format!("{name}.wav"));
    if p.exists() {
      return Some(p);
    }
  }
  None
}

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

fn read_npz_f32(path: &PathBuf, key: &str) -> (Vec<f32>, Vec<u64>) {
  use npyz::npz::NpzArchive;
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("npz");
  let npy = z
    .by_name(key)
    .expect("query")
    .unwrap_or_else(|| panic!("array {key} not in {}", path.display()));
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<f32> = npy.into_vec().expect("decode");
  (data, shape)
}

#[test]
#[ignore = "loads audio-fixtures wav, requires audio + ONNX models"]
fn measure_segmentation_drift_10() {
  let dir = fixture_dir();
  let fixture_name = std::env::var("DIA_DRIFT_FIXTURE")
    .unwrap_or_else(|_| "10_mrbeast_clean_water".to_string());
  let wav_path = match fixture_wav_path(&fixture_name) {
    Some(p) => p,
    None => {
      eprintln!(
        "skip: {fixture_name}.wav not under {} — set DIA_AUDIO_FIXTURES",
        audio_fixtures_root().display()
      );
      return;
    }
  };
  let samples = load_wav(&wav_path);
  let (py_seg, py_shape) = read_npz_f32(&dir.join("segmentations.npz"), "segmentations");
  let py_chunks = py_shape[0] as usize;
  let py_frames = py_shape[1] as usize;
  let py_speakers = py_shape[2] as usize;
  assert_eq!(py_frames, FRAMES_PER_WINDOW);
  assert_eq!(py_speakers, 3);

  let mut model = SegmentModel::bundled().expect("bundled");
  let win = 160_000usize; // WINDOW_SAMPLES at 16 kHz
  let step = 16_000usize; // dia + pyannote step (1 s @ 16 kHz)
  let dia_chunks = if samples.len() <= win {
    1
  } else {
    (samples.len() - win) / step + 1
  };
  eprintln!("[seg_drift] dia computes {dia_chunks} chunks; pyannote captured {py_chunks}");
  let common = dia_chunks.min(py_chunks);

  let mut padded = vec![0.0_f32; win];
  let mut total_cells = 0usize;
  let mut diff_cells = 0usize;
  let mut max_logit_diff: f32 = 0.0;
  let mut diff_chunks: Vec<usize> = Vec::new();
  for c in 0..common {
    padded.fill(0.0);
    let start = c * step;
    let end = (start + win).min(samples.len());
    let n = end - start;
    if n > 0 {
      padded[..n].copy_from_slice(&samples[start..end]);
    }
    let logits = model.infer(&padded).expect("infer");
    assert_eq!(logits.len(), FRAMES_PER_WINDOW * POWERSET_CLASSES);

    let mut chunk_diffs = 0usize;
    for f in 0..FRAMES_PER_WINDOW {
      let mut row = [0.0_f32; POWERSET_CLASSES];
      for k in 0..POWERSET_CLASSES {
        row[k] = logits[f * POWERSET_CLASSES + k];
      }
      let probs = powerset::softmax_row(&row);
      let speakers = powerset::powerset_to_speakers_hard(&probs);
      for s in 0..3 {
        total_cells += 1;
        let py_v = py_seg[(c * py_frames + f) * py_speakers + s];
        if (speakers[s] - py_v).abs() > 0.5 {
          diff_cells += 1;
          chunk_diffs += 1;
        }
      }
    }
    if chunk_diffs > 0 {
      diff_chunks.push(c);
    }
    // Track f32 logit drift indirectly by checking the raw logits
    // against an effective per-frame max (gross sanity). True f32
    // diff would need pyannote's logits captured, which the fixture
    // doesn't include; the post-binarize cell diff is the meaningful
    // metric.
    let max_l = logits
      .iter()
      .copied()
      .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    if max_l > max_logit_diff {
      max_logit_diff = max_l;
    }
  }
  eprintln!(
    "[seg_drift] cells differing post-binarization: {diff_cells}/{total_cells} ({:.4}%)",
    100.0 * (diff_cells as f64) / (total_cells.max(1) as f64)
  );
  eprintln!(
    "[seg_drift] chunks with any diff: {}/{common} (first 20: {:?})",
    diff_chunks.len(),
    &diff_chunks[..20.min(diff_chunks.len())]
  );
}

#[test]
#[ignore = "loads audio-fixtures wav, requires audio + ONNX models"]
fn measure_embedding_drift_10() {
  let dir = fixture_dir();
  let fixture_name = std::env::var("DIA_DRIFT_FIXTURE")
    .unwrap_or_else(|_| "10_mrbeast_clean_water".to_string());
  let wav_path = match fixture_wav_path(&fixture_name) {
    Some(p) => p,
    None => {
      eprintln!(
        "skip: {fixture_name}.wav not under {} — set DIA_AUDIO_FIXTURES",
        audio_fixtures_root().display()
      );
      return;
    }
  };
  let emb_path: PathBuf = std::env::var_os("DIA_EMBED_MODEL_PATH")
    .map(PathBuf::from)
    .unwrap_or_else(|| {
      PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx")
    });
  if !emb_path.exists() {
    eprintln!("skip: embed model not at {}", emb_path.display());
    return;
  }
  let samples = load_wav(&wav_path);
  let (py_seg, _seg_shape) = read_npz_f32(&dir.join("segmentations.npz"), "segmentations");
  let (py_emb, emb_shape) = read_npz_f32(&dir.join("raw_embeddings.npz"), "embeddings");
  let py_chunks = emb_shape[0] as usize;
  let py_speakers = emb_shape[1] as usize;
  let py_dim = emb_shape[2] as usize;
  assert_eq!(py_dim, 256);

  let mut model = EmbedModel::from_file(&emb_path).expect("load embed");
  let win = 160_000usize;
  let step = 16_000usize;
  let dia_chunks = if samples.len() <= win {
    1
  } else {
    (samples.len() - win) / step + 1
  };
  let common = dia_chunks.min(py_chunks);

  let mut padded = vec![0.0_f32; win];
  let mut max_abs_err: f32 = 0.0;
  let mut max_loc = (0usize, 0usize);
  let mut sum_sq_err: f64 = 0.0;
  let mut compared_pairs = 0usize;
  let mut large_drift_pairs = 0usize;
  for c in 0..common {
    padded.fill(0.0);
    let start = c * step;
    let end_w = (start + win).min(samples.len());
    let n = end_w - start;
    if n > 0 {
      padded[..n].copy_from_slice(&samples[start..end_w]);
    }
    // Pyannote applies `exclude_overlap=True` (community-1 default)
    // by zeroing frames with multiple active speakers, then falls
    // back to the full mask when fewer than `min_num_frames=2` clean
    // frames remain. The diagnostic must mirror this to compare like
    // for like: the captured raw_embeddings.npz was produced from
    // overlap-excluded masks, so feeding the full mask to dia's
    // `embed_chunk_with_frame_mask` would compare different inputs.
    let mut clean_frame = [false; FRAMES_PER_WINDOW];
    for f in 0..FRAMES_PER_WINDOW {
      let mut active_count = 0u8;
      for sx in 0..py_speakers {
        if py_seg[(c * FRAMES_PER_WINDOW + f) * 3 + sx] >= 0.5 {
          active_count += 1;
        }
      }
      clean_frame[f] = active_count < 2;
    }

    for s in 0..py_speakers {
      // Build mask from captured pyannote segmentations.
      let mut mask = [false; FRAMES_PER_WINDOW];
      let mut any_active = false;
      for f in 0..FRAMES_PER_WINDOW {
        let v = py_seg[(c * FRAMES_PER_WINDOW + f) * 3 + s];
        mask[f] = v >= 0.5;
        any_active |= mask[f];
      }
      if !any_active {
        continue;
      }
      // Overlap-exclusion (matches dia's offline/owned.rs and
      // pyannote's `get_embeddings(exclude_overlap=True)`).
      let mut used_mask = [false; FRAMES_PER_WINDOW];
      let mut clean_count = 0usize;
      for f in 0..FRAMES_PER_WINDOW {
        let v = mask[f] && clean_frame[f];
        used_mask[f] = v;
        if v {
          clean_count += 1;
        }
      }
      if clean_count <= 2 {
        used_mask = mask;
      }
      let raw = match model.embed_chunk_with_frame_mask(&padded, &used_mask) {
        Ok(v) => v,
        Err(e) => {
          eprintln!("  c={c} s={s}: embed failed: {e}");
          continue;
        }
      };
      let py_base = (c * py_speakers + s) * py_dim;
      let mut row_sq_err: f64 = 0.0;
      for d in 0..py_dim {
        let dia_v = raw[d];
        let py_v = py_emb[py_base + d];
        let err = (dia_v - py_v).abs();
        if err > max_abs_err {
          max_abs_err = err;
          max_loc = (c, s);
        }
        row_sq_err += (dia_v - py_v) as f64 * (dia_v - py_v) as f64;
      }
      sum_sq_err += row_sq_err;
      compared_pairs += 1;
      if row_sq_err.sqrt() > 0.01 {
        large_drift_pairs += 1;
      }
    }
  }
  eprintln!(
    "[emb_drift] compared {compared_pairs} (chunk, speaker) embeddings (active only); \
     max abs f32 element error = {max_abs_err:.3e} at (c={}, s={}); \
     mean L2 row error = {:.3e}; pairs with L2 row error > 0.01: {large_drift_pairs}/{compared_pairs}",
    max_loc.0,
    max_loc.1,
    (sum_sq_err / compared_pairs.max(1) as f64).sqrt()
  );
}
