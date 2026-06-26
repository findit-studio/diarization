//! End-to-end RTTM parity test: full pyannote pipeline (5a + 5b + 5c)
//! → RTTM, compared against captured `reference.rttm`.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::DVector;
use npyz::npz::NpzArchive;

use crate::{
  pipeline::{AssignEmbeddingsInput, assign_embeddings},
  reconstruct::{
    ReconstructInput, SlidingWindow, discrete_to_spans, reconstruct, spans_to_rttm_lines,
  },
};

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

/// Reference RTTMs live in the sister `audio-fixtures` repo
/// (`references/<name>.rttm`). Resolved via `DIA_AUDIO_FIXTURES` env
/// var or the `../audio-fixtures` sibling default.
fn audio_fixtures_reference(fixture_name: &str) -> PathBuf {
  let root = std::env::var_os("DIA_AUDIO_FIXTURES")
    .map(PathBuf::from)
    .unwrap_or_else(|| {
      repo_root()
        .parent()
        .map(|p| p.join("audio-fixtures"))
        .unwrap_or_else(|| PathBuf::from("../audio-fixtures"))
    });
  root.join("references").join(format!("{fixture_name}.rttm"))
}

fn read_npz_array<T>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>)
where
  T: npyz::Deserialize,
{
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode array");
  (data, shape)
}

#[test]
fn rttm_matches_pyannote_reference_01_dialogue() {
  run_rttm_parity("01_dialogue", "clip_16k");
}

#[test]
fn rttm_matches_pyannote_reference_02_pyannote_sample() {
  run_rttm_parity("02_pyannote_sample", "clip_16k");
}

#[test]
fn rttm_matches_pyannote_reference_03_dual_speaker() {
  run_rttm_parity("03_dual_speaker", "clip_16k");
}

#[test]
fn rttm_matches_pyannote_reference_04_three_speaker() {
  run_rttm_parity("04_three_speaker", "clip_16k");
}

#[test]
fn rttm_matches_pyannote_reference_05_four_speaker() {
  run_rttm_parity("05_four_speaker", "clip_16k");
}

/// 06_long_recording (T=1004) — RTTM parity.
/// Pipeline + reconstruct grid are now bit-exact (Kahan-summed VBx +
/// `np.unique`-equivalent AHC canonicalization). Per-line RTTM is
/// structurally bit-exact, with at most ≤1ms drift on the `duration`
/// field for 2/346 lines on this fixture due to f64 subtraction
/// rounding at large timestamps (`end - start` for spans starting
/// past 500s). The per-line tolerance in `run_rttm_parity` accepts
/// this ULP-class drift while flagging any structural deviation.
#[test]
fn rttm_matches_pyannote_reference_06_long_recording() {
  run_rttm_parity("06_long_recording", "clip_16k");
}

#[test]
#[ignore = "ad-hoc capture; localizes RTTM parity on 10_mrbeast_clean_water"]
fn rttm_matches_pyannote_reference_10_mrbeast_clean_water() {
  run_rttm_parity("10_mrbeast_clean_water", "clip_16k");
}

fn run_rttm_parity(fixture_dir: &str, uri: &str) {
  crate::parity_fixtures_or_skip!();
  let base = format!("tests/parity/fixtures/{fixture_dir}");

  // ── Stage 5a + 5b: produce discrete_diarization ───────────────────
  let raw_path = fixture(&format!("{base}/raw_embeddings.npz"));
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  // Row-major flat `[c][s][d]`, matching the new
  // `AssignEmbeddingsInput::embeddings: &[f64]` contract.
  let embeddings: Vec<f64> = raw_flat.iter().map(|&v| v as f64).collect();
  let seg_path = fixture(&format!("{base}/segmentations.npz"));
  let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
  let num_frames_per_chunk = seg_shape[1] as usize;
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();
  let plda_path = fixture(&format!("{base}/plda_embeddings.npz"));
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  let _num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let post_plda: &[f64] = &post_plda_flat;
  let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);
  let (chunk_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  let train_chunk_idx: Vec<usize> = chunk_idx_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_idx_i64.iter().map(|&v| v as usize).collect();
  let ahc_path = fixture(&format!("{base}/ahc_state.npz"));
  let (threshold_data, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let vbx_path = fixture(&format!("{base}/vbx_state.npz"));
  let (fa_arr, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_arr, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_arr, _) = read_npz_array::<i64>(&vbx_path, "max_iters");

  let pipeline_input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames_per_chunk,
    post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  )
  .with_threshold(threshold_data[0])
  .with_fa(fa_arr[0])
  .with_fb(fb_arr[0])
  .with_max_iters(max_iters_arr[0] as usize);
  let hard_clusters = assign_embeddings(&pipeline_input).expect("assign_embeddings");

  let recon_path = fixture(&format!("{base}/reconstruction.npz"));
  let (count_u8, count_shape) = read_npz_array::<u8>(&recon_path, "count");
  let num_output_frames = count_shape[0] as usize;
  let (chunk_start_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_start");
  let (chunk_dur_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_duration");
  let (chunk_step_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_step");
  let (frame_start_arr, _) = read_npz_array::<f64>(&recon_path, "frame_start");
  let (frame_dur_arr, _) = read_npz_array::<f64>(&recon_path, "frame_duration");
  let (frame_step_arr, _) = read_npz_array::<f64>(&recon_path, "frame_step");
  let (min_dur_off_arr, _) = read_npz_array::<f64>(&recon_path, "min_duration_off");
  let chunks_sw = SlidingWindow::new(chunk_start_arr[0], chunk_dur_arr[0], chunk_step_arr[0]);
  let frames_sw = SlidingWindow::new(frame_start_arr[0], frame_dur_arr[0], frame_step_arr[0]);
  let min_duration_off = min_dur_off_arr[0];

  let recon_input = ReconstructInput::new(
    &segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    &hard_clusters,
    &count_u8,
    num_output_frames,
    chunks_sw,
    frames_sw,
  );
  let grid = reconstruct(&recon_input).expect("reconstruct");
  let num_clusters = grid.len() / num_output_frames;

  // ── Stage 5c: discrete grid → RTTM spans ──────────────────────────
  let spans = discrete_to_spans(
    &grid,
    num_output_frames,
    num_clusters,
    frames_sw,
    min_duration_off,
  );
  let lines = spans_to_rttm_lines(&spans, uri);

  // ── Compare to reference.rttm ─────────────────────────────────────
  // Lives in the sister `audio-fixtures` repo (`references/`), not
  // in `base` — the captured pyannote intermediates (npz/npy) stay
  // local but reference RTTMs are shared across the `dia` family.
  let ref_path = audio_fixtures_reference(fixture_dir);
  let ref_text = std::fs::read_to_string(&ref_path).expect("read reference.rttm");
  let ref_lines: Vec<&str> = ref_text.lines().filter(|l| !l.is_empty()).collect();

  // Quick line-count check.
  eprintln!(
    "[parity_rttm] generated {} lines, reference has {} lines",
    lines.len(),
    ref_lines.len()
  );

  // Diff per line: warn on mismatches but don't fail bit-exact yet —
  // the reference file uses pyannote's relabeling (SPEAKER_NN by
  // encounter order). Our output should have the same encounter
  // order if hard_clusters identity-maps to scipy's labels; if it
  // doesn't, the labels need a permutation. For this test, count:
  // line-count parity + total per-cluster duration parity (within
  // tolerance).

  // Parse a list of (start, duration, label) from each side.
  fn parse_rttm(lines: impl Iterator<Item = String>) -> Vec<(f64, f64, String)> {
    lines
      .map(|l| {
        let parts: Vec<&str> = l.split_whitespace().collect();
        let start: f64 = parts[3].parse().expect("rttm start");
        let duration: f64 = parts[4].parse().expect("rttm dur");
        let label = parts[7].to_string();
        (start, duration, label)
      })
      .collect()
  }
  let got_parsed = parse_rttm(lines.iter().cloned());
  let want_parsed = parse_rttm(ref_lines.iter().map(|s| s.to_string()));

  // Per-label total active duration. RTTM spans of the same speaker
  // tile a per-frame active region; the totals should match exactly
  // since the per-frame grid is bit-identical.
  use std::collections::HashMap;
  let mut got_total: HashMap<String, f64> = HashMap::new();
  for (_, d, l) in &got_parsed {
    *got_total.entry(l.clone()).or_default() += d;
  }
  let mut want_total: HashMap<String, f64> = HashMap::new();
  for (_, d, l) in &want_parsed {
    *want_total.entry(l.clone()).or_default() += d;
  }
  eprintln!("[parity_rttm] got per-label totals: {got_total:?}");
  eprintln!("[parity_rttm] want per-label totals: {want_total:?}");

  for (label, &want_dur) in &want_total {
    let got_dur = got_total.get(label).copied().unwrap_or(0.0);
    let diff = (got_dur - want_dur).abs();
    assert!(
      diff < 0.05,
      "per-label total duration mismatch for {label}: got {got_dur:.3}s, want {want_dur:.3}s (|Δ|={diff:.3}s)"
    );
  }
  assert_eq!(
    got_parsed.len(),
    want_parsed.len(),
    "RTTM line count differs: got {}, want {}",
    got_parsed.len(),
    want_parsed.len(),
  );

  // Per-line parity. Reference RTTM is sorted by (start, label); our
  // generator does the same. With min_duration_off=0 and identity
  // cluster mapping every span should line up. Strict-string-equal
  // is the contract for the start, file-uri, channel, and speaker
  // fields. Duration is allowed to differ by up to one ULP at
  // 3-decimal precision (`<= 1ms`) — Segment.duration in pyannote is
  // `end - start`, which loses sub-millisecond precision through f64
  // subtraction at large timestamps (e.g. 561s + 3.3075s round to
  // 3.308 vs 3.307 depending on whether the path passes through a
  // precomputed `timestamps[i]` list or recomputes
  // `frame_start + i * step + duration / 2` inline). Both round to
  // the same RTTM line at 1ms precision, and downstream DER /
  // per-label totals (already enforced above to <50ms tolerance) are
  // unaffected.
  let mut mismatches = 0usize;
  let mut duration_only_mismatches = 0usize;
  let mut first_mismatch: Option<(usize, String, String)> = None;
  for (i, (got_line, want_line)) in lines.iter().zip(ref_lines.iter()).enumerate() {
    let got = got_line.trim();
    let want = want_line.trim();
    if got == want {
      continue;
    }
    // Parse: SPEAKER <uri> 1 <start> <duration> <NA> <NA> <label> <NA> <NA>
    let g_fields: Vec<&str> = got.split_whitespace().collect();
    let w_fields: Vec<&str> = want.split_whitespace().collect();
    if g_fields.len() == 10 && w_fields.len() == 10 {
      let same_structure = g_fields[0] == w_fields[0]
        && g_fields[1] == w_fields[1]
        && g_fields[2] == w_fields[2]
        && g_fields[3] == w_fields[3] // start (3-decimal exact)
        && g_fields[5..] == w_fields[5..]; // <NA> <NA> <label> <NA> <NA>
      let g_dur: f64 = g_fields[4].parse().unwrap_or(f64::NAN);
      let w_dur: f64 = w_fields[4].parse().unwrap_or(f64::NAN);
      // 1ms diff at the 3-decimal RTTM precision can exceed
      // f64::EPSILON in absolute terms — `2.228 - 2.227` rounds to
      // `1.0e-3 + ~2.5e-16`. Use 1.5ms as the practical tolerance:
      // strictly less than the next-higher RTTM precision step (10ms).
      if same_structure && (g_dur - w_dur).abs() <= 0.0015 {
        duration_only_mismatches += 1;
        continue;
      }
    }
    mismatches += 1;
    if first_mismatch.is_none() {
      first_mismatch = Some((i, got_line.clone(), (*want_line).to_string()));
    }
  }
  eprintln!(
    "[parity_rttm] structural mismatches: {mismatches}/{}; \
     duration-only ≤1ms drift: {duration_only_mismatches}/{}; \
     first structural mismatch: {first_mismatch:?}",
    lines.len(),
    lines.len()
  );
  assert!(
    mismatches == 0,
    "per-line RTTM structural mismatch ({mismatches}/{}); first: {first_mismatch:?}",
    lines.len()
  );
}
