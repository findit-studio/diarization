//! End-to-end parity test: run the full dia pipeline on each
//! audio-fixtures clip and assert the produced RTTM matches the
//! captured pyannote 4.0.4 reference (`reference.rttm`) for both
//! speaker count and segment count.
//!
//! Audio fixtures + reference RTTMs live in the sister
//! [`audio-fixtures`] repo. Resolved at test time via the
//! `DIA_AUDIO_FIXTURES` env var (default: `../audio-fixtures` —
//! co-located checkout). The dia-specific captured pyannote
//! intermediates (`raw_embeddings.npz`, `segmentations.npz`, …)
//! that 8 of the 14 fixtures need stay under
//! `tests/parity/fixtures/<name>/` in dia's repo.
//!
//! [`audio-fixtures`]: https://github.com/Findit-AI/audio-fixtures
//!
//! `#[ignore]`-gated because they load the bundled segmentation +
//! WeSpeaker ONNX models (the latter via
//! `models/wespeaker_resnet34_lm.onnx`, which is not in-repo —
//! `scripts/download-embed-model.sh` fetches it) and take ~5–60 s
//! per fixture under release builds. Run explicitly:
//!
//! ```
//! cargo test --release --test parity_fixtures_endtoend \
//!   --features ort,bundled-segmentation -- --ignored --nocapture
//! ```
//!
//! Split into one `#[test]` per fixture so each test's stack/heap
//! state (audio buffer, segmentation tensor, embedding cache) is
//! freed between fixtures — running all 14 in a single test
//! function exhausts the test process on macOS.

#![cfg(all(feature = "ort", feature = "bundled-segmentation"))]

use diarization::{
  embed::EmbedModel,
  offline::{OwnedDiarizationPipeline, OwnedPipelineOptions},
  plda::PldaTransform,
  reconstruct::spans_to_rttm_lines,
  segment::SegmentModel,
};
use std::{collections::BTreeSet, path::PathBuf};

/// Resolves the `audio-fixtures` repo root.
///
/// Honors `DIA_AUDIO_FIXTURES` when set; otherwise defaults to
/// `<crate-root>/../audio-fixtures` for a typical sibling-checkout
/// developer layout (`findit-studio/dia` and
/// `findit-studio/audio-fixtures`).
fn audio_fixtures_root() -> PathBuf {
  if let Some(env) = std::env::var_os("DIA_AUDIO_FIXTURES") {
    return PathBuf::from(env);
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .map(|p| p.join("audio-fixtures"))
    .unwrap_or_else(|| PathBuf::from("../audio-fixtures"))
}

/// Locate the wav for a fixture name. The audio-fixtures layout
/// groups wavs by codec (`pcm_s16le/<name>.wav`,
/// `pcm_f32le/<name>.wav`, …); the same `<name>` is used as the
/// stem. Tries `pcm_s16le` first (the bulk of the corpus) and falls
/// back to `pcm_f32le` for clips that landed there (e.g.
/// `01_dialogue` from the upstream pyannote-audio sample collection).
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

fn fixture_reference_rttm(name: &str) -> PathBuf {
  audio_fixtures_root()
    .join("references")
    .join(format!("{name}.rttm"))
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

fn rttm_counts_str(rttm: &str) -> (usize, usize) {
  let mut speakers: BTreeSet<&str> = BTreeSet::new();
  let mut n_segs = 0_usize;
  for line in rttm.lines() {
    if line.trim().is_empty() {
      continue;
    }
    let cols: Vec<&str> = line.split_whitespace().collect();
    if cols.len() < 8 {
      continue;
    }
    speakers.insert(cols[7]);
    n_segs += 1;
  }
  (speakers.len(), n_segs)
}

fn rttm_counts(path: &PathBuf) -> (usize, usize) {
  let body = std::fs::read_to_string(path).expect("read rttm");
  rttm_counts_str(&body)
}

/// Run dia on the audio-fixtures clip for `name` and assert
/// `(speakers, segments)` matches the captured pyannote 4.0.4
/// reference. Each call freshly builds models + pipeline so per-test
/// state is bounded.
///
/// Skips with a clear `eprintln!` if the audio-fixtures repo isn't
/// present (e.g. a contributor running tests without the sibling
/// checkout); CI must `panic!` since silently skipping a parity test
/// in a release build defeats the point. Toggle via
/// `DIA_REQUIRE_PARITY_FIXTURES=1` (or any non-empty `CI` env value).
fn assert_fixture_parity(name: &str) {
  let wav = match fixture_wav_path(name) {
    Some(p) => p,
    None => {
      let require = std::env::var_os("CI").is_some()
        || std::env::var_os("DIA_REQUIRE_PARITY_FIXTURES").is_some();
      let root = audio_fixtures_root();
      if require {
        panic!(
          "{name}.wav not found under {}/{{pcm_s16le,pcm_f32le}}/ — \
           clone https://github.com/Findit-AI/audio-fixtures or set \
           DIA_AUDIO_FIXTURES",
          root.display()
        );
      }
      eprintln!(
        "[skip {name}] {name}.wav not found under {} — set \
         DIA_AUDIO_FIXTURES or check out audio-fixtures as a sibling \
         of dia",
        root.display()
      );
      return;
    }
  };
  let reference = fixture_reference_rttm(name);
  assert!(
    reference.exists(),
    "missing {} (audio-fixtures references/ should ship with the wav)",
    reference.display()
  );

  let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let emb_path: PathBuf = std::env::var_os("DIA_EMBED_MODEL_PATH")
    .map(PathBuf::from)
    .unwrap_or_else(|| crate_root.join("models/wespeaker_resnet34_lm.onnx"));
  if !emb_path.exists() {
    panic!(
      "{} missing — run scripts/download-embed-model.sh",
      emb_path.display()
    );
  }

  let mut seg = SegmentModel::bundled().expect("load segmentation");
  let mut emb = EmbedModel::from_file(&emb_path).expect("load embed");
  let plda = PldaTransform::new().expect("plda");
  let pipeline = OwnedDiarizationPipeline::with_options(OwnedPipelineOptions::new());

  let samples = load_wav(&wav);
  let out = pipeline
    .run(&mut seg, &mut emb, &plda, &samples)
    .expect("diarize");
  let rttm: String = spans_to_rttm_lines(out.spans_slice(), name)
    .into_iter()
    .map(|mut l| {
      l.push('\n');
      l
    })
    .collect();
  let hyp = rttm_counts_str(&rttm);
  let r = rttm_counts(&reference);
  assert_eq!(
    hyp, r,
    "[{name}] ref={}/{} hyp={}/{} (spk/segs mismatch)",
    r.0, r.1, hyp.0, hyp.1
  );
}

macro_rules! fixture_parity_test {
  ($fn_name:ident, $fixture:literal) => {
    #[test]
    #[ignore = "loads ONNX + audio-fixtures wav (slow); run with --ignored"]
    fn $fn_name() {
      assert_fixture_parity($fixture);
    }
  };
}

// 14 audio-fixtures clips. `findit-ai/audio-fixtures` carries every
// wav + reference rttm; layout is codec-keyed (`pcm_s16le/<name>.wav`,
// `pcm_f32le/<name>.wav`) so the wav resolver above tries both.
fixture_parity_test!(parity_01_dialogue, "01_dialogue");
fixture_parity_test!(parity_02_pyannote_sample, "02_pyannote_sample");
fixture_parity_test!(parity_03_dual_speaker, "03_dual_speaker");
fixture_parity_test!(parity_04_three_speaker, "04_three_speaker");
fixture_parity_test!(parity_05_four_speaker, "05_four_speaker");
fixture_parity_test!(parity_06_long_recording, "06_long_recording");
fixture_parity_test!(
  parity_07_yuhewei_dongbei_english,
  "07_yuhewei_dongbei_english"
);
fixture_parity_test!(parity_08_luyu_jinjing_freedom, "08_luyu_jinjing_freedom");
fixture_parity_test!(parity_09_mrbeast_dollar_date, "09_mrbeast_dollar_date");
fixture_parity_test!(parity_10_mrbeast_clean_water, "10_mrbeast_clean_water");
fixture_parity_test!(parity_11_mrbeast_age_race, "11_mrbeast_age_race");
fixture_parity_test!(parity_12_mrbeast_schools, "12_mrbeast_schools");
fixture_parity_test!(parity_13_mrbeast_saved_animals, "13_mrbeast_saved_animals");
fixture_parity_test!(
  parity_14_mrbeast_strongman_robot,
  "14_mrbeast_strongman_robot"
);
