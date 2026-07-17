//! Smoke test against the bundled pyannote/segmentation-3.0 ONNX model.
//! Skipped by default (`#[ignore]`); run with:
//!
//!     cargo test --test integration_segment -- --ignored

#![cfg(all(feature = "ort", feature = "bundled-segmentation"))]

use diarization::segment::{SegmentModel, SegmentOptions, Segmenter, SegmenterExt};

#[test]
#[ignore = "exercises ONNX runtime"]
fn smoke_test_runs_inference_on_synthetic_audio() {
  let mut model = SegmentModel::bundled().expect("bundled model loads");
  let mut seg = Segmenter::new(SegmentOptions::default());

  // 12 seconds of low-amplitude noise — exercise tail anchoring.
  let mut pcm = vec![0.0f32; 16_000 * 12];
  for (i, x) in pcm.iter_mut().enumerate() {
    *x = ((i as f32) * 0.0001).sin() * 0.01;
  }

  let mut events: usize = 0;
  seg
    .process_samples(&mut model, &pcm, |_| events += 1)
    .expect("ok");
  seg.finish_stream(&mut model, |_| events += 1).expect("ok");

  // We don't assert specific events on synthetic noise (the model may
  // emit none); the point is that the pipeline runs end-to-end without
  // panicking and the inference contract holds.
  let _ = events;
}
