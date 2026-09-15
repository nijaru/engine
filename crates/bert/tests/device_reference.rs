//! Numerically qualify the device encoder against an independent reference.
//!
//! `crates/batch/tests/fixtures/bert-tiny` holds weights generated with a fixed
//! seed by `generate.py` and expected values produced by Hugging Face
//! `transformers`, which is an independent implementation of the same equations.
//! Compiling is not device evidence, so this test is ignored by default and must be
//! run serially with an idle GPU.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use engine_bert::{CudaBertEncoder, EncoderRequest};
use serde_json::Value;

/// Absolute tolerance for one encoder output element. The reference is an
/// independent fp32 implementation, so this covers accumulation-order differences
/// rather than permitting a different algorithm.
const TOLERANCE: f32 = 2.0e-5;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../batch/tests/fixtures/bert-tiny")
}

struct Case {
    name: String,
    token_ids: Vec<u32>,
    token_type_ids: Vec<u32>,
    attention_mask: Vec<i32>,
    last_hidden_state: Vec<f32>,
    pooled_output: Vec<f32>,
}

fn cases() -> Vec<Case> {
    let text = std::fs::read_to_string(fixture().join("reference.json")).expect("reference");
    let document: Value = serde_json::from_str(&text).expect("reference json");
    document["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| {
            // The reference was written from fp32 torch values, so narrowing the
            // JSON doubles back to f32 recovers those values exactly.
            #[allow(
                clippy::cast_possible_truncation,
                reason = "reference values were serialized from fp32"
            )]
            let numbers = |key: &str| -> Vec<f32> {
                case[key]
                    .as_array()
                    .expect(key)
                    .iter()
                    .map(|value| value.as_f64().expect("number") as f32)
                    .collect()
            };
            Case {
                name: case["name"].as_str().expect("name").to_owned(),
                token_ids: case["token_ids"]
                    .as_array()
                    .expect("token_ids")
                    .iter()
                    .map(|value| u32::try_from(value.as_u64().expect("id")).expect("id fits"))
                    .collect(),
                token_type_ids: case["token_type_ids"]
                    .as_array()
                    .expect("token_type_ids")
                    .iter()
                    .map(|value| u32::try_from(value.as_u64().expect("type")).expect("type fits"))
                    .collect(),
                attention_mask: case["attention_mask"]
                    .as_array()
                    .expect("attention_mask")
                    .iter()
                    .map(|value| i32::try_from(value.as_u64().expect("flag")).expect("flag fits"))
                    .collect(),
                last_hidden_state: numbers("last_hidden_state"),
                pooled_output: numbers("pooled_output"),
            }
        })
        .collect()
}

/// Compare against the reference and report the worst deviation, so the recorded
/// evidence carries a measured number rather than only a pass.
fn assert_close(actual: &[f32], expected: &[f32], case: &str, label: &str) -> f32 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case}/{label}: element count differs"
    );
    let mut worst = 0.0_f32;
    let mut worst_index = 0_usize;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let deviation = (actual - expected).abs();
        if deviation > worst {
            worst = deviation;
            worst_index = index;
        }
    }
    assert!(
        worst <= TOLERANCE,
        "{case}/{label}: worst deviation {worst} at {worst_index} \
         (actual {}, expected {}) exceeds {TOLERANCE}",
        actual[worst_index],
        expected[worst_index]
    );
    println!(
        "{case}/{label}: max_abs_deviation={worst:e} elements={}",
        actual.len()
    );
    worst
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn device_encoder_matches_the_transformers_reference() {
    let encoder = CudaBertEncoder::load(fixture(), 0).expect("prepare encoder");
    let cases = cases();
    assert!(!cases.is_empty(), "the reference must carry cases");
    let mut worst_hidden = 0.0_f32;
    let mut worst_pooled = 0.0_f32;
    for case in &cases {
        let request = EncoderRequest {
            token_ids: case.token_ids.clone(),
            token_type_ids: case.token_type_ids.clone(),
            attention_mask: case.attention_mask.clone(),
        };
        let output = encoder.encode(&request).expect("encode");
        let hidden_deviation = assert_close(
            &output.last_hidden_state,
            &case.last_hidden_state,
            &case.name,
            "last_hidden_state",
        );
        let pooled_deviation = assert_close(
            &output.pooled,
            &case.pooled_output,
            &case.name,
            "pooled_output",
        );
        assert!(
            output
                .last_hidden_state
                .iter()
                .all(|value| value.is_finite()),
            "{}: hidden state must stay finite",
            case.name
        );
    }
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn submission_reports_real_completion_and_geometry() {
    let encoder = CudaBertEncoder::load(fixture(), 0).expect("prepare encoder");
    let sequence = 4;
    let request = EncoderRequest::single_segment(vec![4, 1, 9, 3]);
    let bytes = encoder.request_bytes(sequence).expect("geometry");
    assert!(bytes > 0);
    let mut submission = encoder.submit(&request).expect("submit");
    assert_eq!(submission.device_bytes(), bytes);
    submission.synchronize().expect("synchronize");
    assert!(submission.is_complete().expect("completion"));
    let output = submission.read().expect("read");
    assert_eq!(output.last_hidden_state.len(), sequence * 12);
    assert_eq!(output.pooled.len(), 12);
    // Re-reading after completion is stable and does not re-run the request.
    let repeated = submission.read().expect("read again");
    assert_eq!(repeated.last_hidden_state, output.last_hidden_state);
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn invalid_requests_are_refused_without_touching_the_device() {
    let encoder = CudaBertEncoder::load(fixture(), 0).expect("prepare encoder");
    let too_long = EncoderRequest::single_segment(vec![1; 17]);
    assert!(encoder.submit(&too_long).is_err());
    let unknown_token = EncoderRequest::single_segment(vec![99]);
    assert!(encoder.submit(&unknown_token).is_err());
    let mismatched = EncoderRequest {
        token_ids: vec![1, 2],
        token_type_ids: vec![0],
        attention_mask: vec![1, 1],
    };
    assert!(encoder.submit(&mismatched).is_err());
    // The encoder remains usable after a refused request.
    let output = encoder
        .encode(&EncoderRequest::single_segment(vec![5]))
        .expect("healthy request");
    assert!(output.pooled.iter().all(|value| value.is_finite()));
}
