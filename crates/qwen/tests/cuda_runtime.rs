#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use engine_qwen::{QwenCuda, QwenLoadOptions};
use ribn::{
    Engine, EngineConfig, Event, FinishReason, GenerationExecutor, GenerationOptions,
    SchedulePolicy, TokenRequest,
};

struct Reference {
    artifact: String,
    prompt: Vec<u32>,
    output: Vec<u32>,
}

fn reference(text: &str) -> Result<Reference, String> {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() != 3 {
        return Err(
            "reference must contain artifact identity, prompt IDs, and output IDs on three lines"
                .into(),
        );
    }
    let hash = lines[0]
        .strip_prefix("gguf:sha256:")
        .ok_or("reference needs a GGUF SHA-256 identity")?;
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("reference SHA-256 must be 64 lowercase hexadecimal digits".into());
    }
    let parse = |line: &str| -> Result<Vec<u32>, String> {
        let tokens = line
            .split_whitespace()
            .map(|value| value.parse::<u32>().map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        if tokens.is_empty() {
            return Err("reference token lists must be nonempty".into());
        }
        Ok(tokens)
    };
    Ok(Reference {
        artifact: lines[0].into(),
        prompt: parse(lines[1])?,
        output: parse(lines[2])?,
    })
}

#[test]
fn reference_fixture_requires_exact_artifact_and_nonempty_token_streams() {
    let artifact = format!("gguf:sha256:{}", "0".repeat(64));
    let parsed = reference(&format!("{artifact}\n1 2\n3 4\n")).unwrap();
    assert_eq!(parsed.prompt, [1, 2]);
    assert_eq!(parsed.output, [3, 4]);
    assert!(reference("qwen\n1\n2").is_err());
    assert!(reference(&format!("{artifact}\n\n2")).is_err());
    assert!(reference(&format!("{artifact}\n1\n-1")).is_err());
}

#[test]
#[ignore = "requires an idle CUDA GPU, RIBN_MODEL, and independently recorded RIBN_REFERENCE token fixture"]
fn prepared_qwen_matches_reference_and_preserves_cancelled_peers() {
    let model = std::env::var("RIBN_MODEL").expect("RIBN_MODEL GGUF path");
    let fixture = std::env::var("RIBN_REFERENCE").expect("RIBN_REFERENCE three-line fixture path");
    let reference = reference(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    assert!(
        reference.output.len() >= 4,
        "hardware fixture needs at least four output tokens"
    );
    for (concurrency, cancel_decode) in [
        (1, false),
        (2, false),
        (8, false),
        (9, false),
        (2, true),
        (8, true),
    ] {
        run_case(&model, &reference, concurrency, cancel_decode);
    }
}

fn run_case(model: &str, reference: &Reference, concurrency: usize, cancel_decode: bool) {
    let max_output_tokens = u32::try_from(reference.output.len()).unwrap();
    let prompt_tokens = u32::try_from(reference.prompt.len()).unwrap();
    let context_tokens = prompt_tokens.checked_add(max_output_tokens).unwrap();
    let prepared = QwenCuda::load_gguf(
        model,
        QwenLoadOptions {
            context_tokens,
            max_sequences: concurrency,
            ..QwenLoadOptions::default()
        },
    )
    .expect("prepare Qwen");
    // This adapter's name is the full artifact hash, not a caller-supplied label.
    assert_eq!(prepared.info().name, reference.artifact);
    let mut engine = Engine::new(
        prepared,
        EngineConfig {
            max_active_requests: concurrency,
            max_queued_requests: 0,
            max_queued_input_tokens: reference.prompt.len() as u64 * concurrency as u64,
            max_buffered_events: concurrency * 4,
            max_events_per_request: 64,
        },
        SchedulePolicy::default(),
    )
    .unwrap();
    let requests = (0..concurrency)
        .map(|_| {
            engine
                .enqueue(TokenRequest::new(
                    reference.prompt.clone(),
                    GenerationOptions {
                        max_output_tokens,
                        ..GenerationOptions::default()
                    },
                ))
                .unwrap()
        })
        .collect::<Vec<_>>();
    engine.step().unwrap();
    let cancelled = cancel_decode.then(|| requests[0]);
    let mut cancel_issued = false;
    let mut outputs = requests
        .iter()
        .map(|&request| (request, Vec::new()))
        .collect::<HashMap<_, _>>();
    let mut finished = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(600);
    while finished.len() < concurrency {
        assert!(Instant::now() < deadline, "Qwen qualification stalled");
        engine.step().unwrap();
        // The step that commits final prefill has already submitted decode.
        if !cancel_issued
            && let Some(request) = cancelled
            && engine
                .committed_prefix(request)
                .is_some_and(|prefix| prefix >= prompt_tokens)
            && engine.status().in_flight
        {
            engine.cancel(request).unwrap();
            cancel_issued = true;
        }
        while let Some(event) = engine.pop_event() {
            match event {
                Event::Token { request, token } => {
                    outputs.get_mut(&request).unwrap().push(token);
                }
                Event::Finished {
                    request, reason, ..
                } => {
                    assert!(finished.insert(request, reason).is_none());
                }
            }
        }
        std::thread::yield_now();
    }
    assert_eq!(engine.status().active_sequences, 0);
    for request in requests {
        if Some(request) == cancelled {
            assert_eq!(finished[&request], FinishReason::Cancelled);
            // Final prefill's already committed output remains deliverable;
            // cancellation suppresses only the uncommitted decode result.
            assert!(cancel_issued);
            assert_eq!(
                outputs[&request],
                reference.output[..outputs[&request].len()]
            );
            assert!(outputs[&request].len() < reference.output.len());
        } else {
            assert_eq!(finished[&request], FinishReason::Length);
            assert_eq!(
                outputs[&request], reference.output,
                "concurrency {concurrency}"
            );
        }
    }
    engine.shutdown().unwrap();
}
