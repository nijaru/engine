#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use engine_core::{ModelProvider, StateRequirement};
use engine_qwen::{QwenCuda, QwenGguf, QwenLoadOptions};
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

#[test]
#[ignore = "requires an idle CUDA GPU, RIBN_MODEL, and independently recorded RIBN_REFERENCE token fixture"]
fn owned_driver_preserves_reference_with_stalled_and_abandoned_peers() {
    use ribn::driver::{Driver, DriverConfig, GenerationStream};

    fn collect(mut stream: GenerationStream) -> Vec<u32> {
        let request = stream.request_id();
        let mut tokens = Vec::new();
        let mut terminal = false;
        while let Some(event) = stream.next_blocking() {
            let event = event.expect("owned driver event");
            assert_eq!(event.request(), request);
            match event {
                Event::Token { token, .. } => tokens.push(token),
                Event::Finished { reason, .. } => {
                    assert_eq!(reason, FinishReason::Length);
                    assert!(!terminal);
                    terminal = true;
                }
            }
        }
        assert!(terminal);
        tokens
    }

    let model = std::env::var("RIBN_MODEL").expect("RIBN_MODEL GGUF path");
    let fixture = std::env::var("RIBN_REFERENCE").expect("RIBN_REFERENCE fixture path");
    let reference = reference(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    let prepared = QwenCuda::load_gguf(
        model,
        QwenLoadOptions {
            context_tokens: u32::try_from(reference.prompt.len() + reference.output.len()).unwrap(),
            max_sequences: 3,
            ..QwenLoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(prepared.info().name, reference.artifact);
    let engine = Engine::new(
        prepared,
        EngineConfig {
            max_active_requests: 3,
            max_queued_requests: 0,
            max_queued_input_tokens: reference.prompt.len() as u64 * 3,
            max_buffered_events: 6,
            max_events_per_request: 2,
        },
        SchedulePolicy::default(),
    )
    .unwrap();
    let (mut owner, handle) = Driver::spawn(
        engine,
        DriverConfig {
            max_requests: 3,
            events_per_request: 1,
            ..DriverConfig::default()
        },
    )
    .unwrap();
    let request = || {
        TokenRequest::new(
            reference.prompt.clone(),
            GenerationOptions {
                max_output_tokens: u32::try_from(reference.output.len()).unwrap(),
                ..GenerationOptions::default()
            },
        )
    };
    let mut abandoned = handle.stream_blocking(request()).unwrap();
    let stalled = handle.stream_blocking(request()).unwrap();
    let healthy = handle.stream_blocking(request()).unwrap();
    assert!(
        matches!(abandoned.next_blocking(), Some(Ok(Event::Token { token, .. })) if token == reference.output[0])
    );
    drop(abandoned);
    let collector = std::thread::spawn(move || collect(healthy));
    let deadline = Instant::now() + Duration::from_secs(600);
    while !collector.is_finished() {
        assert!(
            Instant::now() < deadline,
            "stalled peer blocked the healthy stream"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(collector.join().unwrap(), reference.output);
    assert_eq!(collect(stalled), reference.output);
    owner.shutdown().unwrap();
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

/// Continuation bytes the declared schema needs for `sequences` requests that each
/// reach `tokens` tokens.
fn continuation_demand(requirements: &[StateRequirement], tokens: u32, sequences: usize) -> u64 {
    let per_sequence = requirements
        .iter()
        .map(|requirement| {
            requirement
                .with_capacity(tokens)
                .unwrap()
                .byte_size()
                .unwrap()
        })
        .sum::<u64>();
    per_sequence * u64::try_from(sequences).unwrap()
}

/// A request is charged for the tokens it can reach, not for the model's context.
///
/// The declared continuation bound is four times what any request here reaches, and
/// the authority is sized for exactly three request-sized charges. Three sequences
/// therefore hold continuation at once, which a context-sized charge — larger than the
/// whole authority — could never do.
#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn continuation_capacity_admits_concurrent_requests_a_context_charge_cannot() {
    const CONCURRENCY: usize = 3;
    let model = std::env::var("RIBN_MODEL").expect("RIBN_MODEL GGUF path");
    let fixture = std::env::var("RIBN_REFERENCE").expect("RIBN_REFERENCE fixture path");
    let reference = reference(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    let prompt_tokens = u32::try_from(reference.prompt.len()).unwrap();
    let max_output_tokens = u32::try_from(reference.output.len()).unwrap();
    let reachable = prompt_tokens + max_output_tokens;
    let declared_bound = 16 * reachable;
    // Read the declared state schema without touching the device.
    let requirements = QwenGguf::open_with_kv_block_tokens(&model, declared_bound)
        .unwrap()
        .description()
        .state_requirements()
        .to_vec();
    let request_charge = continuation_demand(&requirements, reachable, 1);
    let capacity = request_charge * u64::try_from(CONCURRENCY).unwrap();
    let context_charge = continuation_demand(&requirements, declared_bound, 1);
    assert!(
        context_charge > request_charge,
        "a request charge ({request_charge} bytes) must be smaller than the context charge \
         ({context_charge} bytes) for this to discriminate"
    );
    assert!(
        2 * context_charge > capacity,
        "the authority ({capacity} bytes) must hold fewer than two context-sized charges \
         ({context_charge} bytes each) for the concurrency claim to be about capacity"
    );
    let prepared = QwenCuda::load_gguf(
        model,
        QwenLoadOptions {
            context_tokens: declared_bound,
            max_sequences: CONCURRENCY,
            continuation_capacity_bytes: Some(capacity),
            ..QwenLoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(prepared.memory_report().reserved_sequence_bytes, capacity);
    let mut engine = Engine::new(
        prepared,
        EngineConfig {
            max_active_requests: CONCURRENCY,
            max_queued_requests: 0,
            max_queued_input_tokens: u64::from(prompt_tokens) * CONCURRENCY as u64,
            max_buffered_events: CONCURRENCY * 4,
            max_events_per_request: 64,
        },
        SchedulePolicy::default(),
    )
    .unwrap();
    let requests = (0..CONCURRENCY)
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
    assert_eq!(
        engine.status().active_sequences,
        CONCURRENCY,
        "every request reaches less than its own context, so all of them hold \
         continuation at once"
    );
    let mut outputs = requests
        .iter()
        .map(|&request| (request, Vec::new()))
        .collect::<HashMap<_, _>>();
    let mut finished = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(600);
    while finished.len() < CONCURRENCY {
        assert!(Instant::now() < deadline, "Qwen qualification stalled");
        engine.step().unwrap();
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
        assert_eq!(finished[&request], FinishReason::Length);
        assert_eq!(outputs[&request], reference.output);
    }
    engine.shutdown().unwrap();
}
