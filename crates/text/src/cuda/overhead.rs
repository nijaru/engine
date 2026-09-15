//! Matched, single-request frontend cost over the same processor and CUDA engine.
//! This is a device experiment, not a general serving-throughput benchmark.

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine_gguf::GgufFile;
use engine_qwen::{QwenCuda, QwenLoadOptions};
use ribn::driver::{Driver, DriverConfig};
use ribn::{
    Engine, Event, FinishReason, GenerationExecutor, GenerationOptions, TokenRequest, Usage,
};

use crate::stream::Utf8Decoder;
use crate::{
    GgufProcessor, Message, ProcessorLimits, TextConfig, TextEvent, TextInput, TextOwner,
    TextProcessor,
};

#[derive(Debug, PartialEq)]
struct Output {
    tokens: Vec<u32>,
    text: String,
    reason: FinishReason,
    usage: Usage,
}

struct Sample {
    ttft: Duration,
    elapsed: Duration,
    output: Output,
}

fn input(case: usize) -> TextInput {
    match case {
        0 => TextInput::prompt("Name one primary color. Variant 0."),
        1 => TextInput::chat(vec![Message::user("Name one primary color.")]),
        2 => TextInput::prompt(format!(
            "{}\nSummarize the passage above in a paragraph:",
            "A serving engine batches requests, manages device memory, and streams results. "
                .repeat(24)
        )),
        _ => unreachable!(),
    }
}

fn options() -> GenerationOptions {
    GenerationOptions {
        max_output_tokens: 32,
        ..GenerationOptions::default()
    }
}

fn direct(engine: &mut Engine, processor: &dyn TextProcessor, case: usize) -> Sample {
    let start = Instant::now();
    let (tokens, stops) = processor.encode(input(case)).unwrap().into_parts();
    let mut options = options();
    options.stop_tokens = stops;
    engine.enqueue(TokenRequest::new(tokens, options)).unwrap();
    let mut tokens = Vec::new();
    let mut text = String::new();
    let mut bytes = Vec::new();
    let mut decoder = Utf8Decoder::default();
    let mut ttft = None;
    loop {
        assert!(
            start.elapsed() < Duration::from_secs(300),
            "direct request stalled"
        );
        engine.step().unwrap();
        while let Some(event) = engine.pop_event() {
            match event {
                Event::Token { token, .. } => {
                    processor.decode_token_into(token, &mut bytes).unwrap();
                    text.push_str(&decoder.push(token, &bytes).unwrap());
                    ttft.get_or_insert_with(|| start.elapsed());
                    tokens.push(token);
                }
                Event::Finished { reason, usage, .. } => {
                    text.push_str(&decoder.finish());
                    let elapsed = start.elapsed();
                    return Sample {
                        ttft: ttft.expect("workload must emit a token"),
                        elapsed,
                        output: Output {
                            tokens,
                            text,
                            reason,
                            usage,
                        },
                    };
                }
            }
        }
        // Direct embedding drives progress rather than sleeping on driver channels.
        std::thread::yield_now();
    }
}

fn handled(owner: &TextOwner, case: usize) -> Sample {
    let start = Instant::now();
    let stream = owner
        .model()
        .stream_blocking(input(case), options())
        .unwrap();
    let mut tokens = Vec::new();
    let mut text = String::new();
    let mut ttft = None;
    for event in stream {
        match event.unwrap() {
            TextEvent::Delta { token, text: delta } => {
                text.push_str(&delta);
                if let Some(token) = token {
                    ttft.get_or_insert_with(|| start.elapsed());
                    tokens.push(token);
                }
            }
            TextEvent::Finished { reason, usage } => {
                let elapsed = start.elapsed();
                return Sample {
                    ttft: ttft.expect("workload must emit a token"),
                    elapsed,
                    output: Output {
                        tokens,
                        text,
                        reason,
                        usage,
                    },
                };
            }
        }
    }
    panic!("missing text terminal");
}

#[test]
#[ignore = "requires an idle CUDA GPU and RIBN_MODEL; run release, serially, with --nocapture"]
fn matched_frontend_overhead() {
    let model = std::env::var("RIBN_MODEL").expect("RIBN_MODEL GGUF path");
    let mut references: Vec<Option<Output>> = (0..3).map(|_| None).collect();
    // Five independently loaded pairs, alternating mode order to expose drift.
    // Load/shutdown are excluded; each workload warms up once per loaded mode.
    for pair in 0..5 {
        for handled_mode in if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let processor: Arc<dyn TextProcessor> = Arc::new(GgufProcessor::new(
                GgufFile::open(&model).unwrap().tokenizer().unwrap(),
                ProcessorLimits {
                    max_prompt_tokens: 1024,
                    ..ProcessorLimits::default()
                },
            ));
            let executor = QwenCuda::load_gguf(
                &model,
                QwenLoadOptions {
                    context_tokens: 1024,
                    max_sequences: 1,
                    ..QwenLoadOptions::default()
                },
            )
            .unwrap();
            println!(
                "artifact={} pair={pair} handled={handled_mode}",
                executor.info().name
            );
            let mut engine = Engine::with_defaults(executor).unwrap();
            if handled_mode {
                let config = DriverConfig::for_engine(&engine);
                let (shutdown, handle) = Driver::spawn(engine, config).unwrap();
                let mut owner = TextOwner::new(
                    Arc::clone(&processor),
                    shutdown,
                    handle,
                    TextConfig::default(),
                )
                .unwrap();
                for (case, reference) in references.iter_mut().enumerate() {
                    check(reference, handled(&owner, case).output);
                    report(pair, "handle", case, handled(&owner, case), reference);
                }
                owner.shutdown().unwrap();
            } else {
                for (case, reference) in references.iter_mut().enumerate() {
                    check(
                        reference,
                        direct(&mut engine, processor.as_ref(), case).output,
                    );
                    report(
                        pair,
                        "direct",
                        case,
                        direct(&mut engine, processor.as_ref(), case),
                        reference,
                    );
                }
                engine.shutdown().unwrap();
            }
        }
    }
}

fn check(reference: &mut Option<Output>, output: Output) {
    assert!(matches!(
        output.reason,
        FinishReason::Stop | FinishReason::Length
    ));
    if let Some(reference) = reference {
        assert_eq!(*reference, output, "matched workload output changed");
    } else {
        *reference = Some(output);
    }
}

fn report(pair: usize, mode: &str, case: usize, sample: Sample, reference: &mut Option<Output>) {
    println!(
        "sample pair={pair} mode={mode} case={case} ttft_ms={:.3} elapsed_ms={:.3} prompt_tokens={} completion_tokens={} reason={:?}",
        sample.ttft.as_secs_f64() * 1000.0,
        sample.elapsed.as_secs_f64() * 1000.0,
        sample.output.usage.prompt_tokens,
        sample.output.usage.completion_tokens,
        sample.output.reason
    );
    check(reference, sample.output);
}
