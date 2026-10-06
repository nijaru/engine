//! Multi-request benchmark through `QwenCuda` and the production AR Engine.
//!
//! Compatible decode rows use backend-native batch lanes; mixed batches and
//! single rows stay per-row. This is a bounded timing/probe workload, not a
//! representative mixed-arrival serving benchmark or device qualification.
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! cargo run --release -p engine-qwen --features cuda --example qwen_serving_bench -- \
//!   --concurrency=4 --tokens=32 [--prompt-tokens=257] [--prefill-chunk=8]
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engine_nvidia::GemvMode;
use engine_qwen::{QwenCuda, QwenLoadOptions};
use ribn::{
    Engine, EngineConfig, Event, FinishReason, GenerationExecutor, GenerationOptions, RequestId,
    SchedulePolicy, TokenRequest,
};

const PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
/// Diverse fixed prompts for the `--divergence-probe` quality gate. Each
/// exercises different token distributions and attention/GDN mixing so the
/// greedy argmax sees varied logit-gap profiles; one is deliberately
/// near-tie-prone (repeated structure) to expose divergence worst cases.
const PROBE_PROMPTS: [&[u32]; 6] = [
    &[760, 6511, 314, 9338, 369],
    &[15496, 11, 1938, 389, 1015, 264, 7566, 286, 3303, 13],
    &[9311, 29892, 3300, 264, 2361, 29892, 1660, 470, 389, 304, 13],
    &[3210, 1490, 46749, 29468, 3925, 284, 1660, 470, 304, 286, 13],
    &[290, 428, 1838, 373, 2584, 264, 6144, 286, 3303, 30, 30],
    &[3730, 3730, 3730, 3730, 11, 373, 25, 3303, 13],
];
const DEFAULT_CONCURRENCY: usize = 4;
const DEFAULT_OUTPUT_TOKENS: u32 = 32;
const PREFILL_CHUNK_TOKENS: u32 = 16;
const WEIGHT_BUDGET_BYTES: u64 = 20_u64 << 30;

struct Options {
    concurrency: usize,
    output_tokens: u32,
    prompt: Arc<[u32]>,
    fixture_prompt: bool,
    gemv_mode: GemvMode,
    prefill_chunk: Option<usize>,
    print_tokens: bool,
    divergence_probe: bool,
    context_tokens: u32,
}

impl Options {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let concurrency = parse_usize(arguments, "--concurrency=", DEFAULT_CONCURRENCY)?;
        let output_tokens = parse_u32(arguments, "--tokens=", DEFAULT_OUTPUT_TOKENS)?;
        if concurrency == 0 || output_tokens == 0 {
            return Err("concurrency and token count must be greater than zero".to_owned());
        }
        let fixture_tokens = arguments
            .iter()
            .find_map(|argument| argument.strip_prefix("--prompt-fixture="))
            .map(read_fixture_prompt)
            .transpose()?;
        let prompt_tokens = parse_usize(
            arguments,
            "--prompt-tokens=",
            fixture_tokens.as_ref().map_or(PROMPT.len(), Vec::len),
        )?;
        if prompt_tokens == 0 {
            return Err("prompt token count must be greater than zero".to_owned());
        }
        let gemv_mode = parse_gemv_mode(arguments)?;
        let prefill_chunk = arguments
            .iter()
            .find_map(|argument| argument.strip_prefix("--prefill-chunk="))
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| format!("--prefill-chunk expects a number, got {value}"))
            })
            .transpose()?;
        if prefill_chunk == Some(0) {
            return Err("--prefill-chunk must be greater than zero".to_owned());
        }
        let print_tokens = arguments
            .iter()
            .any(|argument| argument == "--print-tokens");
        let divergence_probe = arguments
            .iter()
            .any(|argument| argument == "--divergence-probe");
        if divergence_probe && !print_tokens {
            return Err(
                "--divergence-probe requires --print-tokens to emit comparable token streams"
                    .to_owned(),
            );
        }
        let longest_prompt = if divergence_probe {
            PROBE_PROMPTS
                .iter()
                .map(|prompt| prompt.len())
                .max()
                .expect("six prompts")
        } else {
            prompt_tokens
        };
        let context_tokens = u32::try_from(longest_prompt)
            .map_err(|_| "prompt length does not fit the runtime".to_owned())?
            .checked_add(output_tokens)
            .ok_or_else(|| "prompt plus output budget overflowed".to_owned())?;
        // Without a fixture, repeat the fixed prompt. This is a timing fixture,
        // not real prompt content. Fixture lengths always select a prefix.
        let prompt = match &fixture_tokens {
            Some(tokens) => {
                if tokens.len() < prompt_tokens {
                    return Err(format!(
                        "prompt fixture holds {} tokens, fewer than the requested {prompt_tokens}",
                        tokens.len()
                    ));
                }
                Arc::from(&tokens[..prompt_tokens])
            }
            None => Arc::from(
                PROMPT
                    .iter()
                    .copied()
                    .cycle()
                    .take(prompt_tokens)
                    .collect::<Vec<_>>(),
            ),
        };
        Ok(Self {
            concurrency,
            output_tokens,
            prompt,
            fixture_prompt: fixture_tokens.is_some(),
            gemv_mode,
            prefill_chunk,
            print_tokens,
            divergence_probe,
            context_tokens,
        })
    }

    fn prompt_for(&self, index: usize) -> Arc<[u32]> {
        if self.divergence_probe {
            Arc::from(PROBE_PROMPTS[index % PROBE_PROMPTS.len()])
        } else {
            Arc::clone(&self.prompt)
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen_serving_bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let options = Options::parse(&arguments)?;
    let model_path = arguments
        .iter()
        .find_map(|argument| argument.strip_prefix("--model=").map(str::to_owned))
        .or_else(|| std::env::var("ENGINE_QWEN_GGUF").ok())
        .ok_or_else(|| "set ENGINE_QWEN_GGUF or pass --model=/path/to/model.gguf".to_owned())?;
    let input_budget = u64::from(options.context_tokens - options.output_tokens)
        .checked_mul(u64::try_from(options.concurrency).map_err(|_| "concurrency is too large")?)
        .ok_or_else(|| "aggregate input token budget overflowed".to_owned())?;
    let event_budget = options
        .concurrency
        .checked_mul(2)
        .ok_or_else(|| "aggregate event budget overflowed".to_owned())?;
    let batch_tokens = u32::try_from(options.concurrency)
        .ok()
        .and_then(|count| count.checked_mul(PREFILL_CHUNK_TOKENS))
        .ok_or_else(|| "batch token budget overflowed".to_owned())?;

    let load_started = Instant::now();
    let prepared = QwenCuda::load_gguf(
        model_path,
        QwenLoadOptions {
            context_tokens: options.context_tokens,
            max_sequences: options.concurrency,
            gemv_mode: options.gemv_mode,
            // Preserve the benchmark's serial default, independently of the
            // production loader's enabled-by-default same-sequence lane.
            prefill_chunk_members: options.prefill_chunk,
            weight_budget_bytes: Some(WEIGHT_BUDGET_BYTES),
            ..QwenLoadOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    let load_elapsed = load_started.elapsed();
    let memory = prepared.memory_report();
    let policy = SchedulePolicy {
        max_batch_tokens: batch_tokens.min(prepared.info().limits.max_batch_tokens),
        prefill_chunk_tokens: PREFILL_CHUNK_TOKENS,
        ..SchedulePolicy::default()
    };
    let mut engine = Engine::new(
        prepared,
        EngineConfig {
            max_active_requests: options.concurrency,
            max_queued_requests: 0,
            max_queued_input_tokens: input_budget,
            max_buffered_events: event_budget,
            max_events_per_request: 2,
        },
        policy,
    )
    .map_err(|error| error.to_string())?;
    // Include enqueue errors in the explicit shutdown path. A failed shutdown
    // leaves Engine owning retirement; its defensive Drop retains uncertain work.
    let measured = measure(&mut engine, &options);
    let shutdown = engine
        .shutdown()
        .map_err(|error| format!("shutdown: {error}"));
    let measured = match (measured, shutdown) {
        (Ok(measured), Ok(())) => measured,
        (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(shutdown)) => return Err(format!("{error}; {shutdown}")),
    };
    report(
        &options,
        &measured,
        load_elapsed,
        memory.reserved_sequence_bytes,
        policy,
    );
    Ok(())
}

struct Measurements {
    elapsed: Duration,
    first_token: Vec<Duration>,
    inter_token: Vec<Duration>,
    generated_tokens: u32,
    token_log: Vec<Vec<u32>>,
}

fn measure(engine: &mut Engine, options: &Options) -> Result<Measurements, String> {
    let mut requests = HashMap::<RequestId, usize>::with_capacity(options.concurrency);
    for index in 0..options.concurrency {
        let request = engine
            .enqueue(TokenRequest::new(
                options.prompt_for(index),
                GenerationOptions {
                    max_output_tokens: options.output_tokens,
                    // Greedy, no stop tokens: probes and timing runs must emit the
                    // entire requested continuation, even if an EOS ID is generated.
                    ..GenerationOptions::default()
                },
            ))
            .map_err(|error| error.to_string())?;
        requests.insert(request, index);
    }
    // Exclude enqueue. Engine progress now owns timed admission and logical
    // continuation allocation, plus prefill, decode and host-observed delivery.
    let started = Instant::now();
    let mut first_token = vec![None; options.concurrency];
    let mut previous_token = vec![None; options.concurrency];
    let mut token_log = vec![Vec::new(); options.concurrency];
    let mut inter_token = Vec::new();
    let mut generated_tokens = 0_u32;
    let mut remaining = options.concurrency;
    while remaining > 0 {
        let progress = engine.step().map_err(|error| error.to_string())?;
        while let Some(event) = engine.pop_event() {
            let index = *requests
                .get(&event.request())
                .ok_or_else(|| "generated output belongs to an unknown request".to_owned())?;
            match event {
                Event::Token { token, .. } => {
                    let now = started.elapsed();
                    if options.print_tokens {
                        token_log[index].push(token);
                    }
                    first_token[index].get_or_insert(now);
                    if let Some(previous) = previous_token[index].replace(now) {
                        inter_token.push(now.saturating_sub(previous));
                    }
                    generated_tokens = generated_tokens
                        .checked_add(1)
                        .ok_or_else(|| "generated-token counter overflowed".to_owned())?;
                }
                Event::Finished { reason, usage, .. } => {
                    if reason != FinishReason::Length
                        || usage.completion_tokens != options.output_tokens
                    {
                        return Err(format!(
                            "request {index} ended with {reason:?}, usage {usage:?}"
                        ));
                    }
                    remaining -= 1;
                }
            }
        }
        if remaining > 0 && !progress.completed && !progress.submitted {
            if !engine.status().in_flight {
                // This workload reserves enough continuation for every request
                // and drains all events; resource waiting here is a defect, not
                // a reason to introduce another readiness/scheduling mechanism.
                return Err("benchmark stalled without runnable or in-flight work".to_owned());
            }
            std::thread::yield_now();
        }
    }
    let elapsed = started.elapsed();
    let expected_tokens = u32::try_from(options.concurrency)
        .ok()
        .and_then(|count| count.checked_mul(options.output_tokens))
        .ok_or_else(|| "expected-token count exceeds benchmark reporting range".to_owned())?;
    if generated_tokens != expected_tokens {
        return Err(format!(
            "AR engine emitted {generated_tokens} tokens, expected {expected_tokens}"
        ));
    }
    Ok(Measurements {
        elapsed,
        first_token: first_token
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "at least one request never produced a first token".to_owned())?,
        inter_token,
        generated_tokens,
        token_log,
    })
}

fn report(
    options: &Options,
    measured: &Measurements,
    load_elapsed: Duration,
    state_capacity: u64,
    policy: SchedulePolicy,
) {
    let (ttft_mean, ttft_max) = duration_mean_max(&measured.first_token);
    let (itl_mean, itl_max) = duration_mean_max(&measured.inter_token);
    println!("Qwen3.8 ribn::Engine baseline");
    println!("  concurrency: {}", options.concurrency);
    println!("  output tokens/request: {}", options.output_tokens);
    if options.divergence_probe {
        println!("  prompt tokens/request: varied (six fixed divergence probes, cycled by slot)");
        println!("  prompt source: divergence probes");
    } else {
        println!("  prompt tokens/request: {}", options.prompt.len());
        println!(
            "  prompt source: {}",
            if options.fixture_prompt {
                "reference fixture prompt"
            } else {
                "repeated fixed prompt (timing fixture)"
            }
        );
    }
    println!(
        "  same-sequence prefill chunking: {}",
        match options.prefill_chunk {
            Some(members) => format!("chunks of {members} tokens"),
            None => "off (serial prefill)".to_owned(),
        }
    );
    println!(
        "  prepared model (load/staging/kernels): {:.2} s",
        load_elapsed.as_secs_f64()
    );
    println!("  request-state capacity bytes: {state_capacity} aggregate");
    println!(
        "  scheduler batch token budget: {}",
        policy.max_batch_tokens
    );
    println!("  elapsed: {:.3} s", measured.elapsed.as_secs_f64());
    println!(
        "  aggregate throughput: {:.2} tok/s",
        f64::from(measured.generated_tokens) / measured.elapsed.as_secs_f64()
    );
    println!(
        "  observed TTFT: mean {:.3} s, max {:.3} s",
        ttft_mean.as_secs_f64(),
        ttft_max.as_secs_f64()
    );
    if !measured.inter_token.is_empty() {
        println!(
            "  observed ITL: mean {:.3} ms, max {:.3} ms",
            itl_mean.as_secs_f64() * 1000.0,
            itl_max.as_secs_f64() * 1000.0
        );
    }
    println!(
        "  caveat: compatible multi-row decode uses backend batch lanes; mixed batches and single rows stay per-row; timings include host scheduling and event observation, not device-only latency"
    );
    println!(
        "  gemv mode: {}",
        match options.gemv_mode {
            GemvMode::Scalar => "scalar (correctness oracle, non-default)",
            GemvMode::Warp => "warp-cooperative (default)",
            GemvMode::IntegerDot =>
                "integer-dot (experimental lossy, float fallback for unsupported families)",
        }
    );
    if options.print_tokens {
        for (index, tokens) in measured.token_log.iter().enumerate() {
            let line = tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ");
            println!("  tokens[{index}]: {line}");
        }
    }
}

/// Read the fixture format used in `crates/qwen/tests/fixtures`: artifact
/// identity, prompt token IDs, then reference continuation (unused for timing).
fn read_fixture_prompt(path: &str) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read prompt fixture {path}: {error}"))?;
    let mut lines = text.lines();
    lines
        .next()
        .ok_or_else(|| format!("prompt fixture {path} has no identity line"))?;
    lines
        .next()
        .ok_or_else(|| format!("prompt fixture {path} has no prompt line"))?
        .split_whitespace()
        .map(|token| {
            token
                .parse::<u32>()
                .map_err(|error| format!("prompt fixture {path} token: {error}"))
        })
        .collect()
}

fn parse_usize(arguments: &[String], prefix: &str, default: usize) -> Result<usize, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix(prefix))
        .map_or(Ok(default), |value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{prefix} expects an integer"))
        })
}

fn parse_u32(arguments: &[String], prefix: &str, default: u32) -> Result<u32, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix(prefix))
        .map_or(Ok(default), |value| {
            value
                .parse::<u32>()
                .map_err(|_| format!("{prefix} expects an integer"))
        })
}

fn parse_gemv_mode(arguments: &[String]) -> Result<GemvMode, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix("--gemv="))
        .map_or(Ok(GemvMode::default()), |value| match value {
            "scalar" => Ok(GemvMode::Scalar),
            "warp" => Ok(GemvMode::Warp),
            "int-dot" => Ok(GemvMode::IntegerDot),
            other => Err(format!(
                "--gemv expects scalar, warp, or int-dot, got {other}"
            )),
        })
}

fn duration_mean_max(values: &[Duration]) -> (Duration, Duration) {
    if values.is_empty() {
        return (Duration::ZERO, Duration::ZERO);
    }
    let total = values.iter().map(Duration::as_secs_f64).sum::<f64>();
    let count = u32::try_from(values.len()).expect("benchmark sample count fits u32");
    (
        Duration::from_secs_f64(total / f64::from(count)),
        values.iter().copied().max().unwrap_or(Duration::ZERO),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(arguments: &[&str]) -> Result<Options, String> {
        Options::parse(
            &arguments
                .iter()
                .map(|argument| (*argument).to_owned())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn defaults_and_explicit_kernel_modes_preserve_benchmark_policy() {
        let defaults = options(&[]).unwrap();
        assert_eq!(defaults.concurrency, 4);
        assert_eq!(defaults.output_tokens, 32);
        assert_eq!(&*defaults.prompt, &PROMPT);
        assert_eq!(defaults.context_tokens, 37);
        assert_eq!(defaults.prefill_chunk, None);
        assert_eq!(defaults.gemv_mode, GemvMode::Warp);
        assert_eq!(QwenLoadOptions::default().gemv_mode, GemvMode::Warp);
        for (flag, expected) in [
            ("--gemv=scalar", GemvMode::Scalar),
            ("--gemv=warp", GemvMode::Warp),
            ("--gemv=int-dot", GemvMode::IntegerDot),
        ] {
            let parsed = options(&[
                flag,
                "--prefill-chunk=8",
                "--concurrency=6",
                "--tokens=3",
                "--print-tokens",
            ])
            .unwrap();
            assert_eq!(parsed.gemv_mode, expected);
            assert_eq!(parsed.prefill_chunk, Some(8));
            assert_eq!(parsed.concurrency, 6);
            assert_eq!(parsed.output_tokens, 3);
            assert!(parsed.print_tokens);
        }
    }

    #[test]
    fn probes_cycle_all_six_histories_and_size_for_longest() {
        let parsed = options(&["--divergence-probe", "--print-tokens", "--concurrency=6"]).unwrap();
        for (index, prompt) in PROBE_PROMPTS.iter().enumerate() {
            assert_eq!(&*parsed.prompt_for(index), *prompt);
        }
        assert_eq!(&*parsed.prompt_for(6), PROBE_PROMPTS[0]);
        assert_eq!(parsed.context_tokens, 43);
    }

    #[test]
    fn fixture_defaults_to_whole_prompt_and_explicit_length_selects_prefix() {
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen38-code-fill4096-257.tokens"
        );
        let flag = format!("--prompt-fixture={fixture}");
        let whole = options(&[&flag]).unwrap();
        assert!(whole.fixture_prompt);
        assert_eq!(whole.prompt.len(), 257);
        let prefix = options(&[&flag, "--prompt-tokens=17"]).unwrap();
        assert_eq!(&*prefix.prompt, &whole.prompt[..17]);
        assert!(options(&[&flag, "--prompt-tokens=258"]).is_err());
        let repeated = options(&["--prompt-tokens=7"]).unwrap();
        assert_eq!(&*repeated.prompt, &[760, 6511, 314, 9338, 369, 760, 6511]);
    }

    #[test]
    fn measurement_drains_real_engine_events_into_probe_slot_order() {
        use ribn::{
            Admission, BatchItem, ExecutionError, ExecutorInfo, GenerationLimits, SequenceId,
            StepCompletion, SubmissionId,
        };

        // Substitute only device execution; exercise the real scheduling,
        // globally allocated request IDs, event credits and terminal delivery.
        struct ProbeExecutor {
            info: ExecutorInfo,
            first_tokens: HashMap<SequenceId, u32>,
            pending: Option<Vec<StepCompletion>>,
            polled: bool,
        }
        impl GenerationExecutor for ProbeExecutor {
            fn info(&self) -> &ExecutorInfo {
                &self.info
            }
            fn admit(
                &mut self,
                _: RequestId,
                sequence: SequenceId,
                request: &TokenRequest,
            ) -> Result<Admission, ExecutionError> {
                self.first_tokens.insert(sequence, request.tokens[0]);
                Ok(Admission::Ready)
            }
            fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
                self.pending = Some(
                    batch
                        .iter()
                        .map(|item| StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + item.token_budget,
                            tokens: vec![
                                self.first_tokens[&item.sequence];
                                item.output_budget as usize
                            ],
                        })
                        .collect(),
                );
                self.polled = false;
                Ok(SubmissionId::new(1))
            }
            fn poll(
                &mut self,
                _: SubmissionId,
            ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
                if !self.polled {
                    self.polled = true;
                    return Ok(None);
                }
                Ok(self.pending.take())
            }
            fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
                self.first_tokens.remove(&sequence);
                Ok(())
            }
            fn synchronize(&mut self) -> Result<(), ExecutionError> {
                self.pending = None;
                Ok(())
            }
        }
        let options = options(&[
            "--divergence-probe",
            "--print-tokens",
            "--concurrency=6",
            "--tokens=3",
        ])
        .unwrap();
        let mut engine = Engine::new(
            ProbeExecutor {
                info: ExecutorInfo {
                    name: "host probe".to_owned(),
                    limits: GenerationLimits {
                        context_tokens: options.context_tokens,
                        max_sequences: 6,
                        max_batch_tokens: 128,
                        max_decode_tokens: 1,
                    },
                },
                first_tokens: HashMap::new(),
                pending: None,
                polled: false,
            },
            EngineConfig {
                max_active_requests: 6,
                max_queued_requests: 0,
                max_queued_input_tokens: 66,
                max_buffered_events: 12,
                max_events_per_request: 2,
            },
            SchedulePolicy::default(),
        )
        .unwrap();
        let measured = measure(&mut engine, &options).unwrap();
        engine.shutdown().unwrap();
        assert_eq!(measured.generated_tokens, 18);
        assert_eq!(measured.first_token.len(), 6);
        assert_eq!(measured.inter_token.len(), 12);
        for (index, tokens) in measured.token_log.iter().enumerate() {
            assert_eq!(tokens, &vec![PROBE_PROMPTS[index][0]; 3]);
        }
        assert_eq!(engine.status().requests, 0);
    }

    #[test]
    fn invalid_options_fail_before_device_loading() {
        for flag in [
            "--concurrency=0",
            "--tokens=0",
            "--prompt-tokens=0",
            "--prefill-chunk=0",
            "--gemv=unknown",
            "--tokens=no",
            "--divergence-probe",
            "--tokens=4294967295",
        ] {
            assert!(options(&[flag]).is_err(), "{flag}");
        }
    }
}
