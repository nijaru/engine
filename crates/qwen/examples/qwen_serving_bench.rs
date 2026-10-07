//! Multi-request benchmark through `QwenCuda` and the production AR Engine.
//!
//! Compatible decode rows use backend-native batch lanes; mixed batches and
//! single rows stay per-row. This is a bounded timing/probe workload, not a
//! representative serving benchmark or device qualification. `--workload` adds
//! finite open-loop mixed arrivals over the same Engine (no HTTP/tokenization).
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! cargo run --release -p engine-qwen --features cuda --example qwen_serving_bench -- \
//!   --concurrency=4 --tokens=32 [--prompt-tokens=257] [--prefill-chunk=8]
//! ```

use std::collections::HashMap;
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

#[path = "serving_bench/options.rs"]
mod options;
#[path = "serving_bench/trace.rs"]
mod trace;
use options::Options;

enum RunMeasurements {
    Burst(Measurements),
    Trace(trace::Measurements),
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
    let max_prompt_tokens = options.workload.as_ref().map_or_else(
        || options.context_tokens - options.output_tokens,
        |workload| workload.max_prompt_tokens,
    );
    let request_capacity = options
        .concurrency
        .checked_add(options.queue_capacity)
        .ok_or("request capacity overflowed")?;
    let input_budget = u64::from(max_prompt_tokens)
        .checked_mul(u64::try_from(request_capacity).map_err(|_| "request capacity is too large")?)
        .ok_or_else(|| "aggregate input token budget overflowed".to_owned())?;
    let event_budget = options
        .concurrency
        .checked_add(options.queue_capacity)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| "aggregate event budget overflowed".to_owned())?;
    let batch_tokens = u32::try_from(options.concurrency)
        .ok()
        .and_then(|count| count.checked_mul(PREFILL_CHUNK_TOKENS))
        .ok_or_else(|| "batch token budget overflowed".to_owned())?;

    let load_started = Instant::now();
    let prepared = QwenCuda::load_gguf(
        &model_path,
        QwenLoadOptions {
            context_tokens: options.context_tokens,
            max_sequences: options.concurrency,
            gemv_mode: options.gemv_mode,
            grouped_decode: options.grouped_decode,
            // Burst probes preserve their serial baseline; trace traffic uses
            // the production prefill default unless explicitly set to off.
            prefill_chunk_members: options.prefill_chunk,
            weight_budget_bytes: Some(WEIGHT_BUDGET_BYTES),
            continuation_capacity_bytes: options.continuation_capacity_bytes,
            ..QwenLoadOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    let load_elapsed = load_started.elapsed();
    let memory = prepared.memory_report();
    let artifact = prepared.info().name.clone();
    let policy = SchedulePolicy {
        max_batch_tokens: batch_tokens.min(prepared.info().limits.max_batch_tokens),
        prefill_chunk_tokens: PREFILL_CHUNK_TOKENS,
        ..SchedulePolicy::default()
    };
    let mut engine = Engine::new(
        prepared,
        EngineConfig {
            max_active_requests: options.concurrency,
            max_queued_requests: options.queue_capacity,
            max_queued_input_tokens: input_budget,
            max_buffered_events: event_budget,
            max_events_per_request: 2,
        },
        policy,
    )
    .map_err(|error| error.to_string())?;
    // Include enqueue errors in the explicit shutdown path. A failed shutdown
    // leaves Engine owning retirement; its defensive Drop retains uncertain work.
    let measured = match &options.workload {
        Some(workload) => trace::measure(
            &mut engine,
            workload,
            &options.objectives,
            options.print_tokens,
        )
        .map(RunMeasurements::Trace),
        None => measure(&mut engine, &options).map(RunMeasurements::Burst),
    };
    let shutdown = engine
        .shutdown()
        .map_err(|error| format!("shutdown: {error}"));
    let measured = match (measured, shutdown) {
        (Ok(measured), Ok(())) => measured,
        (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(shutdown)) => return Err(format!("{error}; {shutdown}")),
    };
    match measured {
        RunMeasurements::Burst(measured) => report(
            &options,
            &measured,
            load_elapsed,
            memory.reserved_sequence_bytes,
            policy,
        ),
        RunMeasurements::Trace(measured) => report_trace(
            &options,
            &measured,
            &model_path,
            &artifact,
            load_elapsed,
            memory,
            policy,
        )?,
    }
    Ok(())
}

fn report_trace(
    options: &Options,
    measured: &trace::Measurements,
    model_path: &str,
    artifact: &str,
    load_elapsed: Duration,
    memory: engine_qwen::MemoryReport,
    policy: SchedulePolicy,
) -> Result<(), String> {
    measured.print();
    println!(
        "  continuation capacity: {} bytes (reservation bound, not measured peak)",
        memory.reserved_sequence_bytes
    );
    if options.print_tokens {
        for (index, request) in measured.requests.iter().enumerate() {
            println!(
                "  tokens[{index}]: {}",
                request
                    .tokens
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    if let Some(path) = &options.result_json {
        let output = std::fs::File::create_new(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        serde_json::to_writer_pretty(output, &serde_json::json!({
            "schema_version": 1, "surface": "direct_ar_engine_open_loop",
            "model": model_path, "artifact": artifact, "load_seconds": load_elapsed.as_secs_f64(),
            "max_active_requests": options.concurrency, "max_queued_requests": options.queue_capacity,
            "batch_token_budget": policy.max_batch_tokens, "prefill_token_budget": policy.prefill_chunk_tokens,
            "prefill_chunk_members": options.prefill_chunk, "gemv_mode": format!("{:?}", options.gemv_mode),
            "grouped_decode": options.grouped_decode,
            "continuation_capacity_bytes": memory.reserved_sequence_bytes,
            "free_before_preparation_bytes": memory.free_before_preparation_bytes,
            "free_after_preparation_bytes": memory.free_after_preparation_bytes,
            "objectives": options.objectives,
            "workload": options.workload.as_ref().expect("trace measurements").inputs,
            "measurements": measured,
        })).map_err(|error| format!("write {}: {error}", path.display()))?;
    }
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
            if request.tokens[0] == u32::MAX {
                return Err(ExecutionError::new("invalid fixture token"));
            }
            self.first_tokens.insert(sequence, request.tokens[0]);
            Ok(Admission::Ready)
        }
        fn prepare(&mut self, _: &[BatchItem]) -> Result<ribn::BatchPreparation, ExecutionError> {
            Ok(ribn::BatchPreparation::Ready)
        }
        fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
            Ok(())
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
        fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
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
        assert!(!defaults.grouped_decode);
        assert!(!QwenLoadOptions::default().grouped_decode);
        assert!(options(&["--grouped-decode"]).unwrap().grouped_decode);
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
    fn open_loop_counts_overload_and_local_failure_without_retrying() {
        let request = |prompt: &[u32], output_tokens| trace::ScheduledRequest {
            arrival: Duration::ZERO,
            prompt: std::sync::Arc::from(prompt),
            output_tokens,
        };
        let workload = trace::Workload {
            inputs: vec![],
            requests: vec![
                request(&[7, 8], 3),
                request(&[u32::MAX], 1),
                request(&[9], 2),
                request(&[10], 2),
            ],
            context_tokens: 5,
            max_prompt_tokens: 2,
        };
        let mut engine = probe_engine(2, 0);
        let measured =
            trace::measure(&mut engine, &workload, &trace::Objectives::default(), true).unwrap();
        assert_eq!(
            (
                measured.offered,
                measured.completed,
                measured.failed,
                measured.rejected
            ),
            (4, 1, 1, 2)
        );
        assert_eq!(measured.emitted_tokens, 3);
        assert_eq!(measured.requests[0].tokens, vec![7, 7, 7]);
        assert_eq!(measured.slo_completed, None);
        assert_eq!(engine.status().requests, 0);
        engine.shutdown().unwrap();

        // The sleeping path has no live work to race; only one future arrival.
        let mut request = request(&[9], 1);
        request.arrival = Duration::from_millis(2);
        let future = trace::Workload {
            requests: vec![request],
            ..workload
        };
        let mut engine = probe_engine(1, 0);
        let measured =
            trace::measure(&mut engine, &future, &trace::Objectives::default(), false).unwrap();
        assert_eq!(measured.completed, 1);
        assert!(measured.elapsed_s >= 0.002);
        engine.shutdown().unwrap();
    }

    fn probe_engine(active: usize, queued: usize) -> Engine {
        Engine::new(
            ProbeExecutor {
                info: ExecutorInfo {
                    name: "host trace".to_owned(),
                    limits: GenerationLimits {
                        context_tokens: 64,
                        max_sequences: active,
                        max_batch_tokens: 128,
                        max_decode_tokens: 1,
                    },
                },
                first_tokens: HashMap::new(),
                pending: None,
                polled: false,
            },
            EngineConfig {
                max_active_requests: active,
                max_queued_requests: queued,
                max_queued_input_tokens: 128,
                max_buffered_events: 16,
                max_events_per_request: 2,
            },
            SchedulePolicy::default(),
        )
        .unwrap()
    }

    #[test]
    fn checked_in_mixed_trace_has_valid_reach_and_explicit_kernel_overrides() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/workloads/qwen-mixed-arrivals.json"
        );
        let flag = format!("--workload={path}");
        let parsed = options(&[
            &flag,
            "--queue-capacity=4",
            "--ttft-slo-ms=2000",
            "--continuation-capacity-bytes=524288000",
        ])
        .unwrap();
        assert_eq!(parsed.context_tokens, 289);
        assert_eq!(parsed.queue_capacity, 4);
        assert_eq!(
            parsed.prefill_chunk,
            Some(engine_qwen::DEFAULT_PREFILL_CHUNK_MEMBERS)
        );
        assert_eq!(parsed.gemv_mode, GemvMode::Warp);
        let workload = parsed.workload.unwrap();
        assert_eq!(workload.requests.len(), 8);
        assert_eq!(workload.max_prompt_tokens, 257);
        assert!(std::sync::Arc::ptr_eq(
            &workload.requests[0].prompt,
            &workload.requests[3].prompt
        ));
        assert!(options(&[&flag, "--tokens=8"]).is_err());
        assert!(options(&[&flag, "--divergence-probe", "--print-tokens"]).is_err());
        assert_eq!(
            options(&[&flag, "--prefill-chunk=off"])
                .unwrap()
                .prefill_chunk,
            None
        );
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
            "--queue-capacity=1",
            "--ttft-slo-ms=10",
            "--unknown=1",
            "--continuation-capacity-bytes=0",
        ] {
            assert!(options(&[flag]).is_err(), "{flag}");
        }
        assert!(options(&["--tokens=3", "--tokens=4"]).is_err());
        assert!(
            options(&[
                "--divergence-probe",
                "--print-tokens",
                "--prompt-tokens=4294967296"
            ])
            .is_err()
        );
        assert!(
            options(&[
                "--divergence-probe",
                "--print-tokens",
                "--prompt-fixture=missing"
            ])
            .is_err()
        );
    }
}
