//! Synthetic host-runtime benchmark. No model math, CUDA, or inference tok/s.
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use ribn::{
    Admission, BatchItem, Engine, EngineConfig, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, GenerationOptions, RequestId, SchedulePolicy, SequenceId, StepCompletion,
    SubmissionId, TokenRequest,
};

struct ImmediateModel {
    info: ExecutorInfo,
    prefixes: HashMap<SequenceId, u32>,
    pending: Option<Vec<StepCompletion>>,
    next_submission: u64,
}

impl GenerationExecutor for ImmediateModel {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }
    fn admit(
        &mut self,
        _: RequestId,
        id: SequenceId,
        _: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        self.prefixes.insert(id, 0);
        Ok(Admission::Ready)
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        assert!(self.pending.is_none());
        self.next_submission += 1;
        self.pending = Some(
            batch
                .iter()
                .map(|item| {
                    assert_eq!(self.prefixes[&item.sequence], item.prefix);
                    StepCompletion {
                        sequence: item.sequence,
                        prefix: item.prefix + item.token_budget,
                        tokens: vec![42; item.output_budget as usize],
                    }
                })
                .collect(),
        );
        Ok(SubmissionId::new(self.next_submission))
    }
    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        if let Some(rows) = &self.pending {
            for row in rows {
                self.prefixes.insert(row.sequence, row.prefix);
            }
        }
        Ok(self.pending.take())
    }
    fn release(&mut self, id: SequenceId) -> Result<(), ExecutionError> {
        assert!(self.pending.is_none());
        self.prefixes.remove(&id);
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.pending = None;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--driver") {
        return compare_driver();
    }
    println!(
        "# synthetic host dispatch; includes mock completion allocation; no model/device work"
    );
    println!(
        "concurrency,measured_iterations,median_ns_per_iteration,p99_ns_per_iteration,median_ns_per_row"
    );
    for concurrency in [1, 8, 32, 128] {
        measure(concurrency)?;
    }
    Ok(())
}

fn measure(concurrency: usize) -> Result<(), Box<dyn std::error::Error>> {
    let output_tokens = 10_256;
    let rows = u32::try_from(concurrency)?;
    let info = ExecutorInfo {
        name: "synthetic-host-only".into(),
        limits: GenerationLimits {
            context_tokens: output_tokens + 1,
            max_sequences: concurrency,
            max_batch_tokens: rows,
            max_decode_tokens: 1,
        },
    };
    let model = ImmediateModel {
        info,
        prefixes: HashMap::with_capacity(concurrency),
        pending: None,
        next_submission: 0,
    };
    let mut engine = Engine::new(
        model,
        EngineConfig {
            max_active_requests: concurrency,
            max_queued_requests: 0,
            max_queued_input_tokens: u64::from(rows),
            max_buffered_events: concurrency * 4,
            max_events_per_request: 64,
        },
        SchedulePolicy {
            max_batch_tokens: rows,
            ..SchedulePolicy::default()
        },
    )?;
    for _ in 0..concurrency {
        engine.enqueue(TokenRequest::new(
            vec![1],
            GenerationOptions {
                max_output_tokens: output_tokens,
                ..GenerationOptions::default()
            },
        ))?;
    }
    // Exclude admission, warmup, and terminal teardown from steady-state samples.
    for _ in 0..128 {
        tick(&mut engine)?;
    }
    let mut samples = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let started = Instant::now();
        tick(&mut engine)?;
        samples.push(started.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    println!(
        "{concurrency},{},{median},{},{}",
        samples.len(),
        samples[samples.len() * 99 / 100],
        median / concurrency as u128
    );
    engine.shutdown()?;
    Ok(())
}

fn compare_driver() -> Result<(), Box<dyn std::error::Error>> {
    use ribn::Event;
    use ribn::driver::{Driver, DriverConfig};

    const OUTPUT: u32 = 512;
    println!("# end-to-end synthetic delivery; spawn/load/shutdown excluded; no model math");
    println!("repeat,mode,concurrency,output_tokens_per_request,elapsed_ns,ns_per_output");
    for concurrency in [1, 8, 32] {
        for repeat in 0..5 {
            // Alternate the measured order rather than always running direct first.
            for owned in if repeat % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let rows = u32::try_from(concurrency)?;
                let mut engine = delivery_engine(concurrency, OUTPUT)?;
                let request = || {
                    TokenRequest::new(
                        vec![1],
                        GenerationOptions {
                            max_output_tokens: OUTPUT,
                            ..GenerationOptions::default()
                        },
                    )
                };
                let mut tokens = 0_u64;
                let mut finished = 0;
                let elapsed = if owned {
                    let (mut owner, handle) = Driver::spawn(
                        engine,
                        DriverConfig {
                            max_requests: concurrency,
                            ..DriverConfig::default()
                        },
                    )?;
                    let started = Instant::now();
                    let mut streams = (0..concurrency)
                        .map(|_| handle.stream_blocking(request()))
                        .collect::<Result<Vec<_>, _>>()?;
                    while finished < concurrency {
                        for stream in &mut streams {
                            if let Some(event) = stream.next_blocking() {
                                match black_box(event?) {
                                    Event::Token { .. } => tokens += 1,
                                    Event::Finished { .. } => finished += 1,
                                }
                            }
                        }
                    }
                    let elapsed = started.elapsed().as_nanos();
                    owner.shutdown()?;
                    elapsed
                } else {
                    let started = Instant::now();
                    for _ in 0..concurrency {
                        engine.enqueue(request())?;
                    }
                    while finished < concurrency {
                        engine.step()?;
                        while let Some(event) = engine.pop_event() {
                            match black_box(event) {
                                Event::Token { .. } => tokens += 1,
                                Event::Finished { .. } => finished += 1,
                            }
                        }
                    }
                    let elapsed = started.elapsed().as_nanos();
                    engine.shutdown()?;
                    elapsed
                };
                assert_eq!(tokens, u64::from(OUTPUT) * u64::from(rows));
                let mode = if owned { "driver" } else { "direct" };
                println!(
                    "{repeat},{mode},{concurrency},{OUTPUT},{elapsed},{}",
                    elapsed / u128::from(tokens)
                );
            }
        }
    }
    Ok(())
}

fn delivery_engine(concurrency: usize, output: u32) -> Result<Engine, Box<dyn std::error::Error>> {
    let rows = u32::try_from(concurrency)?;
    let model = ImmediateModel {
        info: ExecutorInfo {
            name: "synthetic-delivery".into(),
            limits: GenerationLimits {
                context_tokens: output + 1,
                max_sequences: concurrency,
                max_batch_tokens: rows,
                max_decode_tokens: 1,
            },
        },
        prefixes: HashMap::with_capacity(concurrency),
        pending: None,
        next_submission: 0,
    };
    Ok(Engine::new(
        model,
        EngineConfig {
            max_active_requests: concurrency,
            max_queued_requests: 0,
            max_queued_input_tokens: u64::from(rows),
            max_buffered_events: concurrency * 4,
            max_events_per_request: 4,
        },
        SchedulePolicy {
            max_batch_tokens: rows,
            ..SchedulePolicy::default()
        },
    )?)
}

fn tick(engine: &mut Engine) -> Result<(), ribn::EngineError> {
    black_box(engine.step()?);
    while let Some(event) = engine.pop_event() {
        black_box(event);
    }
    Ok(())
}
