//! Synthetic host-runtime benchmark. No model math, CUDA, or inference tok/s.
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use ribn::{
    Admission, BatchItem, Engine, EngineConfig, GenerationOptions, ModelError, ModelInfo,
    ModelLimits, PreparedModel, SchedulePolicy, SequenceId, StepCompletion, SubmissionId,
    TokenRequest,
};

struct ImmediateModel {
    info: ModelInfo,
    prefixes: HashMap<SequenceId, u32>,
    pending: Option<Vec<StepCompletion>>,
    next_submission: u64,
}

impl PreparedModel for ImmediateModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn admit(&mut self, id: SequenceId, _: &TokenRequest) -> Result<Admission, ModelError> {
        self.prefixes.insert(id, 0);
        Ok(Admission::Ready)
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ModelError> {
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
    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ModelError> {
        if let Some(rows) = &self.pending {
            for row in rows {
                self.prefixes.insert(row.sequence, row.prefix);
            }
        }
        Ok(self.pending.take())
    }
    fn release(&mut self, id: SequenceId) -> Result<(), ModelError> {
        assert!(self.pending.is_none());
        self.prefixes.remove(&id);
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), ModelError> {
        self.pending = None;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let info = ModelInfo {
        name: "synthetic-host-only".into(),
        limits: ModelLimits {
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

fn tick(engine: &mut Engine) -> Result<(), ribn::EngineError> {
    black_box(engine.step()?);
    while let Some(event) = engine.pop_event() {
        black_box(event);
    }
    Ok(())
}
