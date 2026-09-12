use std::error::Error;
use std::fmt;

use ribn_batch::{BatchConfig, BatchExecutor, BatchRuntime, Job, JobOutput};
use ribn_foundation::ParameterVersion;

#[derive(Clone, Debug, Eq, PartialEq)]
enum EncoderError {
    EmptyInput,
    InputTooLong,
    TokenOutOfRange(u32),
    BatchTooLarge { tokens: usize, limit: usize },
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("encoder input must not be empty"),
            Self::InputTooLong => {
                f.write_str("encoder input is too long for the reference fixture")
            }
            Self::TokenOutOfRange(token) => write!(f, "encoder token {token} is out of range"),
            Self::BatchTooLarge { tokens, limit } => {
                write!(
                    f,
                    "encoder batch has {tokens} tokens but its limit is {limit}"
                )
            }
        }
    }
}

impl Error for EncoderError {}

/// Tiny reference encoder: embedding lookup followed by mean pooling.
///
/// This is deliberately not a production model API. It is large enough to make
/// variable input shapes, vector outputs, parameter versions, and model-specific
/// batch memory/cost constraints concrete without importing AR semantics.
struct ReferenceEncoder {
    version: ParameterVersion,
    embeddings: Vec<[f32; 3]>,
    max_items: usize,
    max_batch_tokens: usize,
}

impl ReferenceEncoder {
    fn fixture(max_batch_tokens: usize) -> Self {
        Self {
            version: ParameterVersion::new(11),
            embeddings: vec![
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            max_items: 2,
            max_batch_tokens,
        }
    }

    fn encode(&self, tokens: &[u32]) -> Result<[f32; 3], EncoderError> {
        if tokens.is_empty() {
            return Err(EncoderError::EmptyInput);
        }
        let mut output = [0.0_f32; 3];
        for &token in tokens {
            let index = usize::try_from(token).map_err(|_| EncoderError::TokenOutOfRange(token))?;
            let row = self
                .embeddings
                .get(index)
                .ok_or(EncoderError::TokenOutOfRange(token))?;
            for (destination, value) in output.iter_mut().zip(row) {
                *destination += value;
            }
        }
        let length = u16::try_from(tokens.len()).map_err(|_| EncoderError::InputTooLong)?;
        let scale = 1.0 / f32::from(length);
        for value in &mut output {
            *value *= scale;
        }
        Ok(output)
    }
}

impl BatchExecutor for ReferenceEncoder {
    type Input = Vec<u32>;
    type Output = [f32; 3];
    type Error = EncoderError;

    fn parameter_version(&self) -> ParameterVersion {
        self.version
    }

    fn max_batch_items(&self) -> usize {
        self.max_items
    }

    fn select_batch(&self, candidates: &[&Self::Input]) -> usize {
        let mut selected = 0;
        let mut tokens = 0_usize;
        for candidate in candidates {
            let next = tokens.saturating_add(candidate.len());
            if selected > 0 && next > self.max_batch_tokens {
                break;
            }
            tokens = next;
            selected += 1;
        }
        selected
    }

    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
        let batch_tokens = batch.iter().map(|job| job.input().len()).sum::<usize>();
        if batch_tokens > self.max_batch_tokens {
            return Err(EncoderError::BatchTooLarge {
                tokens: batch_tokens,
                limit: self.max_batch_tokens,
            });
        }
        batch
            .into_iter()
            .map(|job| {
                let request = job.request();
                let output = self.encode(job.input())?;
                Ok(JobOutput::new(request, output))
            })
            .collect()
    }
}

fn assert_close(actual: [f32; 3], expected: [f32; 3]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn variable_length_encoder_inputs_batch_without_ar_semantics() {
    let mut runtime = BatchRuntime::new(
        ReferenceEncoder::fixture(8),
        BatchConfig {
            max_queued_requests: 8,
        },
    )
    .expect("runtime");

    let first = runtime.submit(vec![0, 1]).expect("first request");
    let second = runtime.submit(vec![2, 3, 0]).expect("second request");
    assert!(runtime.step().expect("encoder batch"));

    let first_output = runtime.pop_completed().expect("first output");
    let second_output = runtime.pop_completed().expect("second output");
    assert_eq!(first_output.request(), first);
    assert_eq!(second_output.request(), second);
    assert_eq!(first_output.parameter_version(), ParameterVersion::new(11));
    assert_eq!(second_output.parameter_version(), ParameterVersion::new(11));
    assert_close(*first_output.output(), [0.5, 0.5, 0.0]);
    assert_close(*second_output.output(), [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]);
}

#[test]
fn executor_informed_selection_respects_encoder_batch_cost() {
    let mut runtime = BatchRuntime::new(
        ReferenceEncoder::fixture(4),
        BatchConfig {
            max_queued_requests: 8,
        },
    )
    .expect("runtime");

    let first = runtime.submit(vec![0, 1, 2]).expect("first request");
    let second = runtime.submit(vec![3, 2, 1]).expect("second request");

    assert!(runtime.step().expect("first encoder batch"));
    assert_eq!(runtime.queued(), 1);
    assert_eq!(
        runtime.pop_completed().expect("first output").request(),
        first
    );

    assert!(runtime.step().expect("second encoder batch"));
    assert_eq!(runtime.queued(), 0);
    assert_eq!(
        runtime.pop_completed().expect("second output").request(),
        second
    );
}
