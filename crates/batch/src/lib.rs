//! Non-autoregressive dynamic-batch runtime used to pressure-test Ribn's common boundaries.
//!
//! Inputs and outputs are executor-defined. The runtime has no token, prefix, KV,
//! decode, or chat semantics. Its batching policy is intentionally minimal until
//! real encoder/pooling models provide shape and memory evidence.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use ribn_foundation::ParameterVersion;

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchConfig {
    pub max_queued_requests: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_queued_requests: 1024,
        }
    }
}

pub struct Job<I> {
    request: RequestId,
    input: I,
}

impl<I> Job<I> {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub fn input(&self) -> &I {
        &self.input
    }

    #[must_use]
    pub fn into_input(self) -> I {
        self.input
    }
}

pub struct JobOutput<O> {
    request: RequestId,
    output: O,
}

impl<O> JobOutput<O> {
    #[must_use]
    pub fn new(request: RequestId, output: O) -> Self {
        Self { request, output }
    }

    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub fn output(&self) -> &O {
        &self.output
    }

    #[must_use]
    pub fn into_output(self) -> O {
        self.output
    }
}

/// Coarse execution boundary for non-autoregressive request batching.
/// Concrete executors own tensor shapes, padding/ragged layout, device buffers,
/// and any model-specific batch preparation.
pub trait BatchExecutor {
    type Input;
    type Output;
    type Error: Error + Send + Sync + 'static;

    fn parameter_version(&self) -> ParameterVersion;
    fn max_batch_items(&self) -> usize;
    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error>;
}

struct Queued<I> {
    request: RequestId,
    input: I,
    parameter_version: ParameterVersion,
}

pub struct Completed<O> {
    request: RequestId,
    output: O,
    parameter_version: ParameterVersion,
}

impl<O> Completed<O> {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub fn output(&self) -> &O {
        &self.output
    }

    #[must_use]
    pub const fn parameter_version(&self) -> ParameterVersion {
        self.parameter_version
    }
}

pub struct BatchRuntime<E: BatchExecutor> {
    executor: E,
    config: BatchConfig,
    queue: VecDeque<Queued<E::Input>>,
    completed: VecDeque<Completed<E::Output>>,
}

impl<E: BatchExecutor> BatchRuntime<E> {
    /// # Errors
    /// Rejects a zero queue bound or an executor that cannot run one item.
    pub fn new(executor: E, config: BatchConfig) -> Result<Self, RuntimeError<E::Error>> {
        if config.max_queued_requests == 0 || executor.max_batch_items() == 0 {
            return Err(RuntimeError::InvalidConfig);
        }
        Ok(Self {
            executor,
            config,
            queue: VecDeque::with_capacity(config.max_queued_requests),
            completed: VecDeque::new(),
        })
    }

    /// # Errors
    /// Rejects a full queue or identity exhaustion.
    pub fn submit(&mut self, input: E::Input) -> Result<RequestId, RuntimeError<E::Error>> {
        if self.queue.len() >= self.config.max_queued_requests {
            return Err(RuntimeError::QueueFull);
        }
        let request = RequestId(
            NEXT_REQUEST_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| RuntimeError::IdentityExhausted)?,
        );
        self.queue.push_back(Queued {
            request,
            input,
            parameter_version: self.executor.parameter_version(),
        });
        Ok(request)
    }

    /// Execute at most one batch.
    ///
    /// # Errors
    /// Rejects a parameter-version change while queued work still targets the
    /// previous version, malformed executor output, or an executor failure.
    pub fn step(&mut self) -> Result<bool, RuntimeError<E::Error>> {
        let Some(front) = self.queue.front() else {
            return Ok(false);
        };
        let version = self.executor.parameter_version();
        if front.parameter_version != version {
            return Err(RuntimeError::ParameterVersionChanged {
                queued: front.parameter_version,
                current: version,
            });
        }

        let mut queued = Vec::new();
        while queued.len() < self.executor.max_batch_items() {
            let Some(next) = self.queue.front() else {
                break;
            };
            if next.parameter_version != version {
                break;
            }
            if let Some(next) = self.queue.pop_front() {
                queued.push(next);
            } else {
                break;
            }
        }
        let expected = queued.iter().map(|item| item.request).collect::<Vec<_>>();
        let jobs = queued
            .into_iter()
            .map(|item| Job {
                request: item.request,
                input: item.input,
            })
            .collect();
        let outputs = match self.executor.execute(jobs) {
            Ok(outputs) => outputs,
            Err(source) => {
                return Err(RuntimeError::Executor {
                    requests: expected,
                    source,
                });
            }
        };
        if outputs.len() != expected.len()
            || outputs
                .iter()
                .zip(expected.iter())
                .any(|(output, request)| output.request() != *request)
        {
            return Err(RuntimeError::MalformedCompletion { requests: expected });
        }
        self.completed.extend(outputs.into_iter().map(|output| Completed {
            request: output.request,
            output: output.output,
            parameter_version: version,
        }));
        Ok(true)
    }

    #[must_use]
    pub fn pop_completed(&mut self) -> Option<Completed<E::Output>> {
        self.completed.pop_front()
    }

    #[must_use]
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    #[must_use]
    pub fn executor(&self) -> &E {
        &self.executor
    }

    #[must_use]
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }
}

#[derive(Debug)]
pub enum RuntimeError<E> {
    InvalidConfig,
    QueueFull,
    IdentityExhausted,
    ParameterVersionChanged {
        queued: ParameterVersion,
        current: ParameterVersion,
    },
    MalformedCompletion {
        requests: Vec<RequestId>,
    },
    Executor {
        requests: Vec<RequestId>,
        source: E,
    },
}

impl<E: fmt::Display> fmt::Display for RuntimeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("batch runtime configuration is invalid"),
            Self::QueueFull => f.write_str("batch runtime request queue is full"),
            Self::IdentityExhausted => {
                f.write_str("batch runtime request identity space is exhausted")
            }
            Self::ParameterVersionChanged { queued, current } => write!(
                f,
                "queued request targets parameter version {}, but executor now exposes version {}",
                queued.get(),
                current.get()
            ),
            Self::MalformedCompletion { requests } => write!(
                f,
                "batch executor returned malformed completion metadata for {} requests",
                requests.len()
            ),
            Self::Executor { source, .. } => source.fmt(f),
        }
    }
}

impl<E: Error + 'static> Error for RuntimeError<E> {}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;

    struct Encoder {
        version: ParameterVersion,
        batches: Vec<Vec<usize>>,
    }

    impl BatchExecutor for Encoder {
        type Input = Vec<f32>;
        type Output = f32;
        type Error = Infallible;

        fn parameter_version(&self) -> ParameterVersion {
            self.version
        }

        fn max_batch_items(&self) -> usize {
            2
        }

        fn execute(
            &mut self,
            batch: Vec<Job<Self::Input>>,
        ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
            self.batches
                .push(batch.iter().map(|job| job.input().len()).collect());
            Ok(batch
                .into_iter()
                .map(|job| {
                    let request = job.request();
                    let sum = job.into_input().into_iter().sum();
                    JobOutput::new(request, sum)
                })
                .collect())
        }
    }

    fn runtime() -> BatchRuntime<Encoder> {
        BatchRuntime::new(
            Encoder {
                version: ParameterVersion::new(7),
                batches: Vec::new(),
            },
            BatchConfig {
                max_queued_requests: 8,
            },
        )
        .expect("runtime")
    }

    #[test]
    fn encoder_batches_without_token_or_sequence_semantics() {
        let mut runtime = runtime();
        let first = runtime.submit(vec![1.0, 2.0]).expect("first");
        let second = runtime.submit(vec![3.0, 4.0, 5.0]).expect("second");
        let third = runtime.submit(vec![6.0, 7.0]).expect("third");
        assert!(runtime.step().expect("step"));
        assert_eq!(runtime.queued(), 1);
        assert_eq!(runtime.executor().batches, vec![vec![2, 3]]);
        let first_output = runtime.pop_completed().expect("first output");
        let second_output = runtime.pop_completed().expect("second output");
        assert_eq!(first_output.request(), first);
        assert_eq!(*first_output.output(), 3.0);
        assert_eq!(first_output.parameter_version(), ParameterVersion::new(7));
        assert_eq!(second_output.request(), second);
        assert_eq!(*second_output.output(), 12.0);
        assert!(runtime.step().expect("second step"));
        assert_eq!(runtime.pop_completed().expect("third output").request(), third);
    }

    #[test]
    fn queued_work_is_pinned_to_a_parameter_version() {
        let mut runtime = runtime();
        runtime.submit(vec![1.0]).expect("request");
        runtime.executor_mut().version = ParameterVersion::new(8);
        assert!(matches!(
            runtime.step(),
            Err(RuntimeError::ParameterVersionChanged {
                queued,
                current
            }) if queued == ParameterVersion::new(7) && current == ParameterVersion::new(8)
        ));
    }
}
