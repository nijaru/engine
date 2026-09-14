//! Model-neutral autoregressive token-generation runtime.
//!
//! An [`Engine`] schedules persistent sequences through one [`GenerationExecutor`].
//! Model implementations own their state layouts, resources, kernels, and
//! artifact adapters. None of those choices appears in the scheduling API.
//!
//! This crate is the current AR runtime, not Ribn's universal model-execution
//! contract. Encoder/pooling, iterative media, realtime/session, and future
//! execution regimes may use different runtimes above the shared foundation.
//! This remains a migration target, not a claim of GPU performance qualification.

mod config;
mod engine;
mod error;
mod executor;
mod output;
mod request;

pub use config::{EngineConfig, SchedulePolicy};
pub use engine::{Engine, EngineStatus, StepStatus};
pub use error::EngineError;
pub use executor::{
    Admission, BatchItem, ExecutionError, ExecutorInfo, GenerationExecutor, GenerationLimits,
    SequenceId, StepCompletion, StepKind, SubmissionId,
};
pub use request::{
    Event, FinishReason, GenerationOptions, RequestId, Sampling, TokenRequest, Usage,
};
