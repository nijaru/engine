//! Model-neutral, asynchronous token-generation runtime.
//!
//! An [`Engine`] schedules persistent sequences through one [`PreparedModel`].
//! Model implementations own their state layouts, resources, kernels, and
//! artifact adapters. None of those choices appears in the scheduling API.
//!
//! This is the migration target, not a claim of GPU performance qualification.

mod engine;
mod model;
mod request;

pub use engine::{Engine, EngineConfig, EngineError, EngineStatus, SchedulePolicy, StepStatus};
pub use model::{
    Admission, BatchItem, ModelError, ModelInfo, ModelLimits, PreparedModel, SequenceId,
    StepCompletion, StepKind, SubmissionId,
};
pub use request::{Event, FinishReason, GenerationOptions, RequestId, Sampling, TokenRequest};
