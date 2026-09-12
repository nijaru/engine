use crate::{ExecutionError, RequestId};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineError {
    InvalidConfig,
    InvalidRequest,
    QueueFull,
    UnknownRequest(RequestId),
    IdentityExhausted,
    Closed,
    Execution(ExecutionError),
    Faulted(ExecutionError),
}

impl From<ExecutionError> for EngineError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => {
                f.write_str("engine/policy limits are invalid or exceed the prepared model")
            }
            Self::InvalidRequest => {
                f.write_str("input/output token limits are invalid or exceed the model context")
            }
            Self::QueueFull => f.write_str("request queue or queued-input token budget is full"),
            Self::UnknownRequest(request) => {
                write!(f, "request {} is unknown or reclaimed", request.get())
            }
            Self::IdentityExhausted => f.write_str("runtime identity space is exhausted"),
            Self::Closed => f.write_str("engine is closed"),
            Self::Execution(error) => error.fmt(f),
            Self::Faulted(error) => write!(
                f,
                "engine is faulted; shutdown retains cleanup ownership: {error}"
            ),
        }
    }
}

impl std::error::Error for EngineError {}
