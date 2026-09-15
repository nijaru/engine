use std::error::Error;
use std::fmt;
use std::str::Utf8Error;

use engine_gguf::GgufError;
use ribn::driver::DriverError;
use ribn::{EngineError, ExecutionError};

/// Text-layer failure with its underlying source preserved.
///
/// Preparation and decode failures settle one request. [`Self::Admission`] and
/// [`Self::Owner`] carry the execution owner's own diagnostic instead of a
/// flattened string.
#[derive(Clone, Debug)]
pub enum TextError {
    /// A declared text bound rejected the payload before it was retained.
    LimitExceeded {
        field: &'static str,
        allowed: usize,
        actual: usize,
    },
    /// The input cannot be prepared as text, for example an empty chat.
    InvalidInput(&'static str),
    /// Chat-template or tokenizer rejection.
    Processor(GgufError),
    /// The runtime rejected this request, or has no execution owner left.
    Admission(DriverError),
    /// One token is not decodable by the loaded vocabulary, or exceeds its
    /// per-token decoded byte bound.
    Decode { token: u32, source: GgufError },
    /// Decoded token bytes cannot form UTF-8 text.
    Encoding { token: u32, source: Utf8Error },
    /// The execution owner failed; events already delivered remain readable.
    Owner(DriverError),
    /// Text preprocessing or the runtime rejected the request before admission.
    Engine(EngineError),
    /// A prepared model rejected or could not complete preparation.
    Execution(ExecutionError),
    /// The text layer is shut down and accepts no new request.
    Closed,
    /// Delivery ended without a terminal event.
    MissingTerminal,
}

impl TextError {
    pub(crate) fn limit(field: &'static str, allowed: usize, actual: usize) -> Self {
        Self::LimitExceeded {
            field,
            allowed,
            actual,
        }
    }

    /// Whether this request's own content caused the failure, so retrying the
    /// same input cannot help. Admission and owner failures are excluded: they
    /// are capacity or lifecycle conditions, not invalid input.
    #[must_use]
    pub const fn is_request_local(&self) -> bool {
        matches!(
            self,
            Self::LimitExceeded { .. }
                | Self::InvalidInput(_)
                | Self::Processor(_)
                | Self::Decode { .. }
                | Self::Encoding { .. }
        )
    }
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded {
                field,
                allowed,
                actual,
            } => write!(
                formatter,
                "{field} is {actual} bytes, over the {allowed}-byte bound"
            ),
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Processor(error) => write!(formatter, "text preparation failed: {error}"),
            Self::Admission(error) => write!(formatter, "request not admitted: {error}"),
            Self::Decode { token, source } => {
                write!(formatter, "token {token} could not be decoded: {source}")
            }
            Self::Encoding { token, source } => write!(
                formatter,
                "token {token} decoded bytes are not valid UTF-8: {source}"
            ),
            Self::Owner(error) => write!(formatter, "execution owner failed: {error}"),
            Self::Engine(error) => write!(formatter, "runtime rejected its configuration: {error}"),
            Self::Execution(error) => write!(formatter, "model preparation failed: {error}"),
            Self::Closed => formatter.write_str("text model is shut down"),
            Self::MissingTerminal => {
                formatter.write_str("generation ended without a terminal event")
            }
        }
    }
}

impl Error for TextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Processor(error) | Self::Decode { source: error, .. } => Some(error),
            Self::Admission(error) | Self::Owner(error) => Some(error),
            Self::Encoding { source, .. } => Some(source),
            Self::Engine(error) => Some(error),
            Self::Execution(error) => Some(error),
            _ => None,
        }
    }
}
