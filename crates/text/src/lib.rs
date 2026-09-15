//! Reusable text-generation frontend for Ribn.
//!
//! This crate owns text input formatting, tokenization, incremental decoding,
//! bounded offline batching, and ordinary generation results. The scheduler and
//! token runtime below it see only encoded tokens and know nothing about chat
//! messages or UTF-8 streams.

#[cfg(feature = "cuda")]
mod cuda;
mod error;
mod input;
mod model;
mod process;
mod stream;

#[cfg(feature = "cuda")]
pub use cuda::{LoadOptions, MemoryReport};
pub use error::TextError;
pub use input::{Message, TextInput};
pub use model::{TextBatch, TextConfig, TextModel, TextOwner, TextRequest, TextResponse};
pub use process::{GgufProcessor, ProcessorLimits, TextProcessor};
pub use ribn::{FinishReason, GenerationOptions, Sampling, Usage};
pub use stream::{TextEvent, TextStream};
