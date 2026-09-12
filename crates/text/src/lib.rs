//! Reusable text-generation frontend for Ribn.
//!
//! This crate owns text input formatting, tokenization, incremental decoding,
//! and ordinary generation results. The scheduler/runtime continues to operate
//! on token IDs and does not need to know about chat messages or UTF-8 streams.

mod input;
#[cfg(feature = "cuda")]
mod model;

pub use input::{Message, TextInput};
#[cfg(feature = "cuda")]
pub use model::{
    LoadOptions, MemoryReport, TextError, TextEvent, TextModel, TextRequest, TextResponse,
    TextStream,
};
pub use ribn::{FinishReason, GenerationOptions, Sampling, Usage};
