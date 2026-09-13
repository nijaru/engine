//! Qwen implementation of Ribn's prepared-model boundary.
//!
//! The CUDA implementation bridges the existing Qwen executor while the new
//! runtime is qualified. Legacy state/plan types stay behind this boundary;
//! they are not requirements for other `ribn::GenerationExecutor` implementations.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(any(feature = "cuda", test))]
mod execution;
#[cfg(feature = "cuda")]
mod loading;
#[cfg(any(feature = "cuda", test))]
mod state;

#[cfg(feature = "cuda")]
pub use cuda::QwenCuda;
#[cfg(feature = "cuda")]
pub use loading::{DEFAULT_PREFILL_CHUNK_MEMBERS, MemoryReport, QwenLoadOptions};

mod config;
pub use config::{ConfigError, QwenConfig, QwenLayerKind};
#[cfg(feature = "gguf")]
mod gguf;
#[cfg(feature = "gguf")]
pub use gguf::QwenGguf;
