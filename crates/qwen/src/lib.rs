//! Qwen implementation of Ribn's prepared-model boundary.
//!
//! The CUDA implementation bridges the existing Qwen executor while the new
//! runtime is qualified. Legacy state/plan types stay behind this boundary;
//! they are not requirements for other `ribn::PreparedModel` implementations.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(any(feature = "cuda", test))]
mod execution;
#[cfg(feature = "cuda")]
mod loading;
#[cfg(any(feature = "cuda", test))]
mod state;

#[cfg(feature = "cuda")]
pub use cuda::QwenPrepared;
#[cfg(feature = "cuda")]
pub use loading::{MemoryReport, QwenLoadOptions};
