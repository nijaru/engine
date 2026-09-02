//! Optional NVIDIA backend substrate.
//!
//! The crate remains optional on hosts without CUDA. Its first concrete
//! implementation includes a stateless F32 linear reference path and a
//! correctness-oriented `Q4_K` GEMV primitive used to validate device setup,
//! transfers, NVRTC dispatch, timing, and the core runtime boundary before
//! model-specific Qwen3.8 execution is attempted.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod quantized;
#[cfg(feature = "cuda")]
mod state;

#[cfg(feature = "cuda")]
pub use cuda::{
    CudaF32Weight, CudaQuantizedWeight, CudaReferenceDispatcher, CudaRuntimeError, CudaWeightError,
    CudaWeightStore,
};

#[cfg(feature = "cuda")]
pub use quantized::{
    CudaIq3SEmbedding, CudaIq3SGemv, CudaIq4NlGemv, CudaIq4XsGemv, CudaQ3KGemv, CudaQ4KGemv,
    CudaQ5KGemv, CudaQ6KGemv, CudaQ8_0Gemv, CudaQuantizedKernelError,
};

#[cfg(feature = "cuda")]
pub use state::{
    CudaHybridState, CudaKvState, CudaRecurrentState, CudaStateBuffer, CudaStateError,
};
