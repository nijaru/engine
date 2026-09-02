//! Optional NVIDIA backend substrate.
//!
//! The crate remains optional on hosts without CUDA. Its first concrete
//! implementation is a stateless F32 linear reference path used to validate
//! device setup, transfers, cuBLAS dispatch, timing, and the core runtime
//! boundary before model-specific Qwen3.8 kernels are attempted.

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{
    CudaF32Weight, CudaQuantizedWeight, CudaReferenceDispatcher, CudaRuntimeError, CudaWeightError,
    CudaWeightStore,
};
