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
mod model_ops;
#[cfg(feature = "cuda")]
mod quantized;
#[cfg(feature = "cuda")]
mod staging;
#[cfg(feature = "cuda")]
mod state;

mod qwen_reference;

#[cfg(feature = "cuda")]
pub use cuda::{
    CudaF32Weight, CudaQuantizedWeight, CudaReferenceDispatcher, CudaRuntimeError, CudaWeightError,
    CudaWeightStore,
};

#[cfg(feature = "cuda")]
pub use model_ops::{CudaModelKernelError, CudaQwen35Ops};

#[cfg(feature = "cuda")]
pub use quantized::{
    CudaIq3SEmbedding, CudaIq3SGemv, CudaIq4NlGemv, CudaIq4XsGemv, CudaQ3KGemv, CudaQ4KEmbedding,
    CudaQ4KGemv, CudaQ5KGemv, CudaQ6KGemv, CudaQ8_0Gemv, CudaQuantizedKernelError,
};

#[cfg(feature = "cuda")]
pub use staging::{CudaQwen35Weights, CudaWeightStagingError, QwenGemvKernel, StagedTensorSource};

#[cfg(feature = "cuda")]
pub use state::{
    CudaHybridState, CudaKvState, CudaRecurrentState, CudaStateBuffer, CudaStateError,
};

pub use qwen_reference::{
    ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTN_Q_HEADS, ATTN_ROPE_BASE, ATTN_ROT_DIMS, AttnLayerWeights,
    AttnStepTrace, FfnLayerWeights, GDN_D_CONV, GDN_HEAD_DIM, GDN_INNER, GDN_K_HEADS, GDN_QKV_DIM,
    GDN_V_HEADS, GdnLayerWeights, GdnStepTrace, N_EMBD, N_FF, gguf_gemv, host_ffn_step,
    host_full_attn_ar_step, host_full_attn_ar_step_traced, host_gdn_ar_step,
    host_gdn_ar_step_traced, l2_normalize, rms_norm_raw, softplus,
};
