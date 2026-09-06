//! Decode-step executor for the pinned Qwen3.8-27B text path.
//!
//! Assembles one batch-1, host-driven, greedy-first decode step over the
//! staged encoded weights: embedding lookup, per-layer GDN/full-attention
//! sequencing across [`CudaHybridState`], FFN, and the greedy output head.
//! Layer sequencing mirrors `host_gdn_ar_step`/`host_full_attn_ar_step`,
//! which are llama.cpp-verified at `cc83d7b48`; each launch is the
//! corresponding parity-tested CUDA kernel. Prefill uses the same
//! state-advancing model body autoregressively, one token at a time, but can
//! skip the output head for intermediate prompt tokens whose successor is
//! already known.
//!
//! The executor defines its own [`QwenLayerKind`] so `engine-gguf` remains
//! optional; model providers translate their layer catalogs into this
//! data-only plan.

use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, CudaView, CudaViewMut, PinnedHostSlice,
};
use engine_core::DataType;

use crate::cuda::CudaF32Weight;
use crate::model_ops::{CudaModelKernelError, CudaQwen35Ops};
use crate::quantized::{CudaQ4KEmbedding, CudaQuantizedKernelError};
use crate::staging::CudaQwen35Weights;
use crate::state::{CudaHybridState, CudaStateError};

use crate::{
    ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTN_Q_HEADS, ATTN_ROPE_BASE, ATTN_ROT_DIMS, GDN_D_CONV,
    GDN_HEAD_DIM, GDN_INNER, GDN_K_HEADS, GDN_QKV_DIM, GDN_V_HEADS, N_EMBD, N_FF,
};

/// First q/k channel offset inside the joint GDN qkv projection output.
const GDN_K_OFFSET: usize = GDN_K_HEADS * GDN_HEAD_DIM;
/// First v channel offset inside the joint GDN qkv projection output.
const GDN_V_OFFSET: usize = 2 * GDN_K_OFFSET;

const RECURRENT_LAYER_TENSORS: &[&str] = &[
    "attn_gate.weight",
    "attn_norm.weight",
    "attn_qkv.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "ssm_a",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight",
];

const FULL_ATTENTION_LAYER_TENSORS: &[&str] = &[
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
];

/// Which decoder block geometry a language layer uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenLayerKind {
    /// Gated-DeltaNet recurrent block.
    Recurrent,
    /// Standard full-attention block.
    FullAttention,
}

/// Errors surfaced by the decode-step executor.
#[derive(Debug)]
pub enum CudaDecodeError {
    /// A weight tensor is missing from the staged set.
    MissingTensor(String),
    /// A required GEMV kernel was not compiled during staging.
    MissingKernel(u32),
    /// The layer plan or the physical state does not match the pinned
    /// geometry.
    InvalidPlan(String),
    /// The physical state rejected a typed access.
    State(CudaStateError),
    /// A quantized GEMV or embedding kernel failed.
    Kernel(CudaQuantizedKernelError),
    /// A model-op kernel failed.
    Ops(CudaModelKernelError),
    /// A device allocation or copy failed.
    Driver(String),
    /// The token position exceeded the KV token capacity.
    PositionOverflow { position: u32, capacity: u32 },
}

impl std::fmt::Display for CudaDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor(name) => write!(f, "staged weights lack {name:?}"),
            Self::MissingKernel(value_type) => {
                write!(f, "no staged GEMV kernel for value type {value_type}")
            }
            Self::InvalidPlan(message) => write!(f, "invalid decode plan: {message}"),
            Self::State(error) => write!(f, "physical state error: {error}"),
            Self::Kernel(error) => write!(f, "quantized kernel error: {error}"),
            Self::Ops(error) => write!(f, "model-op kernel error: {error}"),
            Self::Driver(message) => write!(f, "CUDA driver error: {message}"),
            Self::PositionOverflow { position, capacity } => write!(
                f,
                "token position {position} exceeds KV capacity {capacity}"
            ),
        }
    }
}

impl std::error::Error for CudaDecodeError {}

impl From<CudaStateError> for CudaDecodeError {
    fn from(error: CudaStateError) -> Self {
        Self::State(error)
    }
}

impl From<CudaQuantizedKernelError> for CudaDecodeError {
    fn from(error: CudaQuantizedKernelError) -> Self {
        Self::Kernel(error)
    }
}

impl From<CudaModelKernelError> for CudaDecodeError {
    fn from(error: CudaModelKernelError) -> Self {
        Self::Ops(error)
    }
}

struct CommonLayerTensorNames {
    attn_norm: String,
    post_attention_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

struct RecurrentLayerTensorNames {
    attn_qkv: String,
    attn_gate: String,
    ssm_beta: String,
    ssm_alpha: String,
    ssm_dt_bias: String,
    ssm_a: String,
    ssm_conv1d: String,
    ssm_norm: String,
    ssm_out: String,
}

struct FullAttentionLayerTensorNames {
    q: String,
    q_norm: String,
    k: String,
    k_norm: String,
    v: String,
    output: String,
}

enum AttentionLayerTensorNames {
    Recurrent(RecurrentLayerTensorNames),
    FullAttention(FullAttentionLayerTensorNames),
}

struct LayerTensorNames {
    common: CommonLayerTensorNames,
    attention: AttentionLayerTensorNames,
}

impl LayerTensorNames {
    fn new(layer: usize, kind: QwenLayerKind) -> Self {
        let prefix = format!("blk.{layer}.");
        let common = CommonLayerTensorNames {
            attn_norm: format!("{prefix}attn_norm.weight"),
            post_attention_norm: format!("{prefix}post_attention_norm.weight"),
            ffn_gate: format!("{prefix}ffn_gate.weight"),
            ffn_up: format!("{prefix}ffn_up.weight"),
            ffn_down: format!("{prefix}ffn_down.weight"),
        };
        let attention = match kind {
            QwenLayerKind::Recurrent => {
                AttentionLayerTensorNames::Recurrent(RecurrentLayerTensorNames {
                    attn_qkv: format!("{prefix}attn_qkv.weight"),
                    attn_gate: format!("{prefix}attn_gate.weight"),
                    ssm_beta: format!("{prefix}ssm_beta.weight"),
                    ssm_alpha: format!("{prefix}ssm_alpha.weight"),
                    ssm_dt_bias: format!("{prefix}ssm_dt.bias"),
                    ssm_a: format!("{prefix}ssm_a"),
                    ssm_conv1d: format!("{prefix}ssm_conv1d.weight"),
                    ssm_norm: format!("{prefix}ssm_norm.weight"),
                    ssm_out: format!("{prefix}ssm_out.weight"),
                })
            }
            QwenLayerKind::FullAttention => {
                AttentionLayerTensorNames::FullAttention(FullAttentionLayerTensorNames {
                    q: format!("{prefix}attn_q.weight"),
                    q_norm: format!("{prefix}attn_q_norm.weight"),
                    k: format!("{prefix}attn_k.weight"),
                    k_norm: format!("{prefix}attn_k_norm.weight"),
                    v: format!("{prefix}attn_v.weight"),
                    output: format!("{prefix}attn_output.weight"),
                })
            }
        };
        Self { common, attention }
    }
}

/// Batch-1 decode-step runner for the staged Qwen3.8-27B text path.
///
/// The runner owns scratch buffers sized for the pinned geometry and
/// sequences one token's model-state transition: embed → layer blocks (GDN or
/// full attention by [`QwenLayerKind`]) and optionally the greedy output head.
/// The caller owns the physical state, token loop, and positions.
///
/// State layer mapping matches the distinct state families: full-attention
/// layers share one KV family indexed by their order among full-attention
/// layers, recurrent layers likewise index the recurrent family.
pub struct CudaQwen35Decode {
    stream: Arc<CudaStream>,
    ops: Arc<CudaQwen35Ops>,
    embedding: CudaQ4KEmbedding,
    weights: Arc<CudaQwen35Weights>,
    layer_kinds: Vec<QwenLayerKind>,
    tensor_names: Arc<[LayerTensorNames]>,
    kv_slot: Vec<u32>,
    recurrent_slot: Vec<u32>,
    epsilon: f32,
    hidden: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    attn_incr: CudaSlice<f32>,
    ffn_incr: CudaSlice<f32>,
    qkv_mixed: CudaSlice<f32>,
    z_gate: CudaSlice<f32>,
    beta_raw: CudaSlice<f32>,
    alpha_raw: CudaSlice<f32>,
    decay: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    conv_out: CudaSlice<f32>,
    gdn_q: CudaSlice<f32>,
    gdn_k: CudaSlice<f32>,
    state_out: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    q_raw: CudaSlice<f32>,
    q_packed: CudaSlice<f32>,
    k_raw: CudaSlice<f32>,
    k_normed: CudaSlice<f32>,
    v_raw: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    ffn_gate_buf: CudaSlice<f32>,
    ffn_up_buf: CudaSlice<f32>,
    ffn_act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    selected_token: CudaSlice<u32>,
    /// Attention score scratch, `[q_heads][token capacity]`, allocated on
    /// first use once the KV capacity is known.
    scores: Option<CudaSlice<f32>>,
    scores_stride: usize,
    gemv_mode: GemvMode,
}

/// Which `GEMV` kernel variant the executor launches for quantized
/// projections.
///
/// `Warp` is the default: the warp-cooperative kernels (one warp per output
/// row, coalesced weight/input reads) are hardware-qualified against the
/// scalar oracle by per-family parity tests and a full-model greedy replay.
/// `Scalar` remains available as the parity-tested correctness oracle.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GemvMode {
    /// One-thread-per-output-row scalar kernels (correctness oracle).
    Scalar,
    /// Warp-cooperative row kernels (default qualified path).
    #[default]
    Warp,
}

impl CudaQwen35Decode {
    /// Build the executor over staged weights and a layer plan.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when the plan is empty, epsilon is not
    /// positive and finite, a required tensor is missing from the staged
    /// set on its expected route, or kernel compilation/allocation fails.
    #[allow(
        clippy::too_many_lines,
        reason = "one validation pass plus slot construction; splitting it hides the plan contract"
    )]
    pub fn new(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        weights: Arc<CudaQwen35Weights>,
        layer_kinds: Vec<QwenLayerKind>,
        epsilon: f32,
    ) -> Result<Self, CudaDecodeError> {
        if layer_kinds.is_empty() {
            return Err(CudaDecodeError::InvalidPlan(
                "the layer plan must not be empty".to_owned(),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaDecodeError::InvalidPlan(
                "epsilon must be positive and finite".to_owned(),
            ));
        }
        for (layer, kind) in layer_kinds.iter().enumerate() {
            let prefix = format!("blk.{layer}.");
            let tensors = match kind {
                QwenLayerKind::Recurrent => RECURRENT_LAYER_TENSORS,
                QwenLayerKind::FullAttention => FULL_ATTENTION_LAYER_TENSORS,
            };
            for tensor in tensors {
                validate_staged(&weights, &format!("{prefix}{tensor}"))?;
            }
        }
        validate_staged(&weights, "token_embd.weight")?;
        validate_staged(&weights, "output.weight")?;
        validate_staged(&weights, "output_norm.weight")?;

        let output_weight = weights
            .quantized_tensor("output.weight")
            .ok_or_else(|| CudaDecodeError::MissingTensor("output.weight".to_owned()))?;
        let dimensions = output_weight.spec().dimensions();
        if dimensions.len() != 2 {
            return Err(CudaDecodeError::InvalidPlan(
                "output.weight must be a rank-2 tensor".to_owned(),
            ));
        }
        let input_width = usize::try_from(dimensions[0]).map_err(|_| {
            CudaDecodeError::InvalidPlan("output input width does not fit the host".to_owned())
        })?;
        if input_width != N_EMBD {
            return Err(CudaDecodeError::InvalidPlan(format!(
                "output.weight input width is {input_width}, expected {N_EMBD}"
            )));
        }
        let vocab = usize::try_from(dimensions[1]).map_err(|_| {
            CudaDecodeError::InvalidPlan("output vocabulary does not fit the host".to_owned())
        })?;

        let tensor_names: Arc<[LayerTensorNames]> = layer_kinds
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, kind)| LayerTensorNames::new(layer, kind))
            .collect::<Vec<_>>()
            .into();
        let mut kv_slot = Vec::with_capacity(layer_kinds.len());
        let mut recurrent_slot = Vec::with_capacity(layer_kinds.len());
        let mut kv_count = 0_u32;
        let mut recurrent_count = 0_u32;
        // Both slot tables are indexed by absolute layer index; a layer of
        // one family records the next unused slot of its own family only.
        for kind in &layer_kinds {
            match kind {
                QwenLayerKind::Recurrent => {
                    recurrent_slot.push(recurrent_count);
                    recurrent_count += 1;
                    kv_slot.push(kv_count);
                }
                QwenLayerKind::FullAttention => {
                    kv_slot.push(kv_count);
                    kv_count += 1;
                    recurrent_slot.push(recurrent_count);
                }
            }
        }

        let ops = Arc::new(CudaQwen35Ops::from_context(context, stream.clone())?);
        let embedding = CudaQ4KEmbedding::from_context(context, stream.clone())?;
        let scratch = alloc_scratch(&stream, vocab)?;
        let selected_token = stream
            .alloc_zeros::<u32>(1)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;

        Ok(Self {
            stream,
            ops,
            embedding,
            weights,
            layer_kinds,
            tensor_names,
            kv_slot,
            recurrent_slot,
            epsilon,
            scores: None,
            scores_stride: 0,
            gemv_mode: GemvMode::default(),
            hidden: scratch.hidden,
            normed: scratch.normed,
            attn_incr: scratch.attn_incr,
            ffn_incr: scratch.ffn_incr,
            qkv_mixed: scratch.qkv_mixed,
            z_gate: scratch.z_gate,
            beta_raw: scratch.beta_raw,
            alpha_raw: scratch.alpha_raw,
            decay: scratch.decay,
            beta: scratch.beta,
            conv_out: scratch.conv_out,
            gdn_q: scratch.gdn_q,
            gdn_k: scratch.gdn_k,
            state_out: scratch.state_out,
            gated: scratch.gated,
            q_raw: scratch.q_raw,
            q_packed: scratch.q_packed,
            k_raw: scratch.k_raw,
            k_normed: scratch.k_normed,
            v_raw: scratch.v_raw,
            attn_out: scratch.attn_out,
            ffn_gate_buf: scratch.ffn_gate_buf,
            ffn_up_buf: scratch.ffn_up_buf,
            ffn_act: scratch.ffn_act,
            logits: scratch.logits,
            selected_token,
        })
    }

    /// Advance model state for one known prompt token without producing logits.
    ///
    /// Intermediate prefill tokens do not need an output projection or sampled
    /// token: the next input is already supplied by the prompt. This path
    /// therefore skips output `RMSNorm`, vocabulary projection, argmax, and the
    /// device-to-host token copy. Launches remain ordered on the decoder stream,
    /// so a following prefill or decode step observes the updated model state.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when state/geometry is invalid or execution
    /// fails.
    pub fn prefill_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
        self.run_step(state, token, position)
    }

    /// Run one full forward pass for `token` at absolute sequence `position`
    /// and return the greedy token choice.
    ///
    /// The output-token readback is an intentional blocking host boundary on
    /// the current correctness path.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when state/geometry is invalid or execution
    /// or output selection fails.
    pub fn decode_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<u32, CudaDecodeError> {
        self.run_step(state, token, position)?;
        self.select_greedy_token()
    }

    /// Run one full forward pass for `token` at absolute sequence
    /// `position` and stream-order the greedy selection into one `u32` of
    /// pinned host memory.
    ///
    /// Unlike [`Self::decode_step`], the host is not blocked on the result:
    /// the device-to-host copy is enqueued on the decoder stream and the
    /// caller reads `destination` only after a recorded completion event
    /// covers it. The shared device argmax slot is reused by later rows, so
    /// each row needs its own pinned destination to preserve its result
    /// until completion.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when state/geometry is invalid or execution
    /// or the stream-ordered copy fails.
    pub fn decode_step_into_pinned(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
        destination: &mut PinnedHostSlice<u32>,
    ) -> Result<(), CudaDecodeError> {
        self.run_step(state, token, position)?;
        self.select_greedy_token_into_pinned(destination)
    }

    fn run_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
        self.validate_state(state)?;
        let capacity = state.kv().map_or(u32::MAX, |kv| kv.spec().block_tokens());
        if position >= capacity {
            return Err(CudaDecodeError::PositionOverflow { position, capacity });
        }
        if state.kv().is_some() && self.scores.is_none() {
            let stride = capacity as usize;
            self.scores = Some(
                self.stream
                    .alloc_zeros::<f32>(ATTN_Q_HEADS * stride)
                    .map_err(|error| CudaDecodeError::Driver(error.to_string()))?,
            );
            self.scores_stride = stride;
        }

        let embedding_weight = self
            .weights
            .quantized_tensor("token_embd.weight")
            .ok_or_else(|| CudaDecodeError::MissingTensor("token_embd.weight".to_owned()))?;
        self.embedding
            .execute(embedding_weight, token, &mut self.hidden)?;

        let tensor_names = Arc::clone(&self.tensor_names);
        for (layer, names) in tensor_names.iter().enumerate() {
            self.ops.rms_norm(
                &self.hidden,
                f32_slice(&self.weights, &names.common.attn_norm)?,
                &mut self.normed,
                self.epsilon,
            )?;
            match &names.attention {
                AttentionLayerTensorNames::Recurrent(attention) => {
                    self.recurrent_layer(state, layer, attention)?;
                }
                AttentionLayerTensorNames::FullAttention(attention) => {
                    self.full_attention_layer(state, layer, attention, position)?;
                }
            }
            self.ops.rms_norm(
                &self.hidden,
                f32_slice(&self.weights, &names.common.post_attention_norm)?,
                &mut self.normed,
                self.epsilon,
            )?;
            gemv(
                &self.weights,
                &names.common.ffn_gate,
                &self.normed,
                &mut self.ffn_gate_buf,
                self.gemv_mode,
            )?;
            gemv(
                &self.weights,
                &names.common.ffn_up,
                &self.normed,
                &mut self.ffn_up_buf,
                self.gemv_mode,
            )?;
            self.ops
                .silu_mul(&self.ffn_gate_buf, &self.ffn_up_buf, &mut self.ffn_act)?;
            gemv(
                &self.weights,
                &names.common.ffn_down,
                &self.ffn_act,
                &mut self.ffn_incr,
                self.gemv_mode,
            )?;
            self.ops.residual_add(&mut self.hidden, &self.ffn_incr)?;
        }

        Ok(())
    }

    fn select_greedy_token(&mut self) -> Result<u32, CudaDecodeError> {
        self.run_output_head()?;
        // The host needs the selected token before it can schedule the next
        // autoregressive step, so this remains an intentional blocking
        // completion boundary. Copy into stack storage rather than allocating
        // a new Vec for every token.
        let mut selected = [0_u32; 1];
        self.stream
            .memcpy_dtoh(&self.selected_token, &mut selected)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        Ok(selected[0])
    }

    /// Run the output head and stream-order an async greedy selection copy
    /// into one `u32` of pinned host memory.
    ///
    /// Unlike [`Self::decode_step`], the host is not blocked on the selected
    /// token: the copy is enqueued on the decoder stream, and the caller reads
    /// `destination` only after a recorded completion event covers it. The
    /// shared device argmax slot is reused by later rows; the pinned
    /// destination preserves each row's result until completion.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when the output head or the stream-ordered
    /// copy into pinned memory fails.
    fn select_greedy_token_into_pinned(
        &mut self,
        destination: &mut PinnedHostSlice<u32>,
    ) -> Result<(), CudaDecodeError> {
        self.run_output_head()?;
        self.stream
            .memcpy_dtoh(&self.selected_token, destination)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        Ok(())
    }

    /// Output `RMSNorm`, vocabulary projection, and device-side argmax over the
    /// current residual stream, writing the winning token into the shared
    /// device result slot.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when a kernel launch fails.
    fn run_output_head(&mut self) -> Result<(), CudaDecodeError> {
        self.ops.rms_norm(
            &self.hidden,
            f32_slice(&self.weights, "output_norm.weight")?,
            &mut self.normed,
            self.epsilon,
        )?;
        gemv(
            &self.weights,
            "output.weight",
            &self.normed,
            &mut self.logits,
            self.gemv_mode,
        )?;
        self.ops
            .argmax_into(&self.logits, &mut self.selected_token)?;
        Ok(())
    }

    /// Copy the current residual stream to the host. The copy is
    /// stream-ordered and synchronized; it exists for parity debugging
    /// against the host reference.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError::Driver`] when the device copy fails.
    /// Select which `GEMV` kernel variant subsequent steps launch.
    ///
    /// `Warp` is the qualified default; `Scalar` is the correctness oracle.
    /// Mode changes take effect for the next launched step and never mutate
    /// in-flight launches.
    #[must_use]
    pub const fn gemv_mode(&self) -> GemvMode {
        self.gemv_mode
    }

    /// The staged weights this executor runs over.
    #[must_use]
    pub const fn weights(&self) -> &Arc<CudaQwen35Weights> {
        &self.weights
    }

    /// The layer plan this executor was built with.
    #[must_use]
    pub fn layer_kinds(&self) -> &[QwenLayerKind] {
        &self.layer_kinds
    }

    /// The `RMSNorm` epsilon this executor applies.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Set the `GEMV` kernel variant used by subsequent steps.
    pub fn set_gemv_mode(&mut self, mode: GemvMode) {
        self.gemv_mode = mode;
    }

    /// Copy the current residual stream to the host. The copy is
    /// stream-ordered and synchronized; it exists for parity debugging
    /// against the host reference.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError::Driver`] when the device copy fails.
    pub fn copy_hidden(&self) -> Result<Vec<f32>, CudaDecodeError> {
        self.stream
            .clone_dtoh(&self.hidden)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))
    }

    /// Record a completion event covering everything queued on the decoder
    /// stream so far.
    ///
    /// The event is a single one-stream ordering point: all kernels and
    /// stream-ordered copies of the submission, including pinned output
    /// copies, precede it. Polling the event's completion therefore covers
    /// every resource the submission still owns.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError::Driver`] when event creation or recording
    /// fails.
    pub fn record_completion_event(&self) -> Result<CudaEvent, CudaDecodeError> {
        self.stream
            .record_event(None)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))
    }

    fn recurrent_layer(
        &mut self,
        state: &mut CudaHybridState,
        layer: usize,
        names: &RecurrentLayerTensorNames,
    ) -> Result<(), CudaDecodeError> {
        let slot = self.recurrent_slot[layer];
        let recurrent = state.recurrent_mut().ok_or(missing_recurrent())?;
        let (matrix_buffer, conv_buffer) = recurrent.layer_mut(slot).ok_or_else(|| {
            CudaDecodeError::InvalidPlan(format!("recurrent state lacks slot {slot}"))
        })?;
        let matrix = matrix_buffer.as_f32_mut().ok_or_else(|| {
            CudaDecodeError::InvalidPlan("recurrent matrix must be F32".to_owned())
        })?;
        let history = conv_buffer.as_f32_mut().ok_or_else(|| {
            CudaDecodeError::InvalidPlan("recurrent convolution history must be F32".to_owned())
        })?;

        gemv(
            &self.weights,
            &names.attn_qkv,
            &self.normed,
            &mut self.qkv_mixed,
            self.gemv_mode,
        )?;
        gemv(
            &self.weights,
            &names.attn_gate,
            &self.normed,
            &mut self.z_gate,
            self.gemv_mode,
        )?;
        gemv(
            &self.weights,
            &names.ssm_beta,
            &self.normed,
            &mut self.beta_raw,
            self.gemv_mode,
        )?;
        gemv(
            &self.weights,
            &names.ssm_alpha,
            &self.normed,
            &mut self.alpha_raw,
            self.gemv_mode,
        )?;
        let dt_bias = f32_slice(&self.weights, &names.ssm_dt_bias)?;
        let ssm_a = f32_slice(&self.weights, &names.ssm_a)?;
        let conv_weight = f32_slice(&self.weights, &names.ssm_conv1d)?;
        let ssm_norm = f32_slice(&self.weights, &names.ssm_norm)?;

        self.ops.gdn_scalar_gate(
            &self.alpha_raw,
            &self.beta_raw,
            dt_bias,
            ssm_a,
            &mut self.decay,
            &mut self.beta,
        )?;
        self.ops
            .gdn_conv_silu(&self.qkv_mixed, conv_weight, history, &mut self.conv_out)?;

        copy_range(
            &self.stream,
            &self.conv_out,
            0..GDN_K_OFFSET,
            &mut self.gdn_q,
        )?;
        copy_range(
            &self.stream,
            &self.conv_out,
            GDN_K_OFFSET..GDN_V_OFFSET,
            &mut self.gdn_k,
        )?;
        self.ops
            .l2_norm_heads(&mut self.gdn_q, GDN_K_HEADS, GDN_HEAD_DIM, self.epsilon)?;
        self.ops
            .l2_norm_heads(&mut self.gdn_k, GDN_K_HEADS, GDN_HEAD_DIM, self.epsilon)?;

        self.ops.gdn_state_update(
            matrix,
            &self.gdn_q,
            &self.gdn_k,
            &self.conv_out,
            &self.decay,
            &self.beta,
            &mut self.state_out,
            GDN_V_HEADS,
            GDN_K_HEADS,
            GDN_HEAD_DIM,
            GDN_V_OFFSET,
        )?;
        self.ops.gdn_gated_norm(
            &self.state_out,
            &self.z_gate,
            ssm_norm,
            &mut self.gated,
            GDN_V_HEADS,
            GDN_HEAD_DIM,
            self.epsilon,
        )?;
        gemv(
            &self.weights,
            &names.ssm_out,
            &self.gated,
            &mut self.attn_incr,
            self.gemv_mode,
        )?;
        self.ops.residual_add(&mut self.hidden, &self.attn_incr)?;
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one block sequence per llama.cpp-verified host step"
    )]
    fn full_attention_layer(
        &mut self,
        state: &mut CudaHybridState,
        layer: usize,
        names: &FullAttentionLayerTensorNames,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
        let slot = self.kv_slot[layer];
        gemv(
            &self.weights,
            &names.q,
            &self.normed,
            &mut self.q_raw,
            self.gemv_mode,
        )?;
        let q_norm = f32_slice(&self.weights, &names.q_norm)?;
        self.ops.q_gate_norm(
            &self.q_raw,
            q_norm,
            &mut self.q_packed,
            ATTN_Q_HEADS,
            ATTN_HEAD_DIM,
            self.epsilon,
        )?;
        gemv(
            &self.weights,
            &names.k,
            &self.normed,
            &mut self.k_raw,
            self.gemv_mode,
        )?;
        let k_norm = f32_slice(&self.weights, &names.k_norm)?;
        self.ops.strided_rms_norm(
            &self.k_raw,
            k_norm,
            &mut self.k_normed,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.epsilon,
        )?;
        gemv(
            &self.weights,
            &names.v,
            &self.normed,
            &mut self.v_raw,
            self.gemv_mode,
        )?;
        self.ops.rope_neox(
            &mut self.q_packed,
            u64::from(position),
            ATTN_Q_HEADS,
            ATTN_HEAD_DIM,
            ATTN_ROT_DIMS,
            ATTN_ROPE_BASE,
        )?;
        self.ops.rope_neox(
            &mut self.k_normed,
            u64::from(position),
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            ATTN_ROT_DIMS,
            ATTN_ROPE_BASE,
        )?;

        let kv = state.kv_mut().ok_or(missing_kv())?;
        let (keys_buffer, values_buffer) = kv
            .layer_mut(slot)
            .ok_or_else(|| CudaDecodeError::InvalidPlan(format!("KV state lacks slot {slot}")))?;
        {
            let keys = keys_buffer
                .as_f16_mut()
                .ok_or_else(|| CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned()))?;
            let values = values_buffer
                .as_f16_mut()
                .ok_or_else(|| CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned()))?;
            self.ops.kv_append_f16(
                &self.k_normed,
                &self.v_raw,
                keys,
                values,
                position as usize,
                ATTN_KV_HEADS,
                ATTN_HEAD_DIM,
            )?;
        }
        let keys = keys_buffer
            .as_f16()
            .ok_or_else(|| CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned()))?;
        let values = values_buffer
            .as_f16()
            .ok_or_else(|| CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned()))?;
        let tokens = position as usize + 1;
        let scores = self
            .scores
            .as_mut()
            .ok_or_else(|| CudaDecodeError::InvalidPlan("score scratch is unset".to_owned()))?;
        self.ops.attn_score_gqa(
            &self.q_packed,
            keys,
            values,
            &self.q_raw,
            scores,
            &mut self.attn_out,
            tokens,
            self.scores_stride,
            ATTN_Q_HEADS,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
        )?;
        gemv(
            &self.weights,
            &names.output,
            &self.attn_out,
            &mut self.attn_incr,
            self.gemv_mode,
        )?;
        self.ops.residual_add(&mut self.hidden, &self.attn_incr)?;
        Ok(())
    }

    fn validate_state(&self, state: &CudaHybridState) -> Result<(), CudaDecodeError> {
        let full_count = self
            .layer_kinds
            .iter()
            .filter(|kind| **kind == QwenLayerKind::FullAttention)
            .count();
        let recurrent_count = self.layer_kinds.len() - full_count;
        if full_count > 0 {
            let kv = state.kv().ok_or(missing_kv())?;
            let spec = kv.spec();
            if usize::from(spec.layer_count()) != full_count
                || usize::from(spec.kv_heads()) != ATTN_KV_HEADS
                || usize::from(spec.head_dim()) != ATTN_HEAD_DIM
                || spec.dtype() != DataType::F16
            {
                return Err(CudaDecodeError::InvalidPlan(format!(
                    "KV state must hold {full_count} layers with {ATTN_KV_HEADS} heads, head dim {ATTN_HEAD_DIM}, F16"
                )));
            }
        }
        if recurrent_count > 0 {
            let recurrent = state.recurrent().ok_or(missing_recurrent())?;
            let spec = recurrent.spec();
            let matrix = spec.matrix();
            let convolution = spec.convolution();
            // The state matrix is one [k_dim][v_dim] F32 block per v head
            // (`gdn_state_update` indexes [v_head][k][v]); the convolution
            // buffer holds d_conv - 1 history elements per channel
            // (`gdn_conv_silu` validates channels * 3).
            if usize::from(spec.layer_count()) != recurrent_count
                || usize::from(matrix.matrix_count()) != GDN_V_HEADS
                || usize::from(matrix.rows()) != GDN_HEAD_DIM
                || usize::from(matrix.columns()) != GDN_HEAD_DIM
                || u64::from(convolution.channels()) != GDN_QKV_DIM as u64
                || usize::from(convolution.history_tokens()) != GDN_D_CONV - 1
                || spec.matrix_dtype() != DataType::F32
                || spec.convolution_dtype() != DataType::F32
            {
                return Err(CudaDecodeError::InvalidPlan(format!(
                    "recurrent state must hold {recurrent_count} layers with per-v-head [128x128] F32 matrices and {GDN_QKV_DIM}x3 F32 convolution history"
                )));
            }
        }
        Ok(())
    }
}

/// The state matrix and history access for one recurrent layer slot.
fn missing_recurrent() -> CudaDecodeError {
    CudaDecodeError::State(CudaStateError::MissingFamily {
        family: "recurrent state",
    })
}

/// The KV cache access for one full-attention layer slot.
fn missing_kv() -> CudaDecodeError {
    CudaDecodeError::State(CudaStateError::MissingFamily {
        family: "full-attention KV",
    })
}

/// One persistent F32 scratch buffer set for the pinned geometry.
struct DecodeScratch {
    hidden: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    attn_incr: CudaSlice<f32>,
    ffn_incr: CudaSlice<f32>,
    qkv_mixed: CudaSlice<f32>,
    z_gate: CudaSlice<f32>,
    beta_raw: CudaSlice<f32>,
    alpha_raw: CudaSlice<f32>,
    decay: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    conv_out: CudaSlice<f32>,
    gdn_q: CudaSlice<f32>,
    gdn_k: CudaSlice<f32>,
    state_out: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    q_raw: CudaSlice<f32>,
    q_packed: CudaSlice<f32>,
    k_raw: CudaSlice<f32>,
    k_normed: CudaSlice<f32>,
    v_raw: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    ffn_gate_buf: CudaSlice<f32>,
    ffn_up_buf: CudaSlice<f32>,
    ffn_act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
}

/// Allocate one F32 scratch buffer.
fn alloc(stream: &Arc<CudaStream>, length: usize) -> Result<CudaSlice<f32>, CudaDecodeError> {
    stream
        .alloc_zeros::<f32>(length)
        .map_err(|error| CudaDecodeError::Driver(error.to_string()))
}

/// Allocate every decode-step scratch buffer for the pinned geometry.
fn alloc_scratch(stream: &Arc<CudaStream>, vocab: usize) -> Result<DecodeScratch, CudaDecodeError> {
    Ok(DecodeScratch {
        hidden: alloc(stream, N_EMBD)?,
        normed: alloc(stream, N_EMBD)?,
        attn_incr: alloc(stream, N_EMBD)?,
        ffn_incr: alloc(stream, N_EMBD)?,
        qkv_mixed: alloc(stream, GDN_QKV_DIM)?,
        z_gate: alloc(stream, GDN_INNER)?,
        beta_raw: alloc(stream, GDN_V_HEADS)?,
        alpha_raw: alloc(stream, GDN_V_HEADS)?,
        decay: alloc(stream, GDN_V_HEADS)?,
        beta: alloc(stream, GDN_V_HEADS)?,
        conv_out: alloc(stream, GDN_QKV_DIM)?,
        gdn_q: alloc(stream, GDN_K_HEADS * GDN_HEAD_DIM)?,
        gdn_k: alloc(stream, GDN_K_HEADS * GDN_HEAD_DIM)?,
        state_out: alloc(stream, GDN_INNER)?,
        gated: alloc(stream, GDN_INNER)?,
        q_raw: alloc(stream, ATTN_Q_HEADS * 2 * ATTN_HEAD_DIM)?,
        q_packed: alloc(stream, ATTN_Q_HEADS * ATTN_HEAD_DIM)?,
        k_raw: alloc(stream, ATTN_KV_HEADS * ATTN_HEAD_DIM)?,
        k_normed: alloc(stream, ATTN_KV_HEADS * ATTN_HEAD_DIM)?,
        v_raw: alloc(stream, ATTN_KV_HEADS * ATTN_HEAD_DIM)?,
        attn_out: alloc(stream, ATTN_Q_HEADS * ATTN_HEAD_DIM)?,
        ffn_gate_buf: alloc(stream, N_FF)?,
        ffn_up_buf: alloc(stream, N_FF)?,
        ffn_act: alloc(stream, N_FF)?,
        logits: alloc(stream, vocab)?,
    })
}

/// Whether a tensor name stages through the F32 route (small norms,
/// biases, conv weights) rather than the quantized route.
fn is_f32_staged(name: &str) -> bool {
    name.ends_with("norm.weight")
        || name.ends_with("ssm_dt.bias")
        || name.ends_with("ssm_a")
        || name.ends_with("ssm_conv1d.weight")
}

fn validate_staged(weights: &CudaQwen35Weights, name: &str) -> Result<(), CudaDecodeError> {
    let present = if is_f32_staged(name) {
        weights.f32_tensor(name).is_some()
    } else {
        weights.quantized_tensor(name).is_some()
    };
    if present {
        Ok(())
    } else {
        Err(CudaDecodeError::MissingTensor(name.to_owned()))
    }
}

fn f32_slice<'a>(
    weights: &'a CudaQwen35Weights,
    name: &str,
) -> Result<&'a CudaSlice<f32>, CudaDecodeError> {
    weights
        .f32_tensor(name)
        .map(CudaF32Weight::data)
        .ok_or_else(|| CudaDecodeError::MissingTensor(name.to_owned()))
}

fn gemv(
    weights: &CudaQwen35Weights,
    name: &str,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    mode: GemvMode,
) -> Result<(), CudaDecodeError> {
    let weight = weights
        .quantized_tensor(name)
        .ok_or_else(|| CudaDecodeError::MissingTensor(name.to_owned()))?;
    let kernel = weights
        .gemv_for(weight.value_type())
        .ok_or(CudaDecodeError::MissingKernel(weight.value_type()))?;
    match mode {
        GemvMode::Scalar => kernel.execute(weight, input, output),
        GemvMode::Warp => kernel.execute_warp(weight, input, output),
    }?;
    Ok(())
}

/// Run one quantized projection for `members` batch-major rows in a single
/// batched warp launch.
fn gemv_batch(
    weights: &CudaQwen35Weights,
    name: &str,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    members: usize,
) -> Result<(), CudaDecodeError> {
    let weight = weights
        .quantized_tensor(name)
        .ok_or_else(|| CudaDecodeError::MissingTensor(name.to_owned()))?;
    let kernel = weights
        .gemv_for(weight.value_type())
        .ok_or(CudaDecodeError::MissingKernel(weight.value_type()))?;
    kernel.execute_warp_batch(weight, input, output, members)?;
    Ok(())
}

fn copy_range(
    stream: &Arc<CudaStream>,
    source: &CudaSlice<f32>,
    range: Range<usize>,
    destination: &mut CudaSlice<f32>,
) -> Result<(), CudaDecodeError> {
    let view = source.try_slice(range.clone()).ok_or_else(|| {
        CudaDecodeError::Driver(format!("source slice {range:?} is out of bounds"))
    })?;
    stream
        .memcpy_dtod(&view, destination)
        .map_err(|error| CudaDecodeError::Driver(error.to_string()))
}

// ---------------------------------------------------------------------------
// Batched decode executor: one launch per (layer, op) over all scheduler-batch
// members, batch-major scratch. State families stay per member — each member
// owns its own CudaHybridState, and the per-member state kernels receive
// view-sliced member rows of the batch scratch. The batch-1 executor above
// remains the correctness oracle; this path is opt-in until qualified.

/// Batch-major scratch for `members` concurrent decode rows.
struct BatchScratch {
    tokens: CudaSlice<u32>,
    positions: CudaSlice<i64>,
    hidden: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    attn_incr: CudaSlice<f32>,
    ffn_incr: CudaSlice<f32>,
    qkv_mixed: CudaSlice<f32>,
    z_gate: CudaSlice<f32>,
    beta_raw: CudaSlice<f32>,
    alpha_raw: CudaSlice<f32>,
    decay: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    conv_out: CudaSlice<f32>,
    gdn_q: CudaSlice<f32>,
    gdn_k: CudaSlice<f32>,
    state_out: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    q_raw: CudaSlice<f32>,
    q_packed: CudaSlice<f32>,
    k_raw: CudaSlice<f32>,
    k_normed: CudaSlice<f32>,
    v_raw: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    ffn_gate_buf: CudaSlice<f32>,
    ffn_up_buf: CudaSlice<f32>,
    ffn_act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    selected: CudaSlice<u32>,
    /// Per-member attention scores, `[m][q_heads][stride]`.
    scores: Option<CudaSlice<f32>>,
}

/// Multi-member Qwen3.8 decode executor over shared staged weights.
///
/// `decode_step_batch` advances every member's model state by one token and
/// returns each member's greedy selection in one submission's worth of
/// launches: projections, norms, gates, rope, embedding, and argmax run as
/// single batched kernels; the per-member GDN/attention state kernels run
/// with view-sliced member rows of the batch scratch.
pub struct CudaQwen35BatchDecode {
    stream: Arc<CudaStream>,
    ops: Arc<CudaQwen35Ops>,
    embedding: CudaQ4KEmbedding,
    weights: Arc<CudaQwen35Weights>,
    tensor_names: Arc<[LayerTensorNames]>,
    kv_slot: Vec<u32>,
    recurrent_slot: Vec<u32>,
    epsilon: f32,
    members: usize,
    scratch: BatchScratch,
}

impl CudaQwen35BatchDecode {
    /// Build the batched executor over staged weights for `members` rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] under the same conditions as the batch-1
    /// executor's constructor, plus when `members` is zero or scratch
    /// allocation fails.
    pub fn new(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        weights: Arc<CudaQwen35Weights>,
        layer_kinds: &[QwenLayerKind],
        epsilon: f32,
        members: usize,
    ) -> Result<Self, CudaDecodeError> {
        if members == 0 {
            return Err(CudaDecodeError::InvalidPlan(
                "the batched executor requires at least one member".to_owned(),
            ));
        }
        let vocab = weights
            .quantized_tensor("output.weight")
            .map(|weight| weight.spec().dimensions()[1])
            .ok_or_else(|| CudaDecodeError::MissingTensor("output.weight".to_owned()))
            .and_then(|dim| {
                usize::try_from(dim)
                    .map_err(|_| CudaDecodeError::InvalidPlan("vocabulary does not fit".to_owned()))
            })?;

        let tensor_names: Arc<[LayerTensorNames]> = layer_kinds
            .iter()
            .enumerate()
            .map(|(layer, kind)| LayerTensorNames::new(layer, *kind))
            .collect::<Vec<_>>()
            .into();
        let mut kv_slot = Vec::with_capacity(layer_kinds.len());
        let mut recurrent_slot = Vec::with_capacity(layer_kinds.len());
        let mut kv_count = 0_u32;
        let mut recurrent_count = 0_u32;
        for kind in layer_kinds {
            match kind {
                QwenLayerKind::Recurrent => {
                    recurrent_slot.push(recurrent_count);
                    recurrent_count += 1;
                    kv_slot.push(kv_count);
                }
                QwenLayerKind::FullAttention => {
                    kv_slot.push(kv_count);
                    kv_count += 1;
                    recurrent_slot.push(recurrent_count);
                }
            }
        }

        let ops = Arc::new(CudaQwen35Ops::from_context(context, stream.clone())?);
        let embedding = CudaQ4KEmbedding::from_context(context, stream.clone())?;
        let m = members;
        let alloc = |length: usize| -> Result<CudaSlice<f32>, CudaDecodeError> {
            alloc(&stream, length * m)
        };
        let scratch = BatchScratch {
            tokens: stream
                .alloc_zeros::<u32>(m)
                .map_err(|error| CudaDecodeError::Driver(error.to_string()))?,
            positions: stream
                .alloc_zeros::<i64>(m)
                .map_err(|error| CudaDecodeError::Driver(error.to_string()))?,
            hidden: alloc(N_EMBD)?,
            normed: alloc(N_EMBD)?,
            attn_incr: alloc(N_EMBD)?,
            ffn_incr: alloc(N_EMBD)?,
            qkv_mixed: alloc(GDN_QKV_DIM)?,
            z_gate: alloc(GDN_INNER)?,
            beta_raw: alloc(GDN_V_HEADS)?,
            alpha_raw: alloc(GDN_V_HEADS)?,
            decay: alloc(GDN_V_HEADS)?,
            beta: alloc(GDN_V_HEADS)?,
            conv_out: alloc(GDN_QKV_DIM)?,
            gdn_q: alloc(GDN_K_HEADS * GDN_HEAD_DIM)?,
            gdn_k: alloc(GDN_K_HEADS * GDN_HEAD_DIM)?,
            state_out: alloc(GDN_INNER)?,
            gated: alloc(GDN_INNER)?,
            q_raw: alloc(ATTN_Q_HEADS * 2 * ATTN_HEAD_DIM)?,
            q_packed: alloc(ATTN_Q_HEADS * ATTN_HEAD_DIM)?,
            k_raw: alloc(ATTN_KV_HEADS * ATTN_HEAD_DIM)?,
            k_normed: alloc(ATTN_KV_HEADS * ATTN_HEAD_DIM)?,
            v_raw: alloc(ATTN_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_out: alloc(ATTN_Q_HEADS * ATTN_HEAD_DIM)?,
            ffn_gate_buf: alloc(N_FF)?,
            ffn_up_buf: alloc(N_FF)?,
            ffn_act: alloc(N_FF)?,
            logits: alloc(vocab)?,
            selected: stream
                .alloc_zeros::<u32>(m)
                .map_err(|error| CudaDecodeError::Driver(error.to_string()))?,
            scores: None,
        };

        Ok(Self {
            stream,
            ops,
            embedding,
            weights,
            tensor_names,
            kv_slot,
            recurrent_slot,
            epsilon,
            members,
            scratch,
        })
    }

    /// Number of batch rows this executor was built for.
    #[must_use]
    pub const fn members(&self) -> usize {
        self.members
    }

    /// Upload `[members]` tokens and positions into the batch buffers.
    fn upload_tokens(&mut self, tokens: &[u32], positions: &[u32]) -> Result<(), CudaDecodeError> {
        self.stream
            .memcpy_htod(tokens, &mut self.scratch.tokens)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        let positions: Vec<i64> = positions.iter().map(|&p| i64::from(p)).collect();
        self.stream
            .memcpy_htod(&positions, &mut self.scratch.positions)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        Ok(())
    }

    /// One batched decode step: advance every member's state by its token at
    /// its position and return each member's greedy selection.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when state/geometry is invalid or any
    /// batched launch fails.
    #[allow(
        clippy::too_many_lines,
        reason = "one batched sequence per llama.cpp-verified host step; splitting it hides the step"
    )]
    pub fn decode_step_batch(
        &mut self,
        states: &mut [&mut CudaHybridState],
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<u32>, CudaDecodeError> {
        self.run_step_batch(states, tokens, positions)?;
        let selected = self
            .stream
            .clone_dtoh(&self.scratch.selected)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        Ok(selected)
    }

    /// Enqueue one batched decode step without any host readback.
    ///
    /// All kernels are stream-ordered; the caller copies member selections
    /// out through [`Self::copy_selected_into_pinned`], records one
    /// completion event, and polls it. State positions are not advanced by
    /// this call.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] under the same conditions as
    /// [`Self::decode_step_batch`].
    pub fn decode_step_batch_enqueue(
        &mut self,
        states: &mut [&mut CudaHybridState],
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), CudaDecodeError> {
        self.run_step_batch(states, tokens, positions)
    }

    /// Copy one member's greedy selection from the last batched step into a
    /// pinned host slot, stream-ordered with the step's kernels.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when the member index is out of range or
    /// the copy fails to enqueue.
    pub fn copy_selected_into_pinned(
        &self,
        member: usize,
        destination: &mut PinnedHostSlice<u32>,
    ) -> Result<(), CudaDecodeError> {
        let slot = self
            .scratch
            .selected
            .try_slice(member..=member)
            .ok_or_else(|| {
                CudaDecodeError::Driver("selected token slot out of range".to_owned())
            })?;
        self.stream
            .memcpy_dtoh(&slot, destination)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))
    }

    /// Validation plus all batched launches for one step, shared by the
    /// host-readback and submit-only variants.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when member geometry is invalid, a
    /// required tensor is missing, or any launch fails.
    #[allow(
        clippy::too_many_lines,
        reason = "one batched sequence per llama.cpp-verified host step; splitting it hides the step"
    )]
    fn run_step_batch(
        &mut self,
        states: &mut [&mut CudaHybridState],
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), CudaDecodeError> {
        if states.len() != self.members || tokens.len() != self.members {
            return Err(CudaDecodeError::InvalidPlan(format!(
                "batch step requires {} states and tokens, got {} and {}",
                self.members,
                states.len(),
                tokens.len()
            )));
        }
        if positions.len() != self.members {
            return Err(CudaDecodeError::InvalidPlan(format!(
                "batch step requires {} positions, got {}",
                self.members,
                positions.len()
            )));
        }
        // Validate every member's position against its own state's KV
        // capacity before any launch, mirroring the batch-1 executor's guard.
        for (state, &position) in states.iter().zip(positions) {
            let capacity = state.kv().map_or(u32::MAX, |kv| kv.spec().block_tokens());
            if position >= capacity {
                return Err(CudaDecodeError::PositionOverflow { position, capacity });
            }
        }
        // Lazy score-scratch allocation on the first batched step, once
        // the member states' KV capacity is known. Attention indexing uses
        // each member's own capacity as the scores row stride, so member
        // capacities must agree; mixed-capacity batches would overlap rows.
        if self.scratch.scores.is_none() {
            let mut stride = None;
            for state in states.iter() {
                let Some(kv) = state.kv() else { continue };
                let capacity = kv.spec().block_tokens() as usize;
                if capacity == 0 {
                    return Err(CudaDecodeError::InvalidPlan(
                        "KV state reports a zero token capacity".to_owned(),
                    ));
                }
                if let Some(known) = stride
                    && known != capacity
                {
                    return Err(CudaDecodeError::InvalidPlan(
                        "batched attention requires uniform KV capacity across members".to_owned(),
                    ));
                }
                stride = Some(capacity);
            }
            let stride = stride.unwrap_or(0);
            let scores = alloc(&self.stream, self.members * ATTN_Q_HEADS * stride)
                .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
            self.scratch.scores = Some(scores);
        }
        self.upload_tokens(tokens, positions)?;

        let embedding_weight = self
            .weights
            .quantized_tensor("token_embd.weight")
            .ok_or_else(|| CudaDecodeError::MissingTensor("token_embd.weight".to_owned()))?;
        self.embedding.execute_batch(
            embedding_weight,
            &self.scratch.tokens,
            &mut self.scratch.hidden,
        )?;

        let tensor_names = Arc::clone(&self.tensor_names);
        for (layer, names) in tensor_names.iter().enumerate() {
            self.ops.rms_norm_batch(
                &self.scratch.hidden,
                f32_slice(&self.weights, &names.common.attn_norm)?,
                &mut self.scratch.normed,
                N_EMBD,
                self.epsilon,
            )?;
            match &names.attention {
                AttentionLayerTensorNames::Recurrent(attention) => {
                    self.recurrent_layer_batch(states, layer, attention)?;
                }
                AttentionLayerTensorNames::FullAttention(attention) => {
                    self.full_attention_layer_batch(states, layer, attention, positions)?;
                }
            }
            self.ops.rms_norm_batch(
                &self.scratch.hidden,
                f32_slice(&self.weights, &names.common.post_attention_norm)?,
                &mut self.scratch.normed,
                N_EMBD,
                self.epsilon,
            )?;
            gemv_batch(
                &self.weights,
                &names.common.ffn_gate,
                &self.scratch.normed,
                &mut self.scratch.ffn_gate_buf,
                self.members,
            )?;
            gemv_batch(
                &self.weights,
                &names.common.ffn_up,
                &self.scratch.normed,
                &mut self.scratch.ffn_up_buf,
                self.members,
            )?;
            self.ops.silu_mul(
                &self.scratch.ffn_gate_buf,
                &self.scratch.ffn_up_buf,
                &mut self.scratch.ffn_act,
            )?;
            gemv_batch(
                &self.weights,
                &names.common.ffn_down,
                &self.scratch.ffn_act,
                &mut self.scratch.ffn_incr,
                self.members,
            )?;
            self.ops
                .residual_add(&mut self.scratch.hidden, &self.scratch.ffn_incr)?;
        }

        // Output head over all members, then one batched argmax.
        self.ops.rms_norm_batch(
            &self.scratch.hidden,
            f32_slice(&self.weights, "output_norm.weight")?,
            &mut self.scratch.normed,
            N_EMBD,
            self.epsilon,
        )?;
        gemv_batch(
            &self.weights,
            "output.weight",
            &self.scratch.normed,
            &mut self.scratch.logits,
            self.members,
        )?;
        self.ops
            .argmax_into_batch(&self.scratch.logits, &mut self.scratch.selected)?;
        Ok(())
    }

    /// Batched recurrent (GDN) layer: projections and gates over all members,
    /// then per-member convolution/state kernels on view-sliced rows.
    #[allow(
        clippy::too_many_lines,
        reason = "one batched sequence per llama.cpp-verified host step; splitting it hides the step"
    )]
    fn recurrent_layer_batch(
        &mut self,
        states: &mut [&mut CudaHybridState],
        layer: usize,
        names: &RecurrentLayerTensorNames,
    ) -> Result<(), CudaDecodeError> {
        let m = self.members;
        gemv_batch(
            &self.weights,
            &names.attn_qkv,
            &self.scratch.normed,
            &mut self.scratch.qkv_mixed,
            m,
        )?;
        gemv_batch(
            &self.weights,
            &names.attn_gate,
            &self.scratch.normed,
            &mut self.scratch.z_gate,
            m,
        )?;
        gemv_batch(
            &self.weights,
            &names.ssm_beta,
            &self.scratch.normed,
            &mut self.scratch.beta_raw,
            m,
        )?;
        gemv_batch(
            &self.weights,
            &names.ssm_alpha,
            &self.scratch.normed,
            &mut self.scratch.alpha_raw,
            m,
        )?;
        let dt_bias = f32_slice(&self.weights, &names.ssm_dt_bias)?;
        let ssm_a = f32_slice(&self.weights, &names.ssm_a)?;
        let conv_weight = f32_slice(&self.weights, &names.ssm_conv1d)?;
        let ssm_norm = f32_slice(&self.weights, &names.ssm_norm)?;

        self.ops.gdn_scalar_gate_batch(
            &self.scratch.alpha_raw,
            &self.scratch.beta_raw,
            dt_bias,
            ssm_a,
            &mut self.scratch.decay,
            &mut self.scratch.beta,
            GDN_V_HEADS,
        )?;

        // Per member: conv history + recurrent matrix live in the member's
        // state; scratch rows are sliced views of the batch buffers.
        for member in 0..m {
            let slot = self.recurrent_slot[layer];
            let state = states
                .get_mut(member)
                .ok_or_else(|| CudaDecodeError::InvalidPlan("missing member state".to_owned()))?;
            let recurrent = state.recurrent_mut().ok_or(missing_recurrent())?;
            let (matrix_buffer, conv_buffer) = recurrent.layer_mut(slot).ok_or_else(|| {
                CudaDecodeError::InvalidPlan(format!("recurrent state lacks slot {slot}"))
            })?;
            // Validate the matrix buffer eagerly; the state-update loop below
            // re-borrows it for the actual update.
            let matrix = matrix_buffer.as_f32_mut().ok_or_else(|| {
                CudaDecodeError::InvalidPlan("recurrent matrix must be F32".to_owned())
            })?;
            let _ = matrix.len();
            let history = conv_buffer.as_f32_mut().ok_or_else(|| {
                CudaDecodeError::InvalidPlan("recurrent convolution history must be F32".to_owned())
            })?;

            let qkv = member_row(&self.scratch.qkv_mixed, member, GDN_QKV_DIM)
                .ok_or_else(|| CudaDecodeError::Driver("qkv row out of range".to_owned()))?;
            let mut conv_out = member_row_mut(&mut self.scratch.conv_out, member, GDN_QKV_DIM)
                .ok_or_else(|| CudaDecodeError::Driver("conv_out row out of range".to_owned()))?;
            self.ops
                .gdn_conv_silu_views(&qkv, conv_weight, history, &mut conv_out, GDN_QKV_DIM)?;
        }

        // Q/k normalization in place on each member's conv_out row: q is the
        // [0, K_OFFSET) channel range and k is [K_OFFSET, V_OFFSET) of the
        // member's row, so normalize per member via views (the same math as
        // the batch-1 l2_norm_heads).
        for member in 0..m {
            let start = member * GDN_QKV_DIM;
            let mut conv_q = self
                .scratch
                .conv_out
                .try_slice_mut(start..start + GDN_K_OFFSET)
                .ok_or_else(|| CudaDecodeError::Driver("conv q slice out of range".to_owned()))?;
            self.ops
                .l2_norm_heads_views(&mut conv_q, GDN_K_HEADS, GDN_HEAD_DIM, self.epsilon)?;
            let mut conv_k = self
                .scratch
                .conv_out
                .try_slice_mut(start + GDN_K_OFFSET..start + GDN_V_OFFSET)
                .ok_or_else(|| CudaDecodeError::Driver("conv k slice out of range".to_owned()))?;
            self.ops
                .l2_norm_heads_views(&mut conv_k, GDN_K_HEADS, GDN_HEAD_DIM, self.epsilon)?;
        }

        // Per-member state update with member views of decay/beta/conv_out.
        for member in 0..m {
            let slot = self.recurrent_slot[layer];
            let state = states
                .get_mut(member)
                .ok_or_else(|| CudaDecodeError::InvalidPlan("missing member state".to_owned()))?;
            let recurrent = state.recurrent_mut().ok_or(missing_recurrent())?;
            let (matrix_buffer, _conv_buffer) = recurrent.layer_mut(slot).ok_or_else(|| {
                CudaDecodeError::InvalidPlan(format!("recurrent state lacks slot {slot}"))
            })?;
            let matrix = matrix_buffer.as_f32_mut().ok_or_else(|| {
                CudaDecodeError::InvalidPlan("recurrent matrix must be F32".to_owned())
            })?;

            let q_normed = member_row(&self.scratch.gdn_q, member, GDN_K_HEADS * GDN_HEAD_DIM)
                .ok_or_else(|| CudaDecodeError::Driver("gdn_q row out of range".to_owned()))?;
            let k_normed = member_row(&self.scratch.gdn_k, member, GDN_K_HEADS * GDN_HEAD_DIM)
                .ok_or_else(|| CudaDecodeError::Driver("gdn_k row out of range".to_owned()))?;
            let conv_activated = member_row(&self.scratch.conv_out, member, GDN_QKV_DIM)
                .ok_or_else(|| CudaDecodeError::Driver("conv_out row out of range".to_owned()))?;
            let decay = member_row(&self.scratch.decay, member, GDN_V_HEADS)
                .ok_or_else(|| CudaDecodeError::Driver("decay row out of range".to_owned()))?;
            let beta = member_row(&self.scratch.beta, member, GDN_V_HEADS)
                .ok_or_else(|| CudaDecodeError::Driver("beta row out of range".to_owned()))?;
            let mut state_out = member_row_mut(&mut self.scratch.state_out, member, GDN_INNER)
                .ok_or_else(|| CudaDecodeError::Driver("state_out row out of range".to_owned()))?;
            self.ops.gdn_state_update_views(
                matrix,
                &q_normed,
                &k_normed,
                &conv_activated,
                &decay,
                &beta,
                &mut state_out,
                GDN_V_HEADS,
                GDN_K_HEADS,
                GDN_HEAD_DIM,
                GDN_V_OFFSET,
            )?;
        }

        self.ops.gdn_gated_norm_batch(
            &self.scratch.state_out,
            &self.scratch.z_gate,
            ssm_norm,
            &mut self.scratch.gated,
            GDN_V_HEADS,
            GDN_HEAD_DIM,
            self.epsilon,
        )?;
        gemv_batch(
            &self.weights,
            &names.ssm_out,
            &self.scratch.gated,
            &mut self.scratch.attn_incr,
            m,
        )?;
        self.ops
            .residual_add(&mut self.scratch.hidden, &self.scratch.attn_incr)?;
        Ok(())
    }

    /// Batched full-attention layer: projections/norms/rope over all members,
    /// then per-member KV append and attention against each member's cache.
    #[allow(
        clippy::too_many_lines,
        reason = "one batched sequence per llama.cpp-verified host step; splitting it hides the step"
    )]
    fn full_attention_layer_batch(
        &mut self,
        states: &mut [&mut CudaHybridState],
        layer: usize,
        names: &FullAttentionLayerTensorNames,
        positions: &[u32],
    ) -> Result<(), CudaDecodeError> {
        let m = self.members;
        gemv_batch(
            &self.weights,
            &names.q,
            &self.scratch.normed,
            &mut self.scratch.q_raw,
            m,
        )?;
        let q_norm = f32_slice(&self.weights, &names.q_norm)?;
        self.ops.q_gate_norm_batch(
            &self.scratch.q_raw,
            q_norm,
            &mut self.scratch.q_packed,
            ATTN_Q_HEADS,
            ATTN_HEAD_DIM,
            self.epsilon,
        )?;
        gemv_batch(
            &self.weights,
            &names.k,
            &self.scratch.normed,
            &mut self.scratch.k_raw,
            m,
        )?;
        let k_norm = f32_slice(&self.weights, &names.k_norm)?;
        self.ops.strided_rms_norm_batch(
            &self.scratch.k_raw,
            k_norm,
            &mut self.scratch.k_normed,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.epsilon,
        )?;
        gemv_batch(
            &self.weights,
            &names.v,
            &self.scratch.normed,
            &mut self.scratch.v_raw,
            m,
        )?;
        self.ops.rope_neox_batch(
            &mut self.scratch.q_packed,
            &self.scratch.positions,
            ATTN_Q_HEADS,
            ATTN_HEAD_DIM,
            ATTN_ROT_DIMS,
            ATTN_ROPE_BASE,
        )?;
        self.ops.rope_neox_batch(
            &mut self.scratch.k_normed,
            &self.scratch.positions,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            ATTN_ROT_DIMS,
            ATTN_ROPE_BASE,
        )?;

        // Per member: append to this member's KV cache, then score attention
        // against it, using member views of the batch scratch.
        for (member, &position_u32) in positions.iter().enumerate().take(m) {
            let slot = self.kv_slot[layer];
            let position = usize::try_from(position_u32)
                .map_err(|_| CudaDecodeError::Driver("position overflowed".to_owned()))?;
            let state = states
                .get_mut(member)
                .ok_or_else(|| CudaDecodeError::InvalidPlan("missing member state".to_owned()))?;
            // Read the capacity before borrowing the cache mutably.
            let capacity = state
                .kv()
                .map_or(usize::MAX, |kv| kv.spec().block_tokens() as usize);
            let stride = capacity;
            let tokens = position + 1;
            let kv = state.kv_mut().ok_or(missing_kv())?;
            let (keys_buffer, values_buffer) = kv.layer_mut(slot).ok_or_else(|| {
                CudaDecodeError::InvalidPlan(format!("KV state lacks slot {slot}"))
            })?;
            let cache_keys = keys_buffer
                .as_f16_mut()
                .ok_or_else(|| CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned()))?;
            let cache_values = values_buffer
                .as_f16_mut()
                .ok_or_else(|| CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned()))?;
            let k_member = member_row(
                &self.scratch.k_normed,
                member,
                ATTN_KV_HEADS * ATTN_HEAD_DIM,
            )
            .ok_or_else(|| CudaDecodeError::Driver("k_normed row out of range".to_owned()))?;
            let v_member =
                member_row(&self.scratch.v_raw, member, ATTN_KV_HEADS * ATTN_HEAD_DIM)
                    .ok_or_else(|| CudaDecodeError::Driver("v_raw row out of range".to_owned()))?;
            self.ops.kv_append_f16_views(
                &k_member,
                &v_member,
                cache_keys,
                cache_values,
                position,
                ATTN_KV_HEADS,
                ATTN_HEAD_DIM,
            )?;

            let q_member = member_row(&self.scratch.q_packed, member, ATTN_Q_HEADS * ATTN_HEAD_DIM)
                .ok_or_else(|| CudaDecodeError::Driver("q_packed row out of range".to_owned()))?;
            let gate_member = member_row(
                &self.scratch.q_raw,
                member,
                ATTN_Q_HEADS * 2 * ATTN_HEAD_DIM,
            )
            .ok_or_else(|| CudaDecodeError::Driver("q_raw row out of range".to_owned()))?;
            let scores =
                self.scratch.scores.as_mut().ok_or_else(|| {
                    CudaDecodeError::InvalidPlan("score scratch is unset".to_owned())
                })?;
            let mut scores_member = member_row_mut(scores, member, ATTN_Q_HEADS * stride)
                .ok_or_else(|| CudaDecodeError::Driver("scores row out of range".to_owned()))?;
            let mut out_member = member_row_mut(
                &mut self.scratch.attn_out,
                member,
                ATTN_Q_HEADS * ATTN_HEAD_DIM,
            )
            .ok_or_else(|| CudaDecodeError::Driver("attn_out row out of range".to_owned()))?;
            self.ops.attn_score_gqa_views(
                &q_member,
                cache_keys,
                cache_values,
                &gate_member,
                &mut scores_member,
                &mut out_member,
                tokens,
                stride,
                ATTN_Q_HEADS,
                ATTN_KV_HEADS,
                ATTN_HEAD_DIM,
            )?;
        }

        gemv_batch(
            &self.weights,
            &names.output,
            &self.scratch.attn_out,
            &mut self.scratch.attn_incr,
            m,
        )?;
        self.ops
            .residual_add(&mut self.scratch.hidden, &self.scratch.attn_incr)?;
        Ok(())
    }
}

/// Borrow one member's `[row_length]` row of a batch-major slice as a view.
fn member_row(
    slice: &CudaSlice<f32>,
    member: usize,
    row_length: usize,
) -> Option<CudaView<'_, f32>> {
    let start = member.checked_mul(row_length)?;
    let end = start.checked_add(row_length)?;
    slice.try_slice(start..end)
}

/// Mutably borrow one member's `[row_length]` row of a batch-major slice.
fn member_row_mut(
    slice: &mut CudaSlice<f32>,
    member: usize,
    row_length: usize,
) -> Option<CudaViewMut<'_, f32>> {
    let start = member.checked_mul(row_length)?;
    let end = start.checked_add(row_length)?;
    slice.try_slice_mut(start..end)
}
