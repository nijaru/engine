//! Decode-step executor for the pinned Qwen3.8-27B text path.
//!
//! Assembles one batch-1, host-driven, greedy-first decode step over the
//! staged encoded weights: embedding lookup, per-layer GDN/full-attention
//! sequencing across [`CudaHybridState`], FFN, and the greedy output head.
//! Layer sequencing mirrors `host_gdn_ar_step`/`host_full_attn_ar_step`,
//! which are llama.cpp-verified at `cc83d7b48`; each launch is the
//! corresponding parity-tested CUDA kernel. Prefill runs this same loop
//! autoregressively, one token at a time, exactly as the host reference
//! does.
//!
//! The executor defines its own [`QwenLayerKind`] so `engine-gguf` remains
//! optional; model providers translate their layer catalogs into this
//! data-only plan.

use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
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

/// Batch-1 decode-step runner for the staged Qwen3.8-27B text path.
///
/// The runner owns scratch buffers sized for the pinned geometry and
/// sequences one token's full forward pass: embed → layer blocks (GDN or
/// full attention by [`QwenLayerKind`]) → greedy output head. The caller
/// owns the physical state, the token loop, and positions; `decode_step`
/// returns the greedy token choice for the input `token` at
/// `position`.
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
    /// Attention score scratch, `[q_heads][token capacity]`, allocated on
    /// first use once the KV capacity is known.
    scores: Option<CudaSlice<f32>>,
    scores_stride: usize,
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

        Ok(Self {
            stream,
            ops,
            embedding,
            weights,
            layer_kinds,
            kv_slot,
            recurrent_slot,
            epsilon,
            scores: None,
            scores_stride: 0,
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
        })
    }

    /// Run one full forward pass for `token` at absolute sequence
    /// `position` and return the greedy token choice.
    ///
    /// All kernel launches are stream-ordered; the returned token comes
    /// from a synchronized argmax, so state mutations from this step are
    /// visible to subsequent steps.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when the physical state does not match
    /// the layer plan geometry, the position exceeds the KV capacity, or
    /// any launch fails.
    pub fn decode_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<u32, CudaDecodeError> {
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

        for layer in 0..self.layer_kinds.len() {
            let prefix = format!("blk.{layer}.");
            self.ops.rms_norm(
                &self.hidden,
                f32_slice(&self.weights, &format!("{prefix}attn_norm.weight"))?,
                &mut self.normed,
                self.epsilon,
            )?;
            match self.layer_kinds[layer] {
                QwenLayerKind::Recurrent => {
                    self.recurrent_layer(state, layer, &prefix)?;
                }
                QwenLayerKind::FullAttention => {
                    self.full_attention_layer(state, layer, &prefix, position)?;
                }
            }
            self.ops.rms_norm(
                &self.hidden,
                f32_slice(
                    &self.weights,
                    &format!("{prefix}post_attention_norm.weight"),
                )?,
                &mut self.normed,
                self.epsilon,
            )?;
            gemv(
                &self.weights,
                &format!("{prefix}ffn_gate.weight"),
                &self.normed,
                &mut self.ffn_gate_buf,
            )?;
            gemv(
                &self.weights,
                &format!("{prefix}ffn_up.weight"),
                &self.normed,
                &mut self.ffn_up_buf,
            )?;
            self.ops
                .silu_mul(&self.ffn_gate_buf, &self.ffn_up_buf, &mut self.ffn_act)?;
            gemv(
                &self.weights,
                &format!("{prefix}ffn_down.weight"),
                &self.ffn_act,
                &mut self.ffn_incr,
            )?;
            self.ops.residual_add(&mut self.hidden, &self.ffn_incr)?;
        }

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
        )?;
        Ok(self.ops.argmax(&self.logits)?)
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

    fn recurrent_layer(
        &mut self,
        state: &mut CudaHybridState,
        layer: usize,
        prefix: &str,
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
            &format!("{prefix}attn_qkv.weight"),
            &self.normed,
            &mut self.qkv_mixed,
        )?;
        gemv(
            &self.weights,
            &format!("{prefix}attn_gate.weight"),
            &self.normed,
            &mut self.z_gate,
        )?;
        gemv(
            &self.weights,
            &format!("{prefix}ssm_beta.weight"),
            &self.normed,
            &mut self.beta_raw,
        )?;
        gemv(
            &self.weights,
            &format!("{prefix}ssm_alpha.weight"),
            &self.normed,
            &mut self.alpha_raw,
        )?;
        let dt_bias = f32_slice(&self.weights, &format!("{prefix}ssm_dt.bias"))?;
        let ssm_a = f32_slice(&self.weights, &format!("{prefix}ssm_a"))?;
        let conv_weight = f32_slice(&self.weights, &format!("{prefix}ssm_conv1d.weight"))?;
        let ssm_norm = f32_slice(&self.weights, &format!("{prefix}ssm_norm.weight"))?;

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
            &format!("{prefix}ssm_out.weight"),
            &self.gated,
            &mut self.attn_incr,
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
        prefix: &str,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
        let slot = self.kv_slot[layer];
        gemv(
            &self.weights,
            &format!("{prefix}attn_q.weight"),
            &self.normed,
            &mut self.q_raw,
        )?;
        let q_norm = f32_slice(&self.weights, &format!("{prefix}attn_q_norm.weight"))?;
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
            &format!("{prefix}attn_k.weight"),
            &self.normed,
            &mut self.k_raw,
        )?;
        let k_norm = f32_slice(&self.weights, &format!("{prefix}attn_k_norm.weight"))?;
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
            &format!("{prefix}attn_v.weight"),
            &self.normed,
            &mut self.v_raw,
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
            &format!("{prefix}attn_output.weight"),
            &self.attn_out,
            &mut self.attn_incr,
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
) -> Result<(), CudaDecodeError> {
    let weight = weights
        .quantized_tensor(name)
        .ok_or_else(|| CudaDecodeError::MissingTensor(name.to_owned()))?;
    let kernel = weights
        .gemv_for(weight.value_type())
        .ok_or(CudaDecodeError::MissingKernel(weight.value_type()))?;
    kernel.execute(weight, input, output)?;
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
