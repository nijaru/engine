from pathlib import Path


decode = Path("crates/nvidia/src/decode.rs")
text = decode.read_text()

public_anchor = '''    /// Validation plus all batched launches for one step, shared by the
    /// host-readback and submit-only variants.
'''
if public_anchor not in text:
    raise SystemExit("batch run-step anchor not found")

public_method = '''    /// Advance one sequence through a small known prompt chunk without an output head.
    ///
    /// The fixed batch rows represent consecutive tokens of the **same** sequence,
    /// not independent requests. Token-independent projections, norms, and FFNs run
    /// batch-major; convolution, recurrent-state updates, KV append, and causal
    /// attention remain ordered by token inside each layer. This is an experimental
    /// prefill path for qualification against the batch-1 oracle and is not selected
    /// by serving automatically.
    ///
    /// Physical state buffers are updated, but the mirrored prefix in
    /// [`CudaHybridState`] is not committed here, matching [`CudaQwen35Decode::prefill_step`].
    /// The caller may advance the prefix only after completion is established.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when the chunk length does not match this
    /// executor's fixed row count, positions overflow, state/geometry is invalid,
    /// or any CUDA launch fails. A launch failure can leave physical state partially
    /// updated; callers must treat that state as uncertain rather than rolling it back.
    pub fn prefill_chunk(
        &mut self,
        state: &mut CudaHybridState,
        tokens: &[u32],
        start_position: u32,
    ) -> Result<(), CudaDecodeError> {
        if tokens.len() != self.members {
            return Err(CudaDecodeError::InvalidPlan(format!(
                "prefill chunk requires {} tokens, got {}",
                self.members,
                tokens.len()
            )));
        }
        let positions = (0..self.members)
            .map(|offset| {
                let offset = u32::try_from(offset).map_err(|_| {
                    CudaDecodeError::InvalidPlan("prefill chunk offset overflowed u32".to_owned())
                })?;
                start_position.checked_add(offset).ok_or_else(|| {
                    CudaDecodeError::InvalidPlan("prefill chunk position overflowed u32".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.run_prefill_chunk(state, tokens, &positions)
    }

'''
text = text.replace(public_anchor, public_method + public_anchor, 1)

private_anchor = '''    /// Batched recurrent (GDN) layer: projections and gates over all members,
'''
if private_anchor not in text:
    raise SystemExit("batched recurrent anchor not found")

private_methods = '''    /// Run one same-sequence prompt chunk. This deliberately keeps stateful
    /// operations serial within each layer while amortizing the surrounding
    /// weight-heavy operations across prompt-token rows.
    #[allow(
        clippy::too_many_lines,
        reason = "the experimental prefill path mirrors the layer sequence while keeping causal state updates explicit"
    )]
    fn run_prefill_chunk(
        &mut self,
        state: &mut CudaHybridState,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), CudaDecodeError> {
        if tokens.len() != self.members || positions.len() != self.members {
            return Err(CudaDecodeError::InvalidPlan(format!(
                "prefill chunk requires {} tokens and positions, got {} and {}",
                self.members,
                tokens.len(),
                positions.len()
            )));
        }
        if self.gemv_mode == GemvMode::IntegerDot && self.int_dot.is_none() {
            self.int_dot = Some(CudaIntDotProjector::new(self.stream.clone())?);
        }
        validate_state(self.stream.context(), &self.layer_kinds, state)?;
        for &token in tokens {
            validate_token(token, self.scratch.logits.len() / self.members)?;
        }

        if let Some(kv) = state.kv() {
            let capacity = kv.spec().block_tokens();
            if let Some(&position) = positions.last()
                && position >= capacity
            {
                return Err(CudaDecodeError::PositionOverflow { position, capacity });
            }
            let capacity = capacity as usize;
            let required = attention_scratch_elements(
                self.members,
                std::iter::repeat_n(capacity, self.members),
            )?;
            if self
                .scratch
                .scores
                .as_ref()
                .is_none_or(|scores| scores.len() < required)
            {
                self.scratch.scores = Some(alloc(&self.stream, required)?);
            }
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
                    self.recurrent_layer_prefill_chunk(state, layer, attention)?;
                }
                AttentionLayerTensorNames::FullAttention(attention) => {
                    self.full_attention_layer_prefill_chunk(state, layer, attention, positions)?;
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
                N_EMBD,
                self.gemv_mode,
                self.int_dot.as_mut(),
            )?;
            gemv_batch(
                &self.weights,
                &names.common.ffn_up,
                &self.scratch.normed,
                &mut self.scratch.ffn_up_buf,
                self.members,
                N_EMBD,
                self.gemv_mode,
                self.int_dot.as_mut(),
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
                N_FF,
                self.gemv_mode,
                self.int_dot.as_mut(),
            )?;
            self.ops
                .residual_add(&mut self.scratch.hidden, &self.scratch.ffn_incr)?;
        }
        Ok(())
    }

    /// Same-sequence recurrent layer for a prompt chunk. Projection and norm
    /// work is batched, but convolution history and the GDN matrix are advanced
    /// token-by-token in prompt order so recurrence is never treated as an
    /// independent batch dimension.
    #[allow(
        clippy::too_many_lines,
        reason = "one causal GDN chunk sequence over batched projection scratch"
    )]
    fn recurrent_layer_prefill_chunk(
        &mut self,
        state: &mut CudaHybridState,
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
            N_EMBD,
            self.gemv_mode,
            self.int_dot.as_mut(),
        )?;
        gemv_batch(
            &self.weights,
            &names.attn_gate,
            &self.scratch.normed,
            &mut self.scratch.z_gate,
            m,
            N_EMBD,
            self.gemv_mode,
            self.int_dot.as_mut(),
        )?;
        gemv_batch(
            &self.weights,
            &names.ssm_beta,
            &self.scratch.normed,
            &mut self.scratch.beta_raw,
            m,
            N_EMBD,
            self.gemv_mode,
            self.int_dot.as_mut(),
        )?;
        gemv_batch(
            &self.weights,
            &names.ssm_alpha,
            &self.scratch.normed,
            &mut self.scratch.alpha_raw,
            m,
            N_EMBD,
            self.gemv_mode,
            self.int_dot.as_mut(),
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

        // Convolution history is one sequence-local recurrence. Produce each
        // token row in order before normalizing q/k in a single batch launch.
        {
            let slot = self.recurrent_slot[layer];
            let recurrent = state.recurrent_mut().ok_or(missing_recurrent())?;
            let (_matrix_buffer, conv_buffer) = recurrent.layer_mut(slot).ok_or_else(|| {
                CudaDecodeError::InvalidPlan(format!("recurrent state lacks slot {slot}"))
            })?;
            let history = conv_buffer.as_f32_mut().ok_or_else(|| {
                CudaDecodeError::InvalidPlan("recurrent convolution history must be F32".to_owned())
            })?;
            for member in 0..m {
                let qkv = member_row(&self.scratch.qkv_mixed, member, GDN_QKV_DIM)
                    .ok_or_else(|| CudaDecodeError::Driver("qkv row out of range".to_owned()))?;
                let mut conv_out = member_row_mut(&mut self.scratch.conv_out, member, GDN_QKV_DIM)
                    .ok_or_else(|| {
                        CudaDecodeError::Driver("conv_out row out of range".to_owned())
                    })?;
                self.ops.gdn_conv_silu_views(
                    &qkv,
                    conv_weight,
                    history,
                    &mut conv_out,
                    GDN_QKV_DIM,
                )?;
            }
        }

        for member in 0..m {
            let row = member.checked_mul(GDN_QKV_DIM).ok_or_else(|| {
                CudaDecodeError::Driver("GDN batch row offset overflowed".to_owned())
            })?;
            let conv_q = self
                .scratch
                .conv_out
                .try_slice(row..row + GDN_K_OFFSET)
                .ok_or_else(|| CudaDecodeError::Driver("conv q slice out of range".to_owned()))?;
            let mut gdn_q = member_row_mut(&mut self.scratch.gdn_q, member, GDN_K_OFFSET)
                .ok_or_else(|| CudaDecodeError::Driver("gdn_q row out of range".to_owned()))?;
            self.stream
                .memcpy_dtod(&conv_q, &mut gdn_q)
                .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
            let conv_k = self
                .scratch
                .conv_out
                .try_slice(row + GDN_K_OFFSET..row + GDN_V_OFFSET)
                .ok_or_else(|| CudaDecodeError::Driver("conv k slice out of range".to_owned()))?;
            let mut gdn_k = member_row_mut(&mut self.scratch.gdn_k, member, GDN_K_OFFSET)
                .ok_or_else(|| CudaDecodeError::Driver("gdn_k row out of range".to_owned()))?;
            self.stream
                .memcpy_dtod(&conv_k, &mut gdn_k)
                .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        }
        self.ops.l2_norm_heads_batch(
            &mut self.scratch.gdn_q,
            GDN_K_HEADS,
            GDN_HEAD_DIM,
            self.epsilon,
        )?;
        self.ops.l2_norm_heads_batch(
            &mut self.scratch.gdn_k,
            GDN_K_HEADS,
            GDN_HEAD_DIM,
            self.epsilon,
        )?;

        // The recurrent matrix is also one sequence-local recurrence. Each
        // state update consumes the matrix produced by the previous prompt row.
        {
            let slot = self.recurrent_slot[layer];
            let recurrent = state.recurrent_mut().ok_or(missing_recurrent())?;
            let (matrix_buffer, _conv_buffer) = recurrent.layer_mut(slot).ok_or_else(|| {
                CudaDecodeError::InvalidPlan(format!("recurrent state lacks slot {slot}"))
            })?;
            let matrix = matrix_buffer.as_f32_mut().ok_or_else(|| {
                CudaDecodeError::InvalidPlan("recurrent matrix must be F32".to_owned())
            })?;
            for member in 0..m {
                let q = member_row(&self.scratch.gdn_q, member, GDN_K_OFFSET)
                    .ok_or_else(|| CudaDecodeError::Driver("gdn_q row out of range".to_owned()))?;
                let k = member_row(&self.scratch.gdn_k, member, GDN_K_OFFSET)
                    .ok_or_else(|| CudaDecodeError::Driver("gdn_k row out of range".to_owned()))?;
                let conv = member_row(&self.scratch.conv_out, member, GDN_QKV_DIM)
                    .ok_or_else(|| CudaDecodeError::Driver("conv row out of range".to_owned()))?;
                let decay = member_row(&self.scratch.decay, member, GDN_V_HEADS)
                    .ok_or_else(|| CudaDecodeError::Driver("decay row out of range".to_owned()))?;
                let beta = member_row(&self.scratch.beta, member, GDN_V_HEADS)
                    .ok_or_else(|| CudaDecodeError::Driver("beta row out of range".to_owned()))?;
                let mut output = member_row_mut(&mut self.scratch.state_out, member, GDN_INNER)
                    .ok_or_else(|| {
                        CudaDecodeError::Driver("state output row out of range".to_owned())
                    })?;
                self.ops.gdn_state_update_views(
                    matrix,
                    &q,
                    &k,
                    &conv,
                    &decay,
                    &beta,
                    &mut output,
                    GDN_V_HEADS,
                    GDN_K_HEADS,
                    GDN_HEAD_DIM,
                    GDN_V_OFFSET,
                )?;
            }
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
            GDN_INNER,
            self.gemv_mode,
            self.int_dot.as_mut(),
        )?;
        self.ops
            .residual_add(&mut self.scratch.hidden, &self.scratch.attn_incr)?;
        Ok(())
    }

    /// Same-sequence full-attention layer for a prompt chunk. Q/K/V and RoPE
    /// run batch-major, then rows append and attend in order so each query sees
    /// the prefix plus earlier rows from this chunk and never later rows.
    #[allow(
        clippy::too_many_lines,
        reason = "one causal attention chunk sequence over batched projection scratch"
    )]
    fn full_attention_layer_prefill_chunk(
        &mut self,
        state: &mut CudaHybridState,
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
            N_EMBD,
            self.gemv_mode,
            self.int_dot.as_mut(),
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
            N_EMBD,
            self.gemv_mode,
            self.int_dot.as_mut(),
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
            N_EMBD,
            self.gemv_mode,
            self.int_dot.as_mut(),
        )?;
        self.ops.rope_neox_batch(
            &mut self.scratch.q_packed,
            &self.scratch.positions,
            ATTN_Q_HEADS,
            ATTN_HEAD_DIM,
            ATTN_ROT_DIMS / 2,
            ATTN_ROPE_BASE,
        )?;
        self.ops.rope_neox_batch(
            &mut self.scratch.k_normed,
            &self.scratch.positions,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            ATTN_ROT_DIMS / 2,
            ATTN_ROPE_BASE,
        )?;

        let capacity = state
            .kv()
            .ok_or(missing_kv())?
            .spec()
            .block_tokens() as usize;
        let stride = capacity;
        for (member, &position_u32) in positions.iter().enumerate() {
            let position = usize::try_from(position_u32)
                .map_err(|_| CudaDecodeError::Driver("position overflowed usize".to_owned()))?;
            let tokens = position.checked_add(1).ok_or_else(|| {
                CudaDecodeError::Driver("attention token count overflowed".to_owned())
            })?;
            let slot = self.kv_slot[layer];
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
            let v_member = member_row(
                &self.scratch.v_raw,
                member,
                ATTN_KV_HEADS * ATTN_HEAD_DIM,
            )
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

            let q_member = member_row(
                &self.scratch.q_packed,
                member,
                ATTN_Q_HEADS * ATTN_HEAD_DIM,
            )
            .ok_or_else(|| CudaDecodeError::Driver("q_packed row out of range".to_owned()))?;
            let gate_member = member_row(
                &self.scratch.q_raw,
                member,
                ATTN_Q_HEADS * 2 * ATTN_HEAD_DIM,
            )
            .ok_or_else(|| CudaDecodeError::Driver("q_raw row out of range".to_owned()))?;
            let scores = self.scratch.scores.as_mut().ok_or_else(|| {
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
            ATTN_Q_HEADS * ATTN_HEAD_DIM,
            self.gemv_mode,
            self.int_dot.as_mut(),
        )?;
        self.ops
            .residual_add(&mut self.scratch.hidden, &self.scratch.attn_incr)?;
        Ok(())
    }

'''
text = text.replace(private_anchor, private_methods + private_anchor, 1)
decode.write_text(text)


test = Path("crates/nvidia/tests/cuda_reference.rs")
test_text = test.read_text()
test_anchor = '''/// Bisect the batched divergence by real-weight prefix: plans [R], [R,R],
'''
if test_anchor not in test_text:
    raise SystemExit("CUDA reference insertion anchor not found")

new_test = r'''#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
#[allow(
    clippy::too_many_lines,
    reason = "one real-weight hybrid-prefix parity gate between token-major and chunk-prefill execution"
)]
fn same_sequence_prefill_chunk_matches_batch1_hybrid_prefix() {
    use engine_core::{
        ConvolutionStateShape, DataType, KvStateSpec, RecurrentMatrixShape, RecurrentStateSpec,
    };
    use engine_nvidia::{
        CudaQwen35BatchDecode, CudaQwen35Decode, CudaQwen35Weights, QwenLayerKind,
        StagedTensorSource,
    };
    use std::sync::Arc;

    const GGUF: &str = "/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf";
    const EPS: f32 = 1.0e-6;
    const TOKENS: [u32; 4] = [12_675, 1017, 760, 6511];
    const NEXT_TOKEN: u32 = 42;
    const LAYERS: usize = 4;
    const TOLERANCE: f32 = 5.0e-3;

    let provider = QwenGguf::open(GGUF).expect("open pinned Qwen GGUF");
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let device = DeviceId::new(0);

    let mut names: Vec<String> = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
        "output.weight".to_owned(),
    ];
    for layer in 0..LAYERS {
        let binding = provider
            .layer_weight_binding(device, u32::try_from(layer).expect("layer fits u32"))
            .expect("layer binding");
        for spec in binding.tensors() {
            names.push(spec.name().to_owned());
        }
    }
    let tensors: Vec<StagedTensorSource> = names
        .iter()
        .map(|name| {
            let reader = provider.open_tensor(name).expect("open tensor");
            let spec = reader.spec().clone();
            let value_type = reader.value_type();
            let encoded_bytes = reader.remaining();
            if matches!(value_type, 0 | 1) {
                let blocks = engine_nvidia::wrap_f32_stream(reader);
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(std::io::empty()),
                    f32_blocks: Some(Box::new(blocks)),
                }
            } else {
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(reader),
                    f32_blocks: None,
                }
            }
        })
        .collect();
    let staged = Arc::new(
        CudaQwen35Weights::stage(&context, &stream, 4_u64 << 30, tensors)
            .expect("stage the four-layer prefix"),
    );
    let layer_kinds = [QwenLayerKind::Recurrent; 3]
        .into_iter()
        .chain([QwenLayerKind::FullAttention])
        .collect::<Vec<_>>();
    let kv_spec = KvStateSpec::new(1, 4, 256, 8, DataType::F16).expect("KV spec");
    let recurrent_spec = RecurrentStateSpec::new(
        3,
        RecurrentMatrixShape::new(48, 128, 128).expect("matrix shape"),
        ConvolutionStateShape::new(10_240, 3).expect("convolution shape"),
        DataType::F32,
        DataType::F32,
    )
    .expect("recurrent spec");
    let fresh_state = || {
        let mut state = CudaHybridState::from_specs(
            stream.clone(),
            Some(kv_spec),
            Some(recurrent_spec),
        )
        .expect("physical hybrid state");
        state.zero().expect("zero state");
        state
    };

    let mut oracle = CudaQwen35Decode::new(
        &context,
        stream.clone(),
        Arc::clone(&staged),
        layer_kinds.clone(),
        EPS,
    )
    .expect("batch-1 oracle");
    let mut oracle_state = fresh_state();
    let mut oracle_hidden = Vec::with_capacity(TOKENS.len());
    for (position, &token) in TOKENS.iter().enumerate() {
        let position = u32::try_from(position).expect("position fits u32");
        oracle
            .prefill_step(&mut oracle_state, token, position)
            .expect("oracle prefill");
        oracle_state
            .advance_to(position + 1)
            .expect("oracle advance");
        oracle_hidden.push(oracle.copy_hidden().expect("oracle hidden"));
    }

    let mut candidate_single = CudaQwen35Decode::new(
        &context,
        stream.clone(),
        Arc::clone(&staged),
        layer_kinds,
        EPS,
    )
    .expect("candidate single executor");
    let mut chunk = CudaQwen35BatchDecode::from_decode(&candidate_single, TOKENS.len())
        .expect("chunk executor");
    let mut chunk_state = fresh_state();
    chunk
        .prefill_chunk(&mut chunk_state, &TOKENS, 0)
        .expect("same-sequence prefill chunk");
    chunk_state
        .advance_to(u32::try_from(TOKENS.len()).expect("chunk length fits u32"))
        .expect("chunk advance");

    for (member, expected) in oracle_hidden.iter().enumerate() {
        let actual = chunk
            .copy_hidden_member(member)
            .expect("chunk hidden row");
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < TOLERANCE,
            "chunk residual diverged at prompt row {member}: max abs {max_abs}"
        );
    }

    let next_position = u32::try_from(TOKENS.len()).expect("chunk length fits u32");
    oracle
        .prefill_step(&mut oracle_state, NEXT_TOKEN, next_position)
        .expect("oracle continuation");
    candidate_single
        .prefill_step(&mut chunk_state, NEXT_TOKEN, next_position)
        .expect("chunk continuation");
    let oracle_next = oracle.copy_hidden().expect("oracle continuation hidden");
    let chunk_next = candidate_single
        .copy_hidden()
        .expect("chunk continuation hidden");
    let max_abs = oracle_next
        .iter()
        .zip(&chunk_next)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs < TOLERANCE,
        "chunk continuation state diverged: max abs {max_abs}"
    );
}

'''
test_text = test_text.replace(test_anchor, new_test + test_anchor, 1)
test.write_text(test_text)
