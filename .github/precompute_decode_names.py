from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


p = Path("crates/nvidia/src/decode.rs")
s = p.read_text()

# Add a compact load-time name plan. This deliberately remains a Qwen-specific
# decoder detail rather than becoming a general IR: it only removes repeated
# String construction from the steady-state token path.
marker = '''/// Batch-1 decode-step runner for the staged Qwen3.8-27B text path.
'''
types = '''struct CommonLayerTensorNames {
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
    attn_q: String,
    attn_q_norm: String,
    attn_k: String,
    attn_k_norm: String,
    attn_v: String,
    attn_output: String,
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
                    attn_q: format!("{prefix}attn_q.weight"),
                    attn_q_norm: format!("{prefix}attn_q_norm.weight"),
                    attn_k: format!("{prefix}attn_k.weight"),
                    attn_k_norm: format!("{prefix}attn_k_norm.weight"),
                    attn_v: format!("{prefix}attn_v.weight"),
                    attn_output: format!("{prefix}attn_output.weight"),
                })
            }
        };
        Self { common, attention }
    }
}

'''
s = replace_once(s, marker, types + marker, "layer tensor-name plan types")

s = replace_once(
    s,
    '''    layer_kinds: Vec<QwenLayerKind>,
    kv_slot: Vec<u32>,
''',
    '''    layer_kinds: Vec<QwenLayerKind>,
    tensor_names: Arc<[LayerTensorNames]>,
    kv_slot: Vec<u32>,
''',
    "decoder tensor-name field",
)

s = replace_once(
    s,
    '''        let mut kv_slot = Vec::with_capacity(layer_kinds.len());
        let mut recurrent_slot = Vec::with_capacity(layer_kinds.len());
''',
    '''        let tensor_names: Arc<[LayerTensorNames]> = layer_kinds
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, kind)| LayerTensorNames::new(layer, kind))
            .collect::<Vec<_>>()
            .into();
        let mut kv_slot = Vec::with_capacity(layer_kinds.len());
        let mut recurrent_slot = Vec::with_capacity(layer_kinds.len());
''',
    "construct decoder tensor-name plan",
)

s = replace_once(
    s,
    '''            weights,
            layer_kinds,
            kv_slot,
''',
    '''            weights,
            layer_kinds,
            tensor_names,
            kv_slot,
''',
    "initialize decoder tensor-name plan",
)

old_loop = '''        for layer in 0..self.layer_kinds.len() {
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
'''
new_loop = '''        let tensor_names = Arc::clone(&self.tensor_names);
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
            )?;
            gemv(
                &self.weights,
                &names.common.ffn_up,
                &self.normed,
                &mut self.ffn_up_buf,
            )?;
            self.ops
                .silu_mul(&self.ffn_gate_buf, &self.ffn_up_buf, &mut self.ffn_act)?;
            gemv(
                &self.weights,
                &names.common.ffn_down,
                &self.ffn_act,
                &mut self.ffn_incr,
            )?;
            self.ops.residual_add(&mut self.hidden, &self.ffn_incr)?;
        }
'''
s = replace_once(s, old_loop, new_loop, "allocation-free decode layer-name loop")

s = replace_once(
    s,
    '''    fn recurrent_layer(
        &mut self,
        state: &mut CudaHybridState,
        layer: usize,
        prefix: &str,
    ) -> Result<(), CudaDecodeError> {
''',
    '''    fn recurrent_layer(
        &mut self,
        state: &mut CudaHybridState,
        layer: usize,
        names: &RecurrentLayerTensorNames,
    ) -> Result<(), CudaDecodeError> {
''',
    "recurrent layer name-plan signature",
)
recurrent_replacements = {
    '&format!("{prefix}attn_qkv.weight")': '&names.attn_qkv',
    '&format!("{prefix}attn_gate.weight")': '&names.attn_gate',
    '&format!("{prefix}ssm_beta.weight")': '&names.ssm_beta',
    '&format!("{prefix}ssm_alpha.weight")': '&names.ssm_alpha',
    '&format!("{prefix}ssm_dt.bias")': '&names.ssm_dt_bias',
    '&format!("{prefix}ssm_a")': '&names.ssm_a',
    '&format!("{prefix}ssm_conv1d.weight")': '&names.ssm_conv1d',
    '&format!("{prefix}ssm_norm.weight")': '&names.ssm_norm',
    '&format!("{prefix}ssm_out.weight")': '&names.ssm_out',
}
for old, new in recurrent_replacements.items():
    if old not in s:
        raise SystemExit(f"recurrent hot-path target not found: {old}")
    s = s.replace(old, new, 1)

s = replace_once(
    s,
    '''    fn full_attention_layer(
        &mut self,
        state: &mut CudaHybridState,
        layer: usize,
        prefix: &str,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
''',
    '''    fn full_attention_layer(
        &mut self,
        state: &mut CudaHybridState,
        layer: usize,
        names: &FullAttentionLayerTensorNames,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
''',
    "full-attention name-plan signature",
)
full_replacements = {
    '&format!("{prefix}attn_q.weight")': '&names.attn_q',
    '&format!("{prefix}attn_q_norm.weight")': '&names.attn_q_norm',
    '&format!("{prefix}attn_k.weight")': '&names.attn_k',
    '&format!("{prefix}attn_k_norm.weight")': '&names.attn_k_norm',
    '&format!("{prefix}attn_v.weight")': '&names.attn_v',
    '&format!("{prefix}attn_output.weight")': '&names.attn_output',
}
for old, new in full_replacements.items():
    if old not in s:
        raise SystemExit(f"full-attention hot-path target not found: {old}")
    s = s.replace(old, new, 1)

# Guard the intent with a source-level unit test that can run without CUDA.
# It ensures all hot-path tensor names are materialized once and stable.
marker = '''/// The state matrix and history access for one recurrent layer slot.
'''
helper = '''#[cfg(test)]
mod tensor_name_tests {
    use super::*;

    #[test]
    fn layer_tensor_names_are_materialized_from_layer_identity() {
        let recurrent = LayerTensorNames::new(7, QwenLayerKind::Recurrent);
        assert_eq!(recurrent.common.attn_norm, "blk.7.attn_norm.weight");
        let AttentionLayerTensorNames::Recurrent(attention) = recurrent.attention else {
            panic!("expected recurrent names");
        };
        assert_eq!(attention.ssm_out, "blk.7.ssm_out.weight");

        let full = LayerTensorNames::new(11, QwenLayerKind::FullAttention);
        assert_eq!(full.common.ffn_down, "blk.11.ffn_down.weight");
        let AttentionLayerTensorNames::FullAttention(attention) = full.attention else {
            panic!("expected full-attention names");
        };
        assert_eq!(attention.attn_output, "blk.11.attn_output.weight");
    }
}

'''
s = replace_once(s, marker, helper + marker, "tensor-name plan test")

p.write_text(s)
