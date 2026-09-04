from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


p = Path("crates/nvidia/src/decode.rs")
s = p.read_text()

# Precompute the pinned Qwen decoder's tensor names once at construction time.
# This is intentionally a model-specific execution detail rather than a new IR.
marker = '''/// Batch-1 decode-step runner for the staged Qwen3.8-27B text path.\n'''
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

'''
s = replace_once(s, marker, types + marker, "layer tensor-name plan types")

s = replace_once(
    s,
    '''    layer_kinds: Vec<QwenLayerKind>,\n    kv_slot: Vec<u32>,\n''',
    '''    layer_kinds: Vec<QwenLayerKind>,\n    tensor_names: Arc<[LayerTensorNames]>,\n    kv_slot: Vec<u32>,\n''',
    "decoder tensor-name field",
)

s = replace_once(
    s,
    '''        let mut kv_slot = Vec::with_capacity(layer_kinds.len());\n        let mut recurrent_slot = Vec::with_capacity(layer_kinds.len());\n''',
    '''        let tensor_names: Arc<[LayerTensorNames]> = layer_kinds\n            .iter()\n            .copied()\n            .enumerate()\n            .map(|(layer, kind)| LayerTensorNames::new(layer, kind))\n            .collect::<Vec<_>>()\n            .into();\n        let mut kv_slot = Vec::with_capacity(layer_kinds.len());\n        let mut recurrent_slot = Vec::with_capacity(layer_kinds.len());\n''',
    "construct decoder tensor-name plan",
)

s = replace_once(
    s,
    '''            weights,\n            layer_kinds,\n            kv_slot,\n''',
    '''            weights,\n            layer_kinds,\n            tensor_names,\n            kv_slot,\n''',
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
    '''    fn recurrent_layer(\n        &mut self,\n        state: &mut CudaHybridState,\n        layer: usize,\n        prefix: &str,\n    ) -> Result<(), CudaDecodeError> {\n''',
    '''    fn recurrent_layer(\n        &mut self,\n        state: &mut CudaHybridState,\n        layer: usize,\n        names: &RecurrentLayerTensorNames,\n    ) -> Result<(), CudaDecodeError> {\n''',
    "recurrent layer name-plan signature",
)
for old, new in {
    '&format!("{prefix}attn_qkv.weight")': '&names.attn_qkv',
    '&format!("{prefix}attn_gate.weight")': '&names.attn_gate',
    '&format!("{prefix}ssm_beta.weight")': '&names.ssm_beta',
    '&format!("{prefix}ssm_alpha.weight")': '&names.ssm_alpha',
    '&format!("{prefix}ssm_dt.bias")': '&names.ssm_dt_bias',
    '&format!("{prefix}ssm_a")': '&names.ssm_a',
    '&format!("{prefix}ssm_conv1d.weight")': '&names.ssm_conv1d',
    '&format!("{prefix}ssm_norm.weight")': '&names.ssm_norm',
    '&format!("{prefix}ssm_out.weight")': '&names.ssm_out',
}.items():
    s = replace_once(s, old, new, f"recurrent hot-path {old}")

s = replace_once(
    s,
    '''    fn full_attention_layer(\n        &mut self,\n        state: &mut CudaHybridState,\n        layer: usize,\n        prefix: &str,\n        position: u32,\n    ) -> Result<(), CudaDecodeError> {\n''',
    '''    fn full_attention_layer(\n        &mut self,\n        state: &mut CudaHybridState,\n        layer: usize,\n        names: &FullAttentionLayerTensorNames,\n        position: u32,\n    ) -> Result<(), CudaDecodeError> {\n''',
    "full-attention name-plan signature",
)
for old, new in {
    '&format!("{prefix}attn_q.weight")': '&names.q',
    '&format!("{prefix}attn_q_norm.weight")': '&names.q_norm',
    '&format!("{prefix}attn_k.weight")': '&names.k',
    '&format!("{prefix}attn_k_norm.weight")': '&names.k_norm',
    '&format!("{prefix}attn_v.weight")': '&names.v',
    '&format!("{prefix}attn_output.weight")': '&names.output',
}.items():
    s = replace_once(s, old, new, f"full-attention hot-path {old}")

p.write_text(s)
