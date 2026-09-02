//! Host-only reference implementation of the Qwen3.8-27B decoder layer math,
//! pinned from llama.cpp `cc83d7b48` (see
//! `ai/research/qwen35-forward-semantics-llama-cpp-2026-09-02.md`).
//!
//! This module exists to cross-validate Engine's understanding of the model
//! semantics before any CUDA dispatcher work: it consumes dequantized GGUF
//! weights on the host and reproduces one decode step with plain equations.
//! It makes no performance claim and is not an execution path.

/// Qwen3.8-27B GDN layer geometry, verified against the pinned artifact.
pub const GDN_K_HEADS: usize = 16;
pub const GDN_V_HEADS: usize = 48;
pub const GDN_HEAD_DIM: usize = 128;
pub const GDN_D_CONV: usize = 4;
const GDN_HEAD_DIM_U16: u16 = 128;
const GDN_K_OFFSET: usize = 2048;
const GDN_V_OFFSET: usize = 4096;
/// Total conv-channel width: q 2048 | k 2048 | v 6144.
pub const GDN_QKV_DIM: usize = 10240;
pub const GDN_INNER: usize = 6144;
pub const N_EMBD: usize = 5120;

/// ggml `mul_mat` GEMV against a GGUF `[in_dim, out_dim]` tensor stored flat
/// column-major over `ne0 = in_dim`: column `n` is `values[n*in_dim..(n+1)*in_dim]`.
///
/// # Panics
///
/// Panics when `x` does not match `in_dim` or the weight length is not a
/// multiple of `in_dim`.
#[must_use]
pub fn gguf_gemv(weight: &[f32], in_dim: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), in_dim);
    assert!(weight.len().is_multiple_of(in_dim));
    (0..weight.len() / in_dim)
        .map(|n| {
            let column = &weight[n * in_dim..(n + 1) * in_dim];
            x.iter().zip(column).map(|(xv, wv)| xv * wv).sum()
        })
        .collect()
}

/// Per-head l2 normalization with an eps floor on the norm, matching
/// `ggml_compute_forward_l2_norm_f32` (`1 / max(sqrt(sum), eps)`).
#[must_use]
pub fn l2_normalize(values: &[f32], eps: f32) -> Vec<f32> {
    let sum: f32 = values.iter().map(|x| x * x).sum();
    let scale = 1.0 / sum.sqrt().max(eps);
    values.iter().map(|x| x * scale).collect()
}

/// softplus with the ggml-compatible large-input shortcut.
#[must_use]
pub fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

/// Column-oriented recurrent state update for one V head, matching
/// `ggml_compute_forward_gated_delta_net_one_chunk` (fused AR) and
/// `build_delta_net_autoregressive` (non-fused):
/// `sk = S^T k`, `d = (v - sk) * beta`, `S += k (outer) d`,
/// `o = (S^T q) / sqrt(head_dim)` — llama.cpp scales `q` by `1/sqrt(S_v)`
/// before the state read (non-fused) or the output (fused); both are the
/// same scalar and neither affects the state update itself.
/// `S` is row-major `[head_dim][head_dim]`. Names match the pinned equations.
#[allow(clippy::many_single_char_names)]
fn gdn_state_step(
    state: &mut [f32],
    k: &[f32],
    v: &[f32],
    q: &[f32],
    decay: f32,
    beta: f32,
) -> Vec<f32> {
    let head_dim = GDN_HEAD_DIM;
    for state_elem in state.iter_mut() {
        *state_elem *= decay;
    }
    let mut sk = vec![0.0_f32; head_dim];
    for (row, state_row) in state.chunks_exact(head_dim).enumerate() {
        for (col, state_elem) in state_row.iter().enumerate() {
            sk[col] += state_elem * k[row];
        }
    }
    let d: Vec<f32> = (0..head_dim).map(|col| (v[col] - sk[col]) * beta).collect();
    for (row, state_row) in state.chunks_exact_mut(head_dim).enumerate() {
        for (state_elem, d_elem) in state_row.iter_mut().zip(&d) {
            *state_elem += k[row] * d_elem;
        }
    }
    let attention_scale = 1.0 / f32::sqrt(f32::from(GDN_HEAD_DIM_U16));
    (0..head_dim)
        .map(|col| {
            (0..head_dim)
                .map(|row| state[row * head_dim + col] * q[row])
                .sum::<f32>()
                * attention_scale
        })
        .collect()
}

/// Causal depth-4 convolution over `[history | new]` per channel, then `SiLU`.
/// The GGUF conv weight stores element `(tap, channel)` at
/// `tap + channel * d_conv`. Advances `conv` in place: drop the oldest
/// input, append the new token's projection. Returns the raw and activated
/// outputs.
fn gdn_conv_step(qkv_mixed: &[f32], ssm_conv1d: &[f32], conv: &mut [f32]) -> (Vec<f32>, Vec<f32>) {
    let history_len = GDN_D_CONV - 1;
    let conv_out: Vec<f32> = qkv_mixed
        .iter()
        .enumerate()
        .map(|(channel, new_input)| {
            (0..GDN_D_CONV)
                .map(|tap| {
                    let input = if tap < history_len {
                        conv[channel * history_len + tap]
                    } else {
                        *new_input
                    };
                    input * ssm_conv1d[tap + channel * GDN_D_CONV]
                })
                .sum::<f32>()
        })
        .collect();
    for (channel, new_input) in qkv_mixed.iter().enumerate() {
        let base = channel * history_len;
        conv[base] = conv[base + 1];
        conv[base + 1] = conv[base + 2];
        conv[base + 2] = *new_input;
    }
    let activated = conv_out.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    (conv_out, activated)
}

/// Dequantized weights for one GDN layer, in GGUF flat order.
pub struct GdnLayerWeights {
    /// `attn_qkv` [5120][10240].
    pub attn_qkv: Vec<f32>,
    /// `attn_gate` [5120][6144].
    pub attn_gate: Vec<f32>,
    /// `ssm_beta` [5120][48].
    pub ssm_beta: Vec<f32>,
    /// `ssm_alpha` [5120][48].
    pub ssm_alpha: Vec<f32>,
    /// `ssm_dt.bias` [48].
    pub ssm_dt_bias: Vec<f32>,
    /// `ssm_a` [48], already `-exp(A_log)`.
    pub ssm_a: Vec<f32>,
    /// `ssm_conv1d` [4][10240], element (tap, channel) at tap + channel*4.
    pub ssm_conv1d: Vec<f32>,
    /// `ssm_norm` [128], NOT +1'd.
    pub ssm_norm: Vec<f32>,
    /// `ssm_out` [6144][5120].
    pub ssm_out: Vec<f32>,
}

/// Intermediate tensors from one host GDN AR step, in ggml flat order, for
/// empirical parity checks against llama.cpp debug captures.
#[derive(Debug)]
pub struct GdnStepTrace {
    /// `linear_attn_qkv_mixed` `[10240]`.
    pub qkv_mixed: Vec<f32>,
    /// `z` gate projection `[6144]`.
    pub z_gate: Vec<f32>,
    /// Post-sigmoid `beta` `[48]`.
    pub beta: Vec<f32>,
    /// Decay log-gate `[48]`.
    pub gate: Vec<f32>,
    /// `conv_output_raw` `[10240]` (pre-`SiLU`).
    pub conv_raw: Vec<f32>,
    /// `conv_output_silu` `[10240]`.
    pub conv_act: Vec<f32>,
    /// Tiled, l2-normalized q `[48][128]`.
    pub q_tiled: Vec<f32>,
    /// Tiled, l2-normalized k `[48][128]`.
    pub k_tiled: Vec<f32>,
    /// `final_output` after the gated norm, before `ssm_out` `[6144]`.
    pub gated: Vec<f32>,
    /// Layer output `[5120]` (`linear_attn_out`).
    pub out: Vec<f32>,
}

/// One host-reference Gated-DeltaNet autoregressive decode step, returning
/// all intermediates for parity validation.
///
/// `matrix` is all V-head states `[heads][head_dim][head_dim]`; `conv` is the
/// per-channel history `[channels][d_conv-1]` in oldest-first order.
///
/// # Panics
///
/// Panics when `hidden` is not `[5120]` or the state/weight slices do not
/// match the pinned Qwen3.8-27B GDN geometry.
// The argument list mirrors the layer's weight/state set; a struct would
// obscure the reference equations under test.
#[allow(clippy::too_many_arguments)]
pub fn host_gdn_ar_step_traced(
    weights: &GdnLayerWeights,
    hidden: &[f32],
    matrix: &mut [f32],
    conv: &mut [f32],
    eps: f32,
) -> GdnStepTrace {
    assert_eq!(hidden.len(), N_EMBD);

    let qkv_mixed = gguf_gemv(&weights.attn_qkv, N_EMBD, hidden);
    let z_gate = gguf_gemv(&weights.attn_gate, N_EMBD, hidden);
    let beta: Vec<f32> = gguf_gemv(&weights.ssm_beta, N_EMBD, hidden)
        .iter()
        .map(|raw| 1.0 / (1.0 + (-raw).exp()))
        .collect();
    let gate: Vec<f32> = gguf_gemv(&weights.ssm_alpha, N_EMBD, hidden)
        .iter()
        .zip(&weights.ssm_dt_bias)
        .zip(&weights.ssm_a)
        .map(|((raw, dt), a_precomputed)| softplus(raw + dt) * a_precomputed)
        .collect();

    let (conv_raw, conv_act) = gdn_conv_step(&qkv_mixed, &weights.ssm_conv1d, conv);

    // Tile normalized k-heads across v-heads (tiled GGUF V ordering).
    let mut q48 = vec![0.0_f32; GDN_V_HEADS * GDN_HEAD_DIM];
    let mut k48 = vec![0.0_f32; GDN_V_HEADS * GDN_HEAD_DIM];
    for head in 0..GDN_K_HEADS {
        let qh = l2_normalize(
            &conv_act[head * GDN_HEAD_DIM..(head + 1) * GDN_HEAD_DIM],
            eps,
        );
        let kh = l2_normalize(
            &conv_act[GDN_K_OFFSET + head * GDN_HEAD_DIM..GDN_K_OFFSET + (head + 1) * GDN_HEAD_DIM],
            eps,
        );
        for rep in 0..(GDN_V_HEADS / GDN_K_HEADS) {
            let dst = (rep * GDN_K_HEADS + head) * GDN_HEAD_DIM;
            q48[dst..dst + GDN_HEAD_DIM].copy_from_slice(&qh);
            k48[dst..dst + GDN_HEAD_DIM].copy_from_slice(&kh);
        }
    }

    let head_dim = GDN_HEAD_DIM;
    let mut o = vec![0.0_f32; GDN_INNER];
    for head in 0..GDN_V_HEADS {
        let state = &mut matrix[head * head_dim * head_dim..(head + 1) * head_dim * head_dim];
        let k = &k48[head * head_dim..(head + 1) * head_dim];
        let v = &conv_act[GDN_V_OFFSET + head * head_dim..GDN_V_OFFSET + (head + 1) * head_dim];
        let q = &q48[head * head_dim..(head + 1) * head_dim];
        let out_head = gdn_state_step(state, k, v, q, gate[head].exp(), beta[head]);
        o[head * head_dim..(head + 1) * head_dim].copy_from_slice(&out_head);
    }

    // Gated norm: `rms(o) * ssm_norm * silu(z)`, per head; ssm_norm is NOT +1'd.
    let head_dim_f = f32::from(GDN_HEAD_DIM_U16);
    let mut gated = vec![0.0_f32; GDN_INNER];
    for head in 0..GDN_V_HEADS {
        let o_head = &o[head * head_dim..(head + 1) * head_dim];
        let sum: f32 = o_head.iter().map(|x| x * x).sum();
        let inv = (sum / head_dim_f + eps).sqrt().recip();
        for i in 0..head_dim {
            let zv = z_gate[head * head_dim + i];
            let silu = zv / (1.0 + (-zv).exp());
            gated[head * head_dim + i] = o_head[i] * inv * weights.ssm_norm[i] * silu;
        }
    }

    let out = gguf_gemv(&weights.ssm_out, GDN_INNER, &gated);
    GdnStepTrace {
        qkv_mixed,
        z_gate,
        beta,
        gate,
        conv_raw,
        conv_act,
        q_tiled: q48,
        k_tiled: k48,
        gated,
        out,
    }
}

/// One host-reference Gated-DeltaNet autoregressive decode step.
///
/// Returns the layer output `[5120]`.
///
/// # Panics
///
/// Panics when `hidden` is not `[5120]` or the state/weight slices do not
/// match the pinned Qwen3.8-27B GDN geometry.
pub fn host_gdn_ar_step(
    weights: &GdnLayerWeights,
    hidden: &[f32],
    matrix: &mut [f32],
    conv: &mut [f32],
    eps: f32,
) -> Vec<f32> {
    host_gdn_ar_step_traced(weights, hidden, matrix, conv, eps).out
}
