use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::cuda::CudaQuantizedWeight;

const Q8_0_VALUE_TYPE: u32 = 8;
const Q3_K_VALUE_TYPE: u32 = 11;
const Q4_K_VALUE_TYPE: u32 = 12;
const Q5_K_VALUE_TYPE: u32 = 13;
const Q6_K_VALUE_TYPE: u32 = 14;
const IQ4_NL_VALUE_TYPE: u32 = 20;
const IQ3_S_VALUE_TYPE: u32 = 21;
const IQ4_XS_VALUE_TYPE: u32 = 23;
const Q8_0_BLOCK_ELEMENTS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const IQ4_NL_BLOCK_ELEMENTS: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const IQ3_S_BLOCK_ELEMENTS: usize = 256;
const IQ3_S_BLOCK_BYTES: usize = 110;
const Q3_K_BLOCK_ELEMENTS: usize = 256;
const Q3_K_BLOCK_BYTES: usize = 110;
const Q4_K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_ELEMENTS: usize = 256;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_ELEMENTS: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const IQ4_XS_BLOCK_ELEMENTS: usize = 256;
const IQ4_XS_BLOCK_BYTES: usize = 136;

const IQ3_GRID_HEX: &[u8] = include_bytes!("iq3_grid.hex");
const IQ3_GRID_VALUES: [u8; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

fn iq3_hex_nibble(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'a'..=b'f' => value - b'a' + 10,
        b'A'..=b'F' => value - b'A' + 10,
        _ => 0,
    }
}

fn iq3_grid_values() -> Vec<u8> {
    let mut values = Vec::with_capacity(512 * 4);
    for code in 0..512 {
        for lane in 0..4 {
            let value_index = code * 4 + lane;
            let packed_index = value_index / 2;
            let high = iq3_hex_nibble(IQ3_GRID_HEX[packed_index * 2]);
            let low = iq3_hex_nibble(IQ3_GRID_HEX[packed_index * 2 + 1]);
            let packed = (high << 4) | low;
            let grid_index = usize::from((packed >> ((value_index % 2) * 4)) & 0x07);
            values.push(IQ3_GRID_VALUES[grid_index]);
        }
    }
    values
}

const Q_K_GEMV_SOURCE: &str = r#"
extern "C" __device__ __forceinline__ float decode_f16(unsigned short bits) {
    const int sign = (bits & 0x8000u) != 0u ? -1 : 1;
    const int exponent = (bits >> 10u) & 0x1fu;
    const int fraction = bits & 0x03ffu;
    if (exponent == 0) {
        return (float)sign * ((float)fraction / 1024.0f) * 0.00006103515625f;
    }
    if (exponent == 31) {
        if (fraction == 0) {
            return sign > 0 ? __int_as_float(0x7f800000) : __int_as_float(0xff800000);
        }
        return __int_as_float(0x7fc00000);
    }
    float scale = 1.0f;
    int shift = exponent - 15;
    if (shift > 0) {
        for (int i = 0; i < shift; ++i) {
            scale *= 2.0f;
        }
    } else {
        for (int i = 0; i > shift; --i) {
            scale *= 0.5f;
        }
    }
    return (float)sign * (1.0f + (float)fraction / 1024.0f) * scale;
}

extern "C" __device__ __forceinline__ int scale_value(const unsigned char* block, int group) {
    if (group < 4) {
        return (int)(block[4 + group] & 0x3fu);
    }
    const int index = group - 4;
    return (int)((block[12 + index] & 0x0fu) | ((block[4 + index] >> 2u) & 0x30u));
}

extern "C" __device__ __forceinline__ int minimum_value(const unsigned char* block, int group) {
    if (group < 4) {
        return (int)(block[8 + group] & 0x3fu);
    }
    const int index = group - 4;
    return (int)((block[12 + index] >> 4u) | ((block[8 + index] >> 2u) & 0x30u));
}

extern "C" __global__ void q4_k_embedding(
    const unsigned char* weights,
    unsigned int token_index,
    float* output,
    int hidden_size,
    int vocabulary_size
) {
    const int hidden_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (hidden_index >= hidden_size || token_index >= (unsigned int)vocabulary_size) {
        return;
    }

    const int blocks_per_token = hidden_size / 256;
    const int block_index = hidden_index / 256;
    const int local = hidden_index & 255;
    const unsigned char* block =
        weights + ((int)token_index * blocks_per_token + block_index) * 144;
    const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
    const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
    const int group = local / 32;
    const int index = local & 31;
    const int data_offset = 16 + (group / 2) * 32;
    const int shift = (group & 1) * 4;
    const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
    output[hidden_index] =
        d * (float)scale_value(block, group) * (float)quantized
        - min * (float)minimum_value(block, group);
}

extern "C" __global__ void iq3_s_embedding(
    const unsigned char* weights,
    unsigned int token_index,
    float* output,
    const unsigned char* grid,
    int hidden_size,
    int vocabulary_size
) {
    const int hidden_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (hidden_index >= hidden_size || token_index >= (unsigned int)vocabulary_size) {
        return;
    }

    const int blocks_per_token = hidden_size / 256;
    const int block_index = hidden_index / 256;
    const int local = hidden_index & 255;
    const unsigned char* block =
        weights + ((int)token_index * blocks_per_token + block_index) * 110;
    const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
    const unsigned char* low_codes = block + 2;
    const unsigned char* high_codes = block + 66;
    const unsigned char* signs = block + 74;
    const unsigned char* scales = block + 106;
    const int group = local / 32;
    const int within_group = local & 31;
    const int sub = within_group / 8;
    const int lane = within_group & 7;
    const int scale_nibble =
        (int)((scales[group / 2] >> ((group & 1) * 4)) & 0x0fu);
    const float group_scale = d * (1.0f + 2.0f * (float)scale_nibble);
    const int code_index = group * 8 + sub * 2 + lane / 4;
    const int high_bit =
        (int)((high_codes[code_index / 8] >> (code_index & 7)) & 1u);
    const int code = (int)low_codes[code_index] | (high_bit << 8);
    const int sign = ((signs[group * 4 + sub] >> lane) & 1u) == 0u ? 1 : -1;
    const int grid_index = code * 4 + (lane & 3);
    output[hidden_index] =
        group_scale * (float)grid[grid_index] * (float)sign;
}

extern "C" __global__ void iq3_s_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    const unsigned char* grid,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 110;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned char* low_codes = block + 2;
        const unsigned char* high_codes = block + 66;
        const unsigned char* signs = block + 74;
        const unsigned char* scales = block + 106;
        for (int group = 0; group < 8; ++group) {
            const int scale_nibble =
                (int)((scales[group / 2] >> ((group & 1) * 4)) & 0x0fu);
            const float group_scale = d * (1.0f + 2.0f * (float)scale_nibble);
            for (int sub = 0; sub < 4; ++sub) {
                const unsigned char sign_bits = signs[group * 4 + sub];
                for (int lane = 0; lane < 8; ++lane) {
                    const int code_index = group * 8 + sub * 2 + lane / 4;
                    const int high_bit =
                        (int)((high_codes[code_index / 8] >> (code_index & 7)) & 1u);
                    const int code = (int)low_codes[code_index] | (high_bit << 8);
                    const int sign = ((sign_bits >> lane) & 1u) == 0u ? 1 : -1;
                    const int grid_index = code * 4 + (lane & 3);
                    const float value = group_scale * (float)grid[grid_index] * (float)sign;
                    const int input_index = block_index * 256 + group * 32 + sub * 8 + lane;
                    accumulator += value * input[input_index];
                }
            }
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q8_0_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 32;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 34;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        for (int local = 0; local < 32; ++local) {
            const int quantized = (int)((signed char)block[2 + local]);
            accumulator += d * (float)quantized * input[block_index * 32 + local];
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void iq4_nl_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }
    const int blocks_per_output = input_size / 32;
    const signed char values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 18;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        for (int local = 0; local < 32; ++local) {
            const int index = local < 16 ? local : local - 16;
            const int nibble = local < 16
                ? (int)(block[2 + index] & 0x0fu)
                : (int)(block[2 + index] >> 4u);
            accumulator += d * (float)values[nibble] * input[block_index * 32 + local];
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void iq4_xs_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }
    const int blocks_per_output = input_size / 256;
    const signed char values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 136;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned short high_scales =
            (unsigned short)block[2] | ((unsigned short)block[3] << 8u);
        for (int group = 0; group < 8; ++group) {
            const int low = (int)((block[4 + group / 2] >> ((group & 1) * 4)) & 0x0fu);
            const int high = (int)((high_scales >> (group * 2)) & 0x03u);
            const float group_scale = d * (float)((low | (high << 4)) - 32);
            const int data_offset = 8 + group * 16;
            for (int local = 0; local < 16; ++local) {
                const unsigned char packed = block[data_offset + local];
                accumulator += group_scale * (float)values[packed & 0x0fu]
                    * input[block_index * 256 + group * 32 + local];
                accumulator += group_scale * (float)values[packed >> 4u]
                    * input[block_index * 256 + group * 32 + 16 + local];
            }
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q3_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 110;
        const float d = decode_f16((unsigned short)block[108] | ((unsigned short)block[109] << 8u));
        const unsigned char* high_bits = block;
        const unsigned char* low_bits = block + 32;
        const unsigned char* packed_scales = block + 96;
        for (int group = 0; group < 16; ++group) {
            const int index = group < 8 ? group : group - 8;
            const int scale_bits = group < 8
                ? (int)((packed_scales[index] & 0x0fu)
                    | (((packed_scales[8 + index % 4] >> ((index / 4) * 2)) & 0x03u) << 4u))
                : (int)((packed_scales[index] >> 4u)
                    | (((packed_scales[8 + index % 4] >> ((group / 4) * 2)) & 0x03u) << 4u));
            const int group_scale = scale_bits - 32;
            const int chunk = group / 8;
            const int variant = (group / 2) & 3;
            const int half = group & 1;
            const int low_offset = chunk * 32 + half * 16;
            const int high_offset = half * 16;
            for (int local = 0; local < 16; ++local) {
                const int low = (int)((low_bits[low_offset + local] >> (variant * 2)) & 0x03u);
                const int high = ((int)(high_bits[high_offset + local] >> (group / 2)) & 1) ^ 1;
                const int quantized = low - high * 4;
                accumulator += d * (float)group_scale * (float)quantized
                    * input[block_index * 256 + group * 16 + local];
            }
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q6_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 210;
        const unsigned char* low_bits = block;
        const unsigned char* high_bits = block + 128;
        const unsigned char* scales = block + 192;
        const float d = decode_f16((unsigned short)block[208] | ((unsigned short)block[209] << 8u));
        for (int group = 0; group < 8; ++group) {
            const int chunk = group / 4;
            const int variant = group & 3;
            const int low_offset = chunk * 64 + (variant & 1) * 32;
            const int low_shift = variant < 2 ? 0 : 4;
            const int high_offset = chunk * 32;
            const int high_shift = variant;
            for (int half = 0; half < 2; ++half) {
                const int group_scale = (int)(signed char)scales[group * 2 + half];
                for (int local = 0; local < 16; ++local) {
                    const int position = half * 16 + local;
                    const int low = (int)((low_bits[low_offset + position] >> low_shift) & 0x0fu);
                    const int high = (int)((high_bits[high_offset + position] >> (high_shift * 2)) & 0x03u);
                    const int quantized = (low | (high << 4)) - 32;
                    accumulator += d * (float)group_scale * (float)quantized
                        * input[block_index * 256 + group * 32 + half * 16 + local];
                }
            }
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q4_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 144;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        for (int local = 0; local < 256; ++local) {
            const int group = local / 32;
            const int index = local & 31;
            const int data_offset = 16 + (group / 2) * 32;
            const int shift = (group & 1) * 4;
            const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const float value =
                d * (float)scale_value(block, group) * (float)quantized
                - min * (float)minimum_value(block, group);
            accumulator += value * input[block_index * 256 + local];
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q5_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 176;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        for (int local = 0; local < 256; ++local) {
            const int group = local / 32;
            const int index = local & 31;
            const int data_offset = 48 + (group / 2) * 32;
            const int shift = (group & 1) * 4;
            const int low = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const int high = (int)((block[16 + index] >> group) & 1u);
            const int quantized = low | (high << 4);
            const float value =
                d * (float)scale_value(block, group) * (float)quantized
                - min * (float)minimum_value(block, group);
            accumulator += value * input[block_index * 256 + local];
        }
    }
    output[output_index] = accumulator;
}

// ------------------------------------------------------------------------
// Warp-cooperative variants: one warp per output row. Lanes split each
// 256-element block into 8 consecutive elements (32-element blocks map one
// element per lane), so weight bytes and input floats are read by sector-
// coalesced transactions and every lane stays busy. Each lane accumulates
// its partial products across all blocks; one shuffle reduction produces the
// row result. The decode math is identical to the scalar kernels above,
// which remain the correctness oracle.

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void q8_0_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 32;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 34;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const int quantized = (int)(signed char)block[2 + lane];
        accumulator += d * (float)quantized * input[block_index * 32 + lane];
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void iq4_nl_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 32;
    const signed char values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 18;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const int index = lane < 16 ? lane : lane - 16;
        const int nibble = lane < 16
            ? (int)(block[2 + index] & 0x0fu)
            : (int)(block[2 + index] >> 4u);
        accumulator += d * (float)values[nibble] * input[block_index * 32 + lane];
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void iq4_xs_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    const signed char values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 136;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned short high_scales =
            (unsigned short)block[2] | ((unsigned short)block[3] << 8u);
        const int group = lane >> 2;
        const int within = (lane & 3) * 8;
        const int low = (int)((block[4 + group / 2] >> ((group & 1) * 4)) & 0x0fu);
        const int high = (int)((high_scales >> (group * 2)) & 0x03u);
        const float group_scale = d * (float)((low | (high << 4)) - 32);
        const int data_offset = 8 + group * 16;
        for (int j = 0; j < 8; ++j) {
            const int local = group * 32 + within + j;
            const int element = within + j;
            const unsigned char packed = block[data_offset + (element & 15)];
            const int nibble = element < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4u);
            accumulator +=
                group_scale * (float)values[nibble] * input[block_index * 256 + local];
        }
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void q3_k_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 110;
        const float d = decode_f16((unsigned short)block[108] | ((unsigned short)block[109] << 8u));
        const unsigned char* high_bits = block;
        const unsigned char* low_bits = block + 32;
        const unsigned char* packed_scales = block + 96;
        const int group = lane >> 1;
        const int local = (lane & 1) * 8;
        const int index = group < 8 ? group : group - 8;
        const int scale_bits = group < 8
            ? (int)((packed_scales[index] & 0x0fu)
                | (((packed_scales[8 + index % 4] >> ((index / 4) * 2)) & 0x03u) << 4u))
            : (int)((packed_scales[index] >> 4u)
                | (((packed_scales[8 + index % 4] >> ((group / 4) * 2)) & 0x03u) << 4u));
        const int group_scale = scale_bits - 32;
        const int chunk = group / 8;
        const int variant = (group / 2) & 3;
        const int half = group & 1;
        const int low_offset = chunk * 32 + half * 16;
        const int high_offset = half * 16;
        for (int j = 0; j < 8; ++j) {
            const int within = local + j;
            const int low = (int)((low_bits[low_offset + within] >> (variant * 2)) & 0x03u);
            const int high = ((int)(high_bits[high_offset + within] >> (group / 2)) & 1) ^ 1;
            const int quantized = low - high * 4;
            accumulator += d * (float)group_scale * (float)quantized
                * input[block_index * 256 + group * 16 + within];
        }
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void q6_k_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 210;
        const unsigned char* low_bits = block;
        const unsigned char* high_bits = block + 128;
        const unsigned char* scales = block + 192;
        const float d = decode_f16((unsigned short)block[208] | ((unsigned short)block[209] << 8u));
        const int group = lane >> 2;
        const int position_base = (lane & 3) * 8;
        const int chunk = group / 4;
        const int variant = group & 3;
        const int low_offset = chunk * 64 + (variant & 1) * 32;
        const int low_shift = variant < 2 ? 0 : 4;
        const int high_offset = chunk * 32;
        for (int j = 0; j < 8; ++j) {
            const int position = position_base + j;
            const int half = position >= 16 ? 1 : 0;
            const int group_scale = (int)(signed char)scales[group * 2 + half];
            const int low = (int)((low_bits[low_offset + position] >> low_shift) & 0x0fu);
            const int high = (int)((high_bits[high_offset + position] >> (variant * 2)) & 0x03u);
            const int quantized = (low | (high << 4)) - 32;
            accumulator += d * (float)group_scale * (float)quantized
                * input[block_index * 256 + group * 32 + position];
        }
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void q4_k_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 144;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        const int group = lane >> 2;
        const int index_base = (lane & 3) * 8;
        const int data_offset = 16 + (group / 2) * 32;
        const int shift = (group & 1) * 4;
        const float group_scale = (float)scale_value(block, group);
        const float group_minimum = (float)minimum_value(block, group);
        for (int j = 0; j < 8; ++j) {
            const int index = index_base + j;
            const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const float value =
                d * group_scale * (float)quantized - min * group_minimum;
            accumulator += value * input[block_index * 256 + group * 32 + index];
        }
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void q5_k_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 176;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        const int group = lane >> 2;
        const int index_base = (lane & 3) * 8;
        const int data_offset = 48 + (group / 2) * 32;
        const int shift = (group & 1) * 4;
        const float group_scale = (float)scale_value(block, group);
        const float group_minimum = (float)minimum_value(block, group);
        for (int j = 0; j < 8; ++j) {
            const int index = index_base + j;
            const int low = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const int high = (int)((block[16 + index] >> group) & 1u);
            const int quantized = low | (high << 4);
            const float value =
                d * group_scale * (float)quantized - min * group_minimum;
            accumulator += value * input[block_index * 256 + group * 32 + index];
        }
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void iq3_s_gemv_warp(
    const unsigned char* weights,
    const float* input,
    float* output,
    const unsigned char* grid,
    int input_size,
    int output_size
) {
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 110;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned char* low_codes = block + 2;
        const unsigned char* high_codes = block + 66;
        const unsigned char* signs = block + 74;
        const unsigned char* scales = block + 106;
        const int group = lane >> 2;
        const int sub = (lane & 3) * 2;
        const int scale_nibble =
            (int)((scales[group / 2] >> ((group & 1) * 4)) & 0x0fu);
        const float group_scale = d * (1.0f + 2.0f * (float)scale_nibble);
        for (int j = 0; j < 8; ++j) {
            const int lane_in = sub * 8 + j;
            const int code_index = group * 8 + sub * 2 + lane_in / 4;
            const int high_bit =
                (int)((high_codes[code_index / 8] >> (code_index & 7)) & 1u);
            const int code = (int)low_codes[code_index] | (high_bit << 8);
            const int sign = ((signs[group * 4 + sub] >> lane_in) & 1u) == 0u ? 1 : -1;
            const int grid_index = code * 4 + (lane_in & 3);
            const float value = group_scale * (float)grid[grid_index] * (float)sign;
            accumulator += value * input[block_index * 256 + group * 32 + lane_in];
        }
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}
"#;

#[derive(Debug)]
pub enum CudaQuantizedKernelError {
    Driver(String),
    Nvrtc(String),
    InvalidWeight(String),
    UnsupportedValueType { expected: u32, actual: u32 },
    ShapeOverflow,
    InputLength { expected: usize, actual: usize },
    OutputLength { expected: usize, actual: usize },
    ContextMismatch,
}

impl fmt::Display for CudaQuantizedKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA driver error: {message}"),
            Self::Nvrtc(message) => write!(f, "NVRTC compilation error: {message}"),
            Self::InvalidWeight(reason) => write!(f, "invalid quantized weight: {reason}"),
            Self::UnsupportedValueType { expected, actual } => write!(
                f,
                "quantized GEMV requires GGML value type {expected}, got {actual}"
            ),
            Self::ShapeOverflow => {
                f.write_str("quantized GEMV shape does not fit CUDA launch arguments")
            }
            Self::InputLength { expected, actual } => {
                write!(
                    f,
                    "quantized GEMV input has {actual} values, expected {expected}"
                )
            }
            Self::OutputLength { expected, actual } => {
                write!(
                    f,
                    "quantized GEMV output has {actual} values, expected {expected}"
                )
            }
            Self::ContextMismatch => {
                f.write_str("quantized GEMV buffers and stream belong to different CUDA contexts")
            }
        }
    }
}

impl std::error::Error for CudaQuantizedKernelError {}

/// Shared launch and validation state for the block-quantized GEMV kernels.
///
/// The kernels follow GGML's column-major tensor ordering: the first tensor
/// dimension is the contiguous input (`K`) count and the second is the output
/// (`N`) count. They deliberately favor a simple one-thread-per-output-column
/// implementation; a measured tiled kernel can replace it without changing
/// the encoded-weight or model-facing boundary.
struct CudaQuantizedGemv {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    warp_kernel: Option<CudaFunction>,
    value_type: u32,
    block_elements: usize,
    block_bytes: usize,
    label: &'static str,
}

impl CudaQuantizedGemv {
    fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        function_name: &'static str,
        value_type: u32,
        block_elements: usize,
        block_bytes: usize,
        label: &'static str,
    ) -> Result<Self, CudaQuantizedKernelError> {
        if context.as_ref() != stream.context().as_ref() {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        let ptx = compile_ptx(Q_K_GEMV_SOURCE)
            .map_err(|error| CudaQuantizedKernelError::Nvrtc(error.to_string()))?;
        let module = context
            .load_module(ptx)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let kernel = module
            .load_function(function_name)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        // Embedding lookups keep the scalar kernel only; GEMV families also
        // compile their warp-cooperative variant for measured selection.
        let warp_name: Option<&'static str> = match function_name {
            "q8_0_gemv" => Some("q8_0_gemv_warp"),
            "iq4_nl_gemv" => Some("iq4_nl_gemv_warp"),
            "iq4_xs_gemv" => Some("iq4_xs_gemv_warp"),
            "q3_k_gemv" => Some("q3_k_gemv_warp"),
            "q6_k_gemv" => Some("q6_k_gemv_warp"),
            "q4_k_gemv" => Some("q4_k_gemv_warp"),
            "q5_k_gemv" => Some("q5_k_gemv_warp"),
            "iq3_s_gemv" => Some("iq3_s_gemv_warp"),
            _ => None,
        };
        let warp_kernel = warp_name
            .map(|name| {
                module
                    .load_function(name)
                    .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))
            })
            .transpose()?;
        Ok(Self {
            stream,
            kernel,
            warp_kernel,
            value_type,
            block_elements,
            block_bytes,
            label,
        })
    }

    fn validate(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
    ) -> Result<(u32, u32, LaunchConfig), CudaQuantizedKernelError> {
        if self.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.stream.context().as_ref() != input.context().as_ref()
            || self.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if weight.value_type() != self.value_type {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: self.value_type,
                actual: weight.value_type(),
            });
        }
        let dimensions = weight.spec().dimensions();
        if dimensions.len() != 2 {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "expected rank-2 tensor, got rank {}",
                dimensions.len()
            )));
        }
        let input_size =
            usize::try_from(dimensions[0]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let output_size =
            usize::try_from(dimensions[1]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        if input_size == 0 || output_size == 0 || !input_size.is_multiple_of(self.block_elements) {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "{} shape is {input_size}x{output_size}; input size must be a positive multiple of {}",
                self.label, self.block_elements
            )));
        }
        let blocks = input_size
            .checked_div(self.block_elements)
            .and_then(|value| value.checked_mul(output_size))
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let expected_bytes = blocks
            .checked_mul(self.block_bytes)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        if weight.encoded_bytes() != expected_bytes {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "encoded length is {}, expected {expected_bytes}",
                weight.encoded_bytes()
            )));
        }
        if blocks > (i32::MAX as usize) / self.block_bytes {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if input_size > i32::MAX as usize || output_size > i32::MAX as usize {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if input.len() != input_size {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: input_size,
                actual: input.len(),
            });
        }
        if output.len() != output_size {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: output_size,
                actual: output.len(),
            });
        }
        let input_size =
            u32::try_from(input_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let output_size =
            u32::try_from(output_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (output_size.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        Ok((input_size, output_size, config))
    }

    /// Execute one block-quantized matrix-vector product into a caller-owned
    /// output.
    ///
    /// The launch is asynchronous with respect to the host. A subsequent copy
    /// or stream synchronization observes completion. This method does not
    /// dequantize or copy the encoded weight through host memory.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight type, shape, encoded
    /// length, or device vector lengths are invalid, or when launch fails.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        let (input_size, output_size, config) = self.validate(weight, input, output)?;
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Execute the warp-cooperative variant when one was compiled for this
    /// kernel family. One warp computes each output row: lanes split each
    /// quantization block into consecutive element runs so both weight-byte
    /// and input reads coalesce, and a shuffle reduction produces the row
    /// result. The scalar kernel remains the correctness oracle.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no warp
    /// variant or validation/launch fails.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        const WARPS_PER_BLOCK: u32 = 4;
        let warp_kernel = self.warp_kernel.as_ref().ok_or_else(|| {
            CudaQuantizedKernelError::InvalidWeight(format!(
                "{} has no warp-cooperative variant",
                self.label
            ))
        })?;
        let (input_size, output_size, _) = self.validate(weight, input, output)?;
        let blocks = output_size.div_ceil(WARPS_PER_BLOCK);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: same slices/validation as the scalar path; the warp kernel
        // writes only rows [0, output_size) once each from lane 0.
        unsafe {
            self.stream
                .launch_builder(warp_kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `IQ4_NL` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaIq4NlGemv {
    inner: CudaQuantizedGemv,
}

impl CudaIq4NlGemv {
    /// Compile and load the `IQ4_NL` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `IQ4_NL` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "iq4_nl_gemv",
                IQ4_NL_VALUE_TYPE,
                IQ4_NL_BLOCK_ELEMENTS,
                IQ4_NL_BLOCK_BYTES,
                "IQ4_NL",
            )?,
        })
    }

    /// Execute one `IQ4_NL` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }
}

/// A correctness-oriented `IQ3_S` matrix-vector kernel.
///
/// The `IQ3_S` grid is kept as a small device-resident lookup table rather than
/// making the optional CUDA crate depend on the GGUF reader. The table is
/// generated from the canonical GGML packed mapping at construction time.
pub struct CudaIq3SGemv {
    inner: CudaQuantizedGemv,
    grid: CudaSlice<u8>,
}

impl CudaIq3SGemv {
    /// Compile and load the `IQ3_S` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `IQ3_S` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC, module loading, or the
    /// lookup-table upload fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        let grid = stream
            .clone_htod(&iq3_grid_values())
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let inner = CudaQuantizedGemv::from_context(
            context,
            stream,
            "iq3_s_gemv",
            IQ3_S_VALUE_TYPE,
            IQ3_S_BLOCK_ELEMENTS,
            IQ3_S_BLOCK_BYTES,
            "IQ3_S",
        )?;
        Ok(Self { inner, grid })
    }

    /// Execute one `IQ3_S` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        let (input_size, output_size, config) = self.inner.validate(weight, input, output)?;
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.inner
                .stream
                .launch_builder(&self.inner.kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&self.grid)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Execute the warp-cooperative `IQ3_S` `GEMV` variant into a
    /// caller-owned output. The launch is asynchronous with respect to the
    /// host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes,
    /// contexts, or launch arguments are invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        const WARPS_PER_BLOCK: u32 = 4;
        let (input_size, output_size, _) = self.inner.validate(weight, input, output)?;
        let warp_kernel = self.inner.warp_kernel.as_ref().ok_or_else(|| {
            CudaQuantizedKernelError::InvalidWeight(
                "IQ3_S has no warp-cooperative variant".to_owned(),
            )
        })?;
        let blocks = output_size.div_ceil(WARPS_PER_BLOCK);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: same slices/validation as the scalar path; the warp kernel
        // writes only rows [0, output_size) once each from lane 0.
        unsafe {
            self.inner
                .stream
                .launch_builder(warp_kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&self.grid)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `IQ3_S` embedding lookup kernel.
///
/// GGML stores an embedding matrix as `[hidden, vocabulary]`, so each token is
/// one contiguous column of `IQ3_S` blocks. This kernel gathers one such column
/// directly into a device F32 vector.
pub struct CudaIq3SEmbedding {
    inner: CudaQuantizedGemv,
    grid: CudaSlice<u8>,
}

impl CudaIq3SEmbedding {
    /// Compile and load the `IQ3_S` embedding kernel on a new CUDA context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `IQ3_S` embedding kernel on an existing context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or the lookup
    /// table upload fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        let grid = stream
            .clone_htod(&iq3_grid_values())
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let inner = CudaQuantizedGemv::from_context(
            context,
            stream,
            "iq3_s_embedding",
            IQ3_S_VALUE_TYPE,
            IQ3_S_BLOCK_ELEMENTS,
            IQ3_S_BLOCK_BYTES,
            "IQ3_S embedding",
        )?;
        Ok(Self { inner, grid })
    }

    /// Gather one token embedding into a caller-owned device vector.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, token index,
    /// output shape, contexts, or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        token_index: u32,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        if self.inner.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.inner.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if weight.value_type() != IQ3_S_VALUE_TYPE {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: IQ3_S_VALUE_TYPE,
                actual: weight.value_type(),
            });
        }
        let dimensions = weight.spec().dimensions();
        if dimensions.len() != 2 {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "expected rank-2 tensor, got rank {}",
                dimensions.len()
            )));
        }
        let hidden_size =
            usize::try_from(dimensions[0]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            usize::try_from(dimensions[1]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        if hidden_size == 0
            || vocabulary_size == 0
            || !hidden_size.is_multiple_of(IQ3_S_BLOCK_ELEMENTS)
        {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "IQ3_S embedding shape is {hidden_size}x{vocabulary_size}; hidden size must be a positive multiple of {IQ3_S_BLOCK_ELEMENTS}"
            )));
        }
        let blocks = hidden_size
            .checked_div(IQ3_S_BLOCK_ELEMENTS)
            .and_then(|value| value.checked_mul(vocabulary_size))
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let expected_bytes = blocks
            .checked_mul(IQ3_S_BLOCK_BYTES)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        if weight.encoded_bytes() != expected_bytes {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "encoded length is {}, expected {expected_bytes}",
                weight.encoded_bytes()
            )));
        }
        if u64::from(token_index) >= dimensions[1] {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "token index {token_index} is outside vocabulary size {vocabulary_size}"
            )));
        }
        if hidden_size > i32::MAX as usize || vocabulary_size > i32::MAX as usize {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if output.len() != hidden_size {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: hidden_size,
                actual: output.len(),
            });
        }
        let hidden_size =
            u32::try_from(hidden_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            u32::try_from(vocabulary_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (hidden_size.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.inner
                .stream
                .launch_builder(&self.inner.kernel)
                .arg(weight.encoded_data())
                .arg(&token_index)
                .arg(output)
                .arg(&self.grid)
                .arg(&hidden_size)
                .arg(&vocabulary_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `Q4_K` embedding lookup kernel.
///
/// `token_embd.weight` in the pinned artifact is `Q4_K` `[hidden, vocab]`;
/// this gathers one vocabulary row into a caller-owned device vector.
pub struct CudaQ4KEmbedding {
    inner: CudaQuantizedGemv,
}

impl CudaQ4KEmbedding {
    /// Compile and load the `Q4_K` embedding kernel on a new CUDA context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module
    /// loading fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q4_K` embedding kernel on an existing context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading
    /// fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        let inner = CudaQuantizedGemv::from_context(
            context,
            stream,
            "q4_k_embedding",
            Q4_K_VALUE_TYPE,
            Q4_K_BLOCK_ELEMENTS,
            Q4_K_BLOCK_BYTES,
            "Q4_K embedding",
        )?;
        Ok(Self { inner })
    }

    /// Gather one token embedding into a caller-owned device vector.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, token index,
    /// output shape, contexts, or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        token_index: u32,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        if self.inner.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.inner.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if weight.value_type() != Q4_K_VALUE_TYPE {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: Q4_K_VALUE_TYPE,
                actual: weight.value_type(),
            });
        }
        let dimensions = weight.spec().dimensions();
        if dimensions.len() != 2 {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "expected rank-2 tensor, got rank {}",
                dimensions.len()
            )));
        }
        let hidden_size =
            usize::try_from(dimensions[0]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            usize::try_from(dimensions[1]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        if hidden_size == 0
            || vocabulary_size == 0
            || !hidden_size.is_multiple_of(Q4_K_BLOCK_ELEMENTS)
        {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "Q4_K embedding shape is {hidden_size}x{vocabulary_size}; hidden size must be a positive multiple of {Q4_K_BLOCK_ELEMENTS}"
            )));
        }
        let blocks = hidden_size
            .checked_div(Q4_K_BLOCK_ELEMENTS)
            .and_then(|value| value.checked_mul(vocabulary_size))
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let expected_bytes = blocks
            .checked_mul(Q4_K_BLOCK_BYTES)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        if weight.encoded_bytes() != expected_bytes {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "encoded length is {}, expected {expected_bytes}",
                weight.encoded_bytes()
            )));
        }
        if u64::from(token_index) >= dimensions[1] {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "token index {token_index} is outside vocabulary size {vocabulary_size}"
            )));
        }
        if hidden_size > i32::MAX as usize || vocabulary_size > i32::MAX as usize {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if output.len() != hidden_size {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: hidden_size,
                actual: output.len(),
            });
        }
        let hidden_size =
            u32::try_from(hidden_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            u32::try_from(vocabulary_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (hidden_size.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.inner
                .stream
                .launch_builder(&self.inner.kernel)
                .arg(weight.encoded_data())
                .arg(&token_index)
                .arg(output)
                .arg(&hidden_size)
                .arg(&vocabulary_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `IQ4_XS` matrix-vector kernel.
///
/// It shares the GGML `[K,N]` contract and launch validation with
/// [`CudaIq4NlGemv`], but decodes per-group scales and the 136-byte block
/// layout used by GGML value type 23.
pub struct CudaIq4XsGemv {
    inner: CudaQuantizedGemv,
}

impl CudaIq4XsGemv {
    /// Compile and load the `IQ4_XS` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `IQ4_XS` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "iq4_xs_gemv",
                IQ4_XS_VALUE_TYPE,
                IQ4_XS_BLOCK_ELEMENTS,
                IQ4_XS_BLOCK_BYTES,
                "IQ4_XS",
            )?,
        })
    }

    /// Execute one `IQ4_XS` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }
}

/// A correctness-oriented `Q3_K` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ3KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ3KGemv {
    /// Compile and load the `Q3_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q3_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q3_k_gemv",
                Q3_K_VALUE_TYPE,
                Q3_K_BLOCK_ELEMENTS,
                Q3_K_BLOCK_BYTES,
                "Q3_K",
            )?,
        })
    }

    /// Execute one `Q3_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }
}

/// A correctness-oriented `Q8_0` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ8_0Gemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ8_0Gemv {
    /// Compile and load the `Q8_0` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q8_0` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q8_0_gemv",
                Q8_0_VALUE_TYPE,
                Q8_0_BLOCK_ELEMENTS,
                Q8_0_BLOCK_BYTES,
                "Q8_0",
            )?,
        })
    }

    /// Execute one `Q8_0` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }
}

/// A correctness-oriented `Q6_K` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ6KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ6KGemv {
    /// Compile and load the `Q6_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q6_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q6_k_gemv",
                Q6_K_VALUE_TYPE,
                Q6_K_BLOCK_ELEMENTS,
                Q6_K_BLOCK_BYTES,
                "Q6_K",
            )?,
        })
    }

    /// Execute one `Q6_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }
}

/// A correctness-oriented `Q4_K` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ4KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ4KGemv {
    /// Compile and load the `Q4_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q4_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q4_k_gemv",
                Q4_K_VALUE_TYPE,
                Q4_K_BLOCK_ELEMENTS,
                Q4_K_BLOCK_BYTES,
                "Q4_K",
            )?,
        })
    }

    /// Execute one `Q4_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }
}

/// A correctness-oriented `Q5_K` matrix-vector kernel.
///
/// It shares the GGML `[K,N]` contract and launch validation with
/// [`CudaQ4KGemv`], but decodes the 5-bit high-bit plane and 176-byte block
/// layout used by GGML value type 13.
pub struct CudaQ5KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ5KGemv {
    /// Compile and load the `Q5_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q5_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q5_k_gemv",
                Q5_K_VALUE_TYPE,
                Q5_K_BLOCK_ELEMENTS,
                Q5_K_BLOCK_BYTES,
                "Q5_K",
            )?,
        })
    }

    /// Execute one `Q5_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }
}
