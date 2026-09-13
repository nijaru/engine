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
    // Same per-lane element mapping as the batched variant, so both paths
    // accumulate identical partial sums in identical order: chunked prefill
    // stays bit-identical to serial prefill while this path's activation reads
    // coalesce like the batched kernel's.
    const int group_low = lane >> 3;
    const int index = (lane & 7) * 4;
    const int sub = (lane & 7) >> 1;
    const int lane_in = (lane & 1) * 4;
    const int code_offset = group_low * 8 + sub * 2 + (lane & 1);
    const int code_byte = code_offset >> 3;
    const int code_bit = code_offset & 7;
    const int sign_shift = lane_in;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 110;
        const unsigned char* low_codes = block + 2;
        const unsigned char* high_codes = block + 66;
        const unsigned char* signs = block + 74;
        const unsigned char* scales = block + 106;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const int scale_nibble_low =
            (int)((scales[group_low / 2] >> ((group_low & 1) * 4)) & 0x0fu);
        const int scale_nibble_high =
            (int)((scales[group_low / 2 + 2] >> ((group_low & 1) * 4)) & 0x0fu);
        const float group_scale_low = d * (1.0f + 2.0f * (float)scale_nibble_low);
        const float group_scale_high = d * (1.0f + 2.0f * (float)scale_nibble_high);
        const unsigned char sign_bits_low = signs[group_low * 4 + sub];
        const unsigned char sign_bits_high = signs[group_low * 4 + sub + 16];
        const int code_low = (int)low_codes[code_offset]
            | ((int)((high_codes[code_byte] >> code_bit) & 1u) << 8);
        const int code_high = (int)low_codes[code_offset + 32]
            | ((int)((high_codes[code_byte + 4] >> code_bit) & 1u) << 8);
        const unsigned int grid_low = *(const unsigned int*)(grid + code_low * 4);
        const unsigned int grid_high = *(const unsigned int*)(grid + code_high * 4);
        float value_low[4];
        float value_high[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float grid_value_low = (float)(int)((grid_low >> (8 * t)) & 0xffu);
            const float grid_value_high = (float)(int)((grid_high >> (8 * t)) & 0xffu);
            const int sign_low =
                ((sign_bits_low >> (sign_shift + t)) & 1u) == 0u ? 1 : -1;
            const int sign_high =
                ((sign_bits_high >> (sign_shift + t)) & 1u) == 0u ? 1 : -1;
            value_low[t] = group_scale_low * grid_value_low * (float)sign_low;
            value_high[t] = group_scale_high * grid_value_high * (float)sign_high;
        }
        const float4* row_input = (const float4*)(input + block_index * 256);
        const float4 low = row_input[lane];
        const float4 high = row_input[32 + lane];
        accumulator += value_low[0] * low.x + value_low[1] * low.y
            + value_low[2] * low.z + value_low[3] * low.w
            + value_high[0] * high.x + value_high[1] * high.y
            + value_high[2] * high.z + value_high[3] * high.w;
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void iq3_s_gemv_warp_batch(
    const unsigned char* weights,
    const float* input,
    float* output,
    const unsigned char* grid,
    int input_size,
    int output_size,
    int members
) {
    // Weights-read-once: one warp per weight row; each decoded element is
    // reused against every member's input.
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int row_base = warp * MAX_BATCH_ROWS;
    const int lane = (int)(threadIdx.x & 31u);
    if (row_base >= output_size) {
        return;
    }
    // Constant initializer plus fully unrolled predicated member loops keep
    // acc[] in registers; a runtime-bounded member loop spills it to local
    // memory and serializes the member input loads.
    float acc[MAX_BATCH_ROWS][MAX_BATCH_MEMBERS] = {0.0f};
    const int blocks_per_output = input_size / 256;
    // Each lane owns four consecutive activations in the low half of the
    // block (elements `4*lane`) and four in the high half (elements
    // `128 + 4*lane`). Both runs are 16-byte aligned, so one member's
    // activations are two `float4` loads per lane and a warp load covers 512
    // contiguous bytes; the strided byte-per-element form it replaced spent
    // eight wavefronts per 1 KiB of activations on sector transactions alone.
    // All four elements of a run share one grid code, so the codebook values
    // are one aligned word load and the sign byte supplies four bits.
    const int group_low = lane >> 3;
    const int index = (lane & 7) * 4;
    const int sub = (lane & 7) >> 1;
    const int lane_in = (lane & 1) * 4;
    const int code_offset = group_low * 8 + sub * 2 + (lane & 1);
    const int code_byte = code_offset >> 3;
    const int code_bit = code_offset & 7;
    const int sign_shift = lane_in;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        float value_low[MAX_BATCH_ROWS][4];
        float value_high[MAX_BATCH_ROWS][4];
        #pragma unroll
        for (int r = 0; r < MAX_BATCH_ROWS; ++r) {
            const int row = row_base + r;
            if (row < output_size) {
                const unsigned char* block =
                    weights + (row * blocks_per_output + block_index) * 110;
                const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
                const unsigned char* low_codes = block + 2;
                const unsigned char* high_codes = block + 66;
                const unsigned char* signs = block + 74;
                const unsigned char* scales = block + 106;
                const int scale_nibble_low =
                    (int)((scales[group_low / 2] >> ((group_low & 1) * 4)) & 0x0fu);
                const int scale_nibble_high =
                    (int)((scales[group_low / 2 + 2] >> ((group_low & 1) * 4)) & 0x0fu);
                const float group_scale_low = d * (1.0f + 2.0f * (float)scale_nibble_low);
                const float group_scale_high = d * (1.0f + 2.0f * (float)scale_nibble_high);
                const unsigned char sign_bits_low = signs[group_low * 4 + sub];
                const unsigned char sign_bits_high = signs[group_low * 4 + sub + 16];
                const int code_low = (int)low_codes[code_offset]
                    | ((int)((high_codes[code_byte] >> code_bit) & 1u) << 8);
                const int code_high = (int)low_codes[code_offset + 32]
                    | ((int)((high_codes[code_byte + 4] >> code_bit) & 1u) << 8);
                const unsigned int grid_low = *(const unsigned int*)(grid + code_low * 4);
                const unsigned int grid_high = *(const unsigned int*)(grid + code_high * 4);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const float grid_value_low = (float)(int)((grid_low >> (8 * t)) & 0xffu);
                    const float grid_value_high = (float)(int)((grid_high >> (8 * t)) & 0xffu);
                    const int sign_low =
                        ((sign_bits_low >> (sign_shift + t)) & 1u) == 0u ? 1 : -1;
                    const int sign_high =
                        ((sign_bits_high >> (sign_shift + t)) & 1u) == 0u ? 1 : -1;
                    value_low[r][t] = group_scale_low * grid_value_low * (float)sign_low;
                    value_high[r][t] = group_scale_high * grid_value_high * (float)sign_high;
                }
            }
        }
        const float* member_input = input + block_index * 256;
        #pragma unroll
        for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
            if (m < members) {
                const float4* row_input =
                    (const float4*)(member_input + (long long)m * input_size);
                const float4 low = row_input[lane];
                const float4 high = row_input[32 + lane];
                // One activation load feeds every blocked row: the rows
                // differ in weights, not in what this member contributes.
                #pragma unroll
                for (int r = 0; r < MAX_BATCH_ROWS; ++r) {
                    acc[r][m] += value_low[r][0] * low.x + value_low[r][1] * low.y
                        + value_low[r][2] * low.z + value_low[r][3] * low.w
                        + value_high[r][0] * high.x + value_high[r][1] * high.y
                        + value_high[r][2] * high.z + value_high[r][3] * high.w;
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < MAX_BATCH_ROWS; ++r) {
        const int row = row_base + r;
        if (row < output_size) {
            #pragma unroll
            for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
                if (m < members) {
                    // Every lane participates in the shuffle reduction; lane 0
                    // holds the full sum and writes it.
                    const float total = warp_sum(acc[r][m]);
                    if (lane == 0) {
                        output[(long long)m * output_size + row] = total;
                    }
                }
            }
        }
    }
}
