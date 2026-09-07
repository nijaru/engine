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
        const int sub = lane & 3;
        const int scale_nibble =
            (int)((scales[group / 2] >> ((group & 1) * 4)) & 0x0fu);
        const float group_scale = d * (1.0f + 2.0f * (float)scale_nibble);
        const unsigned char sign_bits = signs[group * 4 + sub];
        for (int lane_in = 0; lane_in < 8; ++lane_in) {
            const int code_index = group * 8 + sub * 2 + lane_in / 4;
            const int high_bit =
                (int)((high_codes[code_index / 8] >> (code_index & 7)) & 1u);
            const int code = (int)low_codes[code_index] | (high_bit << 8);
            const int sign = ((sign_bits >> lane_in) & 1u) == 0u ? 1 : -1;
            const int grid_index = code * 4 + (lane_in & 3);
            const float value = group_scale * (float)grid[grid_index] * (float)sign;
            accumulator += value * input[block_index * 256 + group * 32 + sub * 8 + lane_in];
        }
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
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }
    // Constant initializer plus fully unrolled predicated member loops keep
    // acc[] in registers; a runtime-bounded member loop spills it to local
    // memory and serializes the member input loads.
    float acc[MAX_BATCH_MEMBERS] = {0.0f};
    const int blocks_per_output = input_size / 256;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 110;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned char* low_codes = block + 2;
        const unsigned char* high_codes = block + 66;
        const unsigned char* signs = block + 74;
        const unsigned char* scales = block + 106;
        const int group = lane >> 2;
        const int sub = lane & 3;
        const int scale_nibble =
            (int)((scales[group / 2] >> ((group & 1) * 4)) & 0x0fu);
        const float group_scale = d * (1.0f + 2.0f * (float)scale_nibble);
        const unsigned char sign_bits = signs[group * 4 + sub];
        for (int lane_in = 0; lane_in < 8; ++lane_in) {
            const int code_index = group * 8 + sub * 2 + lane_in / 4;
            const int high_bit =
                (int)((high_codes[code_index / 8] >> (code_index & 7)) & 1u);
            const int code = (int)low_codes[code_index] | (high_bit << 8);
            const int sign = ((sign_bits >> lane_in) & 1u) == 0u ? 1 : -1;
            const int grid_index = code * 4 + (lane_in & 3);
            const float value = group_scale * (float)grid[grid_index] * (float)sign;
            #pragma unroll
            for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
                if (m < members) {
                    acc[m] += value * input[(long long)m * input_size + block_index * 256 + group * 32 + sub * 8 + lane_in];
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
        if (m < members) {
            // Every lane participates in the shuffle reduction; lane 0
            // holds the full sum and writes it.
            const float total = warp_sum(acc[m]);
            if (lane == 0) {
                output[(long long)m * output_size + row] = total;
            }
        }
    }
}
