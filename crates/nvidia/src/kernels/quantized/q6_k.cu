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

extern "C" __global__ void q6_k_gemv_warp_batch(
    const unsigned char* weights,
    const float* input,
    float* output,
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
            const float value = d * (float)group_scale * (float)quantized;
            #pragma unroll
            for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
                if (m < members) {
                    acc[m] += value * input[(long long)m * input_size + block_index * 256 + group * 32 + position];
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
