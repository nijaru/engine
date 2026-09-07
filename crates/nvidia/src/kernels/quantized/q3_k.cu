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

extern "C" __global__ void q3_k_gemv_warp_batch(
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
            const float value = d * (float)group_scale * (float)quantized;
            #pragma unroll
            for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
                if (m < members) {
                    acc[m] += value * input[(long long)m * input_size + block_index * 256 + group * 16 + within];
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
