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

extern "C" __global__ void iq4_xs_gemv_warp_batch(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size,
    int members
) {
    // Weights-read-once: one warp per weight row; each decoded nibble is
    // reused against every member's input.
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int row = (int)(blockIdx.x * warps_per_block + (threadIdx.x >> 5));
    const int lane = (int)(threadIdx.x & 31u);
    if (row >= output_size) {
        return;
    }
    const signed char values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };
    // Constant initializer plus fully unrolled predicated member loops keep
    // acc[] in registers; a runtime-bounded member loop spills it to local
    // memory and serializes the member input loads.
    float acc[MAX_BATCH_MEMBERS] = {0.0f};
    const int blocks_per_output = input_size / 256;
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
            const float value = group_scale * (float)values[nibble];
            #pragma unroll
            for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
                if (m < members) {
                    acc[m] += value * input[(long long)m * input_size + block_index * 256 + local];
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
