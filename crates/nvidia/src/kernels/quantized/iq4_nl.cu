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

extern "C" __global__ void iq4_nl_gemv_warp_batch(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size,
    int members
) {
    // Weights-read-once: one warp per weight row; the decoded nibble is
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
    const int blocks_per_output = input_size / 32;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 18;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const int index = lane < 16 ? lane : lane - 16;
        const int nibble = lane < 16
            ? (int)(block[2 + index] & 0x0fu)
            : (int)(block[2 + index] >> 4u);
        const float value = d * (float)values[nibble];
        #pragma unroll
        for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
            if (m < members) {
                acc[m] += value * input[(long long)m * input_size + block_index * 32 + lane];
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
