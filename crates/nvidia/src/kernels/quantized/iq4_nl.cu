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
    // Same per-lane element mapping as the batched variant, so both paths
    // accumulate identical partial sums in identical order: chunked prefill
    // stays bit-identical to serial prefill while this path's activation reads
    // coalesce like the batched kernel's.
    const int block_group = lane >> 3;
    const int index = (lane & 7) * 4;
    const signed char values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };
    const int nibble_shift = index < 16 ? 0 : 4;
    const int nibble_offset = index < 16 ? index : index - 16;
    float accumulator = 0.0f;
    for (int base = 0; base < blocks_per_output; base += 4) {
        const int block_index = base + block_group;
        if (block_index >= blocks_per_output) {
            continue;
        }
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 18;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned short low_half = *(const unsigned short*)(block + 2 + nibble_offset);
        const unsigned short high_half =
            *(const unsigned short*)(block + 4 + nibble_offset);
        const float value0 =
            d * (float)values[(int)((low_half >> nibble_shift) & 0x0fu)];
        const float value1 =
            d * (float)values[(int)((low_half >> (nibble_shift + 8)) & 0x0fu)];
        const float value2 =
            d * (float)values[(int)((high_half >> nibble_shift) & 0x0fu)];
        const float value3 =
            d * (float)values[(int)((high_half >> (nibble_shift + 8)) & 0x0fu)];
        const float4 activations = *(const float4*)(input + block_index * 32 + index);
        accumulator += value0 * activations.x + value1 * activations.y
            + value2 * activations.z + value3 * activations.w;
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
    // Weights-read-once: one warp per weight row; the decoded nibbles are
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
    // One lane owns four consecutive activations of its own block: lanes
    // `8k..8k+7` cover block `base + k`. A member's activations are then a
    // single `float4` load per lane instead of four stride-32 scalar loads,
    // which is what the load:FMA ratio of this family turns on. Encoded
    // blocks are 18 bytes, so the packed nibbles are read as two aligned
    // `unsigned short` halves.
    const int block_group = lane >> 3;
    const int index = (lane & 7) * 4;
    const int nibble_shift = index < 16 ? 0 : 4;
    const int nibble_offset = index < 16 ? index : index - 16;
    for (int base = 0; base < blocks_per_output; base += 4) {
        const int block_index = base + block_group;
        if (block_index >= blocks_per_output) {
            continue;
        }
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 18;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned short low_half = *(const unsigned short*)(block + 2 + nibble_offset);
        const unsigned short high_half =
            *(const unsigned short*)(block + 4 + nibble_offset);
        const float value0 =
            d * (float)values[(int)((low_half >> nibble_shift) & 0x0fu)];
        const float value1 =
            d * (float)values[(int)((low_half >> (nibble_shift + 8)) & 0x0fu)];
        const float value2 =
            d * (float)values[(int)((high_half >> nibble_shift) & 0x0fu)];
        const float value3 =
            d * (float)values[(int)((high_half >> (nibble_shift + 8)) & 0x0fu)];
        const float* member_input = input + block_index * 32 + index;
        #pragma unroll
        for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
            if (m < members) {
                const float4* row_input =
                    (const float4*)(member_input + (long long)m * input_size);
                const float4 activations = *row_input;
                acc[m] += value0 * activations.x + value1 * activations.y
                    + value2 * activations.z + value3 * activations.w;
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
