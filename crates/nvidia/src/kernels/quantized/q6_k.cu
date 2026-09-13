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
    // Same per-lane element mapping as the batched variant, so both paths
    // accumulate identical partial sums in identical order: chunked prefill
    // stays bit-identical to serial prefill while this path's activation reads
    // coalesce like the batched kernel's.
    const int group_low = lane >> 3;
    const int index = (lane & 7) * 4;
    const int half = index < 16 ? 0 : 1;
    const int low_shift = group_low < 2 ? 0 : 4;
    const int high_shift = group_low * 2;
    const int low_offset = (group_low & 1) * 32 + index;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 210;
        const unsigned char* low_bits = block;
        const unsigned char* high_bits = block + 128;
        const unsigned char* scales = block + 192;
        const float d = decode_f16((unsigned short)block[208] | ((unsigned short)block[209] << 8u));
        const int scale_low = (int)(signed char)scales[group_low * 2 + half];
        const int scale_high = (int)(signed char)scales[(group_low + 4) * 2 + half];
        const unsigned int packed_low = load_block_word(low_bits, low_offset);
        const unsigned int packed_high = load_block_word(low_bits, low_offset + 64);
        const unsigned int high_low = load_block_word(high_bits, index);
        const unsigned int high_high = load_block_word(high_bits, index + 32);
        float value_low[4];
        float value_high[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int quantized_low =
                ((int)((packed_low >> (8 * t + low_shift)) & 0x0fu)
                    | ((int)((high_low >> (8 * t + high_shift)) & 0x03u) << 4)) - 32;
            const int quantized_high =
                ((int)((packed_high >> (8 * t + low_shift)) & 0x0fu)
                    | ((int)((high_high >> (8 * t + high_shift)) & 0x03u) << 4)) - 32;
            value_low[t] = d * (float)scale_low * (float)quantized_low;
            value_high[t] = d * (float)scale_high * (float)quantized_high;
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
    // Each lane owns four consecutive activations in the low half of the
    // block (elements `4*lane`) and four in the high half (elements
    // `128 + 4*lane`). Both runs are 16-byte aligned, so one member's
    // activations are two `float4` loads per lane and a warp load covers 512
    // contiguous bytes; the strided byte-per-element form it replaced spent
    // eight wavefronts per 1 KiB of activations on sector transactions alone.
    // `group` selects the four low-bit and two high-bit planes that hold those
    // elements; the high half uses the next chunk of both planes with the same
    // in-group positions and shifts.
    const int group = lane >> 3;
    const int index = (lane & 7) * 4;
    const int half = index < 16 ? 0 : 1;
    const int low_shift = group < 2 ? 0 : 4;
    const int high_shift = group * 2;
    const int low_offset = (group & 1) * 32 + index;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 210;
        const unsigned char* low_bits = block;
        const unsigned char* high_bits = block + 128;
        const unsigned char* scales = block + 192;
        const float d = decode_f16((unsigned short)block[208] | ((unsigned short)block[209] << 8u));
        const int scale_low = (int)(signed char)scales[group * 2 + half];
        const int scale_high = (int)(signed char)scales[(group + 4) * 2 + half];
        const unsigned int packed_low = load_block_word(low_bits, low_offset);
        const unsigned int packed_high = load_block_word(low_bits, low_offset + 64);
        const unsigned int high_low = load_block_word(high_bits, index);
        const unsigned int high_high = load_block_word(high_bits, index + 32);
        float value_low[4];
        float value_high[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int quantized_low =
                ((int)((packed_low >> (8 * t + low_shift)) & 0x0fu)
                    | ((int)((high_low >> (8 * t + high_shift)) & 0x03u) << 4)) - 32;
            const int quantized_high =
                ((int)((packed_high >> (8 * t + low_shift)) & 0x0fu)
                    | ((int)((high_high >> (8 * t + high_shift)) & 0x03u) << 4)) - 32;
            value_low[t] = d * (float)scale_low * (float)quantized_low;
            value_high[t] = d * (float)scale_high * (float)quantized_high;
        }
        const float* member_input = input + block_index * 256;
        #pragma unroll
        for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
            if (m < members) {
                const float4* row_input =
                    (const float4*)(member_input + (long long)m * input_size);
                const float4 low = row_input[lane];
                const float4 high = row_input[32 + lane];
                acc[m] += value_low[0] * low.x + value_low[1] * low.y
                    + value_low[2] * low.z + value_low[3] * low.w
                    + value_high[0] * high.x + value_high[1] * high.y
                    + value_high[2] * high.z + value_high[3] * high.w;
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
