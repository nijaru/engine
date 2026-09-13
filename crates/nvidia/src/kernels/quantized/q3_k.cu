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
    // Same per-lane element mapping as the batched variant, so both paths
    // accumulate identical partial sums in identical order: chunked prefill
    // stays bit-identical to serial prefill while this path's activation reads
    // coalesce like the batched kernel's.
    const int group_low = lane >> 2;
    const int index = (lane & 3) * 4;
    const int low_variant = group_low & 1;
    const int low_shift = (group_low >> 1) * 2;
    const int low_offset = low_variant * 16 + index;
    const int high_offset = low_variant * 16 + index;
    const int high_shift_low = group_low >> 1;
    const int high_shift_high = high_shift_low + 4;
    const int chunk = group_low >> 2;
    const int packed_index = group_low & 3;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 110;
        const unsigned char* high_bits = block;
        const unsigned char* low_bits = block + 32;
        const unsigned char* packed_scales = block + 96;
        const float d = decode_f16((unsigned short)block[108] | ((unsigned short)block[109] << 8u));
        const int scale_low = (int)((packed_scales[group_low] & 0x0fu)
            | (((packed_scales[8 + packed_index] >> (chunk * 2)) & 0x03u) << 4u)) - 32;
        const int scale_high = (int)((packed_scales[group_low] >> 4u)
            | (((packed_scales[8 + packed_index] >> (chunk * 2 + 4)) & 0x03u) << 4u)) - 32;
        const unsigned int packed_low = load_block_word(low_bits, low_offset);
        const unsigned int packed_high = load_block_word(low_bits, low_offset + 32);
        const unsigned int high_word = load_block_word(high_bits, high_offset);
        float value_low[4];
        float value_high[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int low_low = (int)((packed_low >> (8 * t + low_shift)) & 0x03u);
            const int low_high = (int)((packed_high >> (8 * t + low_shift)) & 0x03u);
            const int high_low =
                ((int)((high_word >> (8 * t + high_shift_low)) & 1u) ^ 1);
            const int high_high =
                ((int)((high_word >> (8 * t + high_shift_high)) & 1u) ^ 1);
            value_low[t] = d * (float)scale_low * (float)(low_low - high_low * 4);
            value_high[t] = d * (float)scale_high * (float)(low_high - high_high * 4);
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
    // Each lane owns four consecutive activations in the low half of the
    // block (elements `4*lane`) and four in the high half (elements
    // `128 + 4*lane`). Both runs are 16-byte aligned, so one member's
    // activations are two `float4` loads per lane and a warp load covers 512
    // contiguous bytes; the strided byte-per-element form it replaced spent
    // eight wavefronts per 1 KiB of activations on sector transactions alone.
    // Q3_K groups hold 16 elements, so lane `l` decodes group `l / 4` at the
    // group position `4 * (l % 4)`: two low bits per element from one word of
    // the low plane and one high bit per element from one word of the mask.
    // The high half is the next chunk of both planes with the same group
    // position, and the paired group shares its scale byte.
    const int group = lane >> 2;
    const int index = (lane & 3) * 4;
    const int low_variant = group & 1;
    const int low_shift = (group >> 1) * 2;
    const int low_offset = low_variant * 16 + index;
    const int high_offset = low_variant * 16 + index;
    const int high_shift_low = group >> 1;
    const int high_shift_high = high_shift_low + 4;
    const int chunk = group >> 2;
    const int packed_index = group & 3;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 110;
        const float d = decode_f16((unsigned short)block[108] | ((unsigned short)block[109] << 8u));
        const unsigned char* high_bits = block;
        const unsigned char* low_bits = block + 32;
        const unsigned char* packed_scales = block + 96;
        const int scale_low = (int)((packed_scales[group] & 0x0fu)
            | (((packed_scales[8 + packed_index] >> (chunk * 2)) & 0x03u) << 4u)) - 32;
        const int scale_high = (int)((packed_scales[group] >> 4u)
            | (((packed_scales[8 + packed_index] >> (chunk * 2 + 4)) & 0x03u) << 4u)) - 32;
        const unsigned int packed_low = load_block_word(low_bits, low_offset);
        const unsigned int packed_high = load_block_word(low_bits, low_offset + 32);
        const unsigned int high_word = load_block_word(high_bits, high_offset);
        float value_low[4];
        float value_high[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int low_low = (int)((packed_low >> (8 * t + low_shift)) & 0x03u);
            const int low_high = (int)((packed_high >> (8 * t + low_shift)) & 0x03u);
            const int high_low =
                ((int)((high_word >> (8 * t + high_shift_low)) & 1u) ^ 1);
            const int high_high =
                ((int)((high_word >> (8 * t + high_shift_high)) & 1u) ^ 1);
            value_low[t] = d * (float)scale_low * (float)(low_low - high_low * 4);
            value_high[t] = d * (float)scale_high * (float)(low_high - high_high * 4);
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
