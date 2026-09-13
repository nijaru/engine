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
    // Same per-lane element mapping as the batched variant, so both paths
    // accumulate identical partial sums in identical order: chunked prefill
    // stays bit-identical to serial prefill while this path's activation reads
    // coalesce like the batched kernel's.
    const int group_low = lane >> 3;
    const int index = (lane & 7) * 4;
    const int data_offset = 48 + (group_low >> 1) * 32 + index;
    const int shift = (group_low & 1) * 4;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 176;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        const float scale_low = (float)scale_value(block, group_low);
        const float minimum_low = (float)minimum_value(block, group_low);
        const float scale_high = (float)scale_value(block, group_low + 4);
        const float minimum_high = (float)minimum_value(block, group_low + 4);
        const unsigned int packed_low = *(const unsigned int*)(block + data_offset);
        const unsigned int packed_high = *(const unsigned int*)(block + data_offset + 64);
        const unsigned int high_bits = *(const unsigned int*)(block + 16 + index);
        float value_low[4];
        float value_high[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int low_low = (int)((packed_low >> (8 * t + shift)) & 0x0fu);
            const int low_high = (int)((packed_high >> (8 * t + shift)) & 0x0fu);
            const int quantized_low =
                low_low | ((int)((high_bits >> (8 * t + group_low)) & 1u) << 4);
            const int quantized_high =
                low_high | ((int)((high_bits >> (8 * t + group_low + 4)) & 1u) << 4);
            value_low[t] = d * scale_low * (float)quantized_low - min * minimum_low;
            value_high[t] = d * scale_high * (float)quantized_high - min * minimum_high;
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

extern "C" __global__ void q5_k_gemv_warp_batch(
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
    const int group_low = lane >> 3;
    const int index = (lane & 7) * 4;
    const int data_offset = 48 + (group_low >> 1) * 32 + index;
    const int shift = (group_low & 1) * 4;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 176;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        const float scale_low = (float)scale_value(block, group_low);
        const float minimum_low = (float)minimum_value(block, group_low);
        const float scale_high = (float)scale_value(block, group_low + 4);
        const float minimum_high = (float)minimum_value(block, group_low + 4);
        // Two words of packed low nibbles and one word of high-bit plane cover
        // every element this lane owns; the plane byte at `index + t` carries
        // one bit per group, so the low and high halves read different bits of
        // the same byte.
        const unsigned int packed_low = *(const unsigned int*)(block + data_offset);
        const unsigned int packed_high = *(const unsigned int*)(block + data_offset + 64);
        const unsigned int high_bits = *(const unsigned int*)(block + 16 + index);
        float value_low[4];
        float value_high[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int low_low = (int)((packed_low >> (8 * t + shift)) & 0x0fu);
            const int low_high = (int)((packed_high >> (8 * t + shift)) & 0x0fu);
            const int quantized_low =
                low_low | ((int)((high_bits >> (8 * t + group_low)) & 1u) << 4);
            const int quantized_high =
                low_high | ((int)((high_bits >> (8 * t + group_low + 4)) & 1u) << 4);
            value_low[t] = d * scale_low * (float)quantized_low - min * minimum_low;
            value_high[t] = d * scale_high * (float)quantized_high - min * minimum_high;
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
