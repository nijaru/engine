// Experimental Q5_K weights x Q8_1 activations. One warp owns an output
// row. Eight lanes cover each 32-value group in packed four-byte chunks.
//
// Q5_K blocks are 176 bytes: F16 d/minimum, the same 12-byte 6-bit
// scale/minimum packing as Q4_K, a 32-byte high-bit plane (byte 16+i holds
// bit g for position i of every group g), then 128 bytes of low nibbles.
// Each 5-bit quant is low | (high << 4) and lands in one dp4a lane byte.
__device__ __forceinline__ float q5_q8_1_f16_to_f32(unsigned short bits) {
    const unsigned int sign = (unsigned int)(bits >> 15) & 1u;
    const unsigned int exponent = ((unsigned int)(bits >> 10) & 0x1fu);
    const unsigned int fraction = (unsigned int)(bits & 0x3ffu);
    if (exponent == 0u) {
        if (fraction == 0u) {
            return sign ? -0.0f : 0.0f;
        }
        int shift = -1;
        unsigned int value = fraction;
        while (value != 0u) {
            value >>= 1u;
            ++shift;
        }
        const unsigned int normalized = (fraction << (10 - shift)) & 0x3ffu;
        const int new_exponent = -24 + shift + 127;
        const unsigned int result = (sign << 31)
            | ((unsigned int)new_exponent << 23) | (normalized << 13);
        return __int_as_float(result);
    }
    if (exponent == 31u) {
        const unsigned int result =
            (sign << 31) | 0x7f800000u | (fraction ? 0x7fc00000u : 0u);
        return __int_as_float(result);
    }
    const unsigned int result = (sign << 31)
        | ((exponent - 15u + 127u) << 23)
        | (fraction << 13);
    return __int_as_float(result);
}

extern "C" __global__ void q5_k_q8_1_gemv(
    const unsigned char* weights,
    const unsigned int* input,
    float* output,
    unsigned int input_size,
    unsigned int output_size
) {
    const unsigned int row = blockIdx.x * 4u + threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    if (row >= output_size) return;
    const unsigned int blocks_per_row = input_size / 256u;
    const unsigned int chunk = lane & 7u;
    float accumulator = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char* block = weights + (row * blocks_per_row + block_index) * 176u;
        const unsigned int header = *(const unsigned int*)block;
        const float d = q5_q8_1_f16_to_f32((unsigned short)header);
        const float minimum = q5_q8_1_f16_to_f32((unsigned short)(header >> 16));
        #pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const unsigned int group = half * 4u + lane / 8u;
            // Low nibbles mirror the Q4_K payload at base 48; the high-bit
            // plane at base 16 packs bit g for positions 0..31 across groups.
            // Both loads stay four-byte aligned: 176-byte blocks with payload
            // bases and per-group spans that are multiples of four.
            const unsigned int packed_low = *(const unsigned int*)(
                block + 48u + (group / 2u) * 32u + chunk * 4u);
            const unsigned int high_bits = *(const unsigned int*)(block + 16u + chunk * 4u);
            const unsigned int q_low = (packed_low >> ((group & 1u) * 4u)) & 0x0f0f0f0fu;
            const unsigned int q_high = (((high_bits >> group) & 0x01010101u) << 4);
            const unsigned int q_weight = q_low | q_high;
            const unsigned int* activation = input + (block_index * 8u + group) * 9u;
            const int dot = __dp4a((int)q_weight, (int)activation[1u + chunk], 0);
            const float activation_scale = q5_q8_1_f16_to_f32((unsigned short)activation[0]);
            accumulator += d * (float)scale_value(block, (int)group) * activation_scale * (float)dot;
            if (chunk == 0u) {
                // Q8_1 stores half(sum(original input)), not scale * sum(q).
                // Apply the affine minimum correction once per group.
                const float input_sum = q5_q8_1_f16_to_f32((unsigned short)(activation[0] >> 16));
                accumulator -= minimum * (float)minimum_value(block, (int)group) * input_sum;
            }
        }
    }
    accumulator = warp_sum(accumulator);
    if (lane == 0u) output[row] = accumulator;
}

// Batched weights-read-once variant: one warp owns a weight row and
// reuses each decoded element against every member's packed activations.
// Packed inputs and float outputs are batch-major ([member][row]); each
// member carries its own activation scales and input sums.
extern "C" __global__ void q5_k_q8_1_gemv_batch(
    const unsigned char* weights,
    const unsigned int* input,
    float* output,
    unsigned int input_size,
    unsigned int output_size,
    unsigned int members
) {
    const unsigned int row = blockIdx.x * 4u + threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    if (row >= output_size) return;
    // Constant initializer plus fully unrolled predicated member loops keep
    // acc[] in registers; a runtime-bounded member loop would spill it to
    // local memory and serialize the member input loads.
    float acc[MAX_BATCH_MEMBERS] = {0.0f};
    const unsigned int blocks_per_row = input_size / 256u;
    const unsigned int packed_stride = input_size / 32u * 9u;
    const unsigned int chunk = lane & 7u;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char* block = weights + (row * blocks_per_row + block_index) * 176u;
        const unsigned int header = *(const unsigned int*)block;
        const float d = q5_q8_1_f16_to_f32((unsigned short)header);
        const float minimum = q5_q8_1_f16_to_f32((unsigned short)(header >> 16));
        #pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const unsigned int group = half * 4u + lane / 8u;
            const unsigned int packed_low = *(const unsigned int*)(
                block + 48u + (group / 2u) * 32u + chunk * 4u);
            const unsigned int high_bits = *(const unsigned int*)(block + 16u + chunk * 4u);
            const unsigned int q_low = (packed_low >> ((group & 1u) * 4u)) & 0x0f0f0f0fu;
            const unsigned int q_high = (((high_bits >> group) & 0x01010101u) << 4);
            const unsigned int q_weight = q_low | q_high;
            const float scale = (float)scale_value(block, (int)group);
            const float minval = (float)minimum_value(block, (int)group);
            #pragma unroll
            for (unsigned int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
                if (m < members) {
                    const unsigned int* activation =
                        input + m * packed_stride + (block_index * 8u + group) * 9u;
                    const int dot = __dp4a((int)q_weight, (int)activation[1u + chunk], 0);
                    const float activation_scale =
                        q5_q8_1_f16_to_f32((unsigned short)activation[0]);
                    acc[m] += d * scale * activation_scale * (float)dot;
                    if (chunk == 0u) {
                        const float input_sum =
                            q5_q8_1_f16_to_f32((unsigned short)(activation[0] >> 16));
                        acc[m] -= minimum * minval * input_sum;
                    }
                }
            }
        }
    }
    #pragma unroll
    for (unsigned int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
        if (m < members) {
            // Every lane participates in the shuffle reduction; lane 0
            // holds the full sum and writes it.
            const float total = warp_sum(acc[m]);
            if (lane == 0u) output[m * output_size + row] = total;
        }
    }
}
