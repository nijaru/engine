// Experimental Q6_K weights x Q8_1 activations. One warp owns an output
// row; sixteen lanes each own one (group, half) tile of 16 values.
//
// Q6_K blocks are 210 bytes: 128 low-nibble bytes, 64 high-bit bytes (two
// bits per quant), 16 signed scale bytes (one per group-half), then F16 d.
// Quants are signed as (low | (high << 4)) - 32, so the integer dot needs
// no affine minimum term: the -32 bias folds into exact packed-activation
// half sums scaled by 32, computed with ones-vector dp4a over the same
// activation words as the dots.
__device__ __forceinline__ float q6_q8_1_f16_to_f32(unsigned short bits) {
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

extern "C" __global__ void q6_k_q8_1_gemv(
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
    float accumulator = 0.0f;
    // Sixteen (group, half) tiles cover the eight groups; lanes 16..31 idle.
    // One tile owns sixteen contiguous values, so every weight and activation
    // word is loaded exactly once per block with no cross-lane sharing.
    if (lane < 16u) {
        const unsigned int group = lane / 2u;
        const unsigned int half = lane & 1u;
        const unsigned int variant = group & 3u;
        const unsigned int low_base = (group / 4u) * 64u + (variant & 1u) * 32u;
        const unsigned int high_base = 128u + (group / 4u) * 32u;
        const unsigned int low_shift = variant < 2u ? 0u : 4u;
        const unsigned int high_shift = variant * 2u;
        for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
            const unsigned char* block = weights + (row * blocks_per_row + block_index) * 210u;
            const float d = q6_q8_1_f16_to_f32(
                (unsigned short)block[208] | ((unsigned short)block[209] << 8u));
            const float group_scale =
                (float)((signed char)block[192u + group * 2u + half]);
            const unsigned int* activation = input + (block_index * 8u + group) * 9u;
            const float activation_scale = q6_q8_1_f16_to_f32((unsigned short)activation[0]);
            int integer_dot = 0;
            int activation_sum = 0;
            #pragma unroll
            for (unsigned int chunk = 0; chunk < 4u; ++chunk) {
                // All loads stay four-byte aligned: every base is a multiple
                // of 32 and each half-span/chunk step a multiple of four.
                const unsigned int packed_low = *(const unsigned int*)(
                    block + low_base + half * 16u + chunk * 4u);
                const unsigned int packed_high = *(const unsigned int*)(
                    block + high_base + half * 16u + chunk * 4u);
                const unsigned int activation_word = activation[1u + half * 4u + chunk];
                const unsigned int low = (packed_low >> low_shift) & 0x0f0f0f0fu;
                const unsigned int high = (packed_high >> high_shift) & 0x03030303u;
                const unsigned int quants = low | (high << 4u);
                integer_dot += __dp4a((int)quants, (int)activation_word, 0);
                activation_sum += __dp4a(0x01010101, (int)activation_word, 0);
            }
            accumulator +=
                d * group_scale * activation_scale * (float)(integer_dot - 32 * activation_sum);
        }
    }
    accumulator = warp_sum(accumulator);
    if (lane == 0u) output[row] = accumulator;
}
