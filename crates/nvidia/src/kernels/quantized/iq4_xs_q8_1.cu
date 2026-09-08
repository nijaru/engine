// Experimental IQ4_XS weights x Q8_1 activations. One warp owns an output
// row; eight groups of four lanes each own eight consecutive values.
//
// IQ4_XS blocks are 136 bytes: F16 d, a u16 of per-group high scale bits,
// four bytes of low scale nibbles, then eight 16-byte groups. Each byte
// holds two 4-bit codebook indices, resolved through a 16-entry signed
// grid. The grid lives in shared memory with eight replicas (128 bytes)
// so the four lanes sharing a replica rarely bank-conflict; replicas are
// filled once per threadblock before the row guard so no early-exiting
// warp can skip the barrier. Scales are small integers, so the only F16
// decodes are d and the per-group activation scale.
__device__ __forceinline__ float iq4_xs_q8_1_f16_to_f32(unsigned short bits) {
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

extern "C" __global__ void iq4_xs_q8_1_gemv(
    const unsigned char* weights,
    const unsigned int* input,
    float* output,
    unsigned int input_size,
    unsigned int output_size
) {
    __shared__ unsigned char grid[128];
    {
        const unsigned char values[16] = {
            129, 152, 173, 191, 207, 221, 234, 246,
            1, 13, 25, 38, 53, 69, 89, 113
        };
        // values[] holds two's-complement bytes of the signed grid
        // (-127, -104, ..., 113); every thread fills one replica byte.
        grid[threadIdx.x] = values[threadIdx.x & 15u];
    }
    __syncthreads();

    const unsigned int row = blockIdx.x * 4u + threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    if (row >= output_size) return;
    const unsigned int blocks_per_row = input_size / 256u;
    // Four lanes share one replica: same 16 bytes, distinct banks per index
    // class, so random nibbles rarely serialize.
    const unsigned int replica = (lane >> 2u) & 7u;
    const unsigned int replica_base = replica * 16u;
    float accumulator = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char* block = weights + (row * blocks_per_row + block_index) * 136u;
        const float d = iq4_xs_q8_1_f16_to_f32(
            (unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned int high_scales =
            (unsigned int)block[2] | ((unsigned int)block[3] << 8u);
        const unsigned int group = lane >> 2u;
        const unsigned int quarter = lane & 3u;
        const int scale_low =
            (int)((block[4u + group / 2u] >> ((group & 1u) * 4u)) & 0x0fu);
        const int scale_high = (int)((high_scales >> (group * 2u)) & 0x03u);
        const float group_scale = d * (float)((scale_low | (scale_high << 4)) - 32);
        const unsigned int* activation = input + (block_index * 8u + group) * 9u;
        const float activation_scale = iq4_xs_q8_1_f16_to_f32((unsigned short)activation[0]);
        // Eight consecutive positions: two u32 words of packed nibbles and
        // two activation words. Upper quarters read the high nibbles of the
        // same words the lower quarters read as low nibbles.
        const unsigned int shift = quarter < 2u ? 0u : 4u;
        const unsigned int word_base = 8u + group * 16u + (quarter & 1u) * 8u;
        const unsigned int packed0 = *(const unsigned int*)(block + word_base);
        const unsigned int packed1 = *(const unsigned int*)(block + word_base + 4u);
        const unsigned int nibbles0 = (packed0 >> shift) & 0x0f0f0f0fu;
        const unsigned int nibbles1 = (packed1 >> shift) & 0x0f0f0f0fu;
        const unsigned int act_base = 1u + quarter * 2u;
        const unsigned int act0 = activation[act_base];
        const unsigned int act1 = activation[act_base + 1u];
        unsigned int quad0 = 0u;
        unsigned int quad1 = 0u;
        #pragma unroll
        for (unsigned int byte = 0; byte < 4u; ++byte) {
            const unsigned int n0 = (nibbles0 >> (byte * 8u)) & 0x0fu;
            const unsigned int n1 = (nibbles1 >> (byte * 8u)) & 0x0fu;
            quad0 |= (unsigned int)grid[replica_base + n0] << (byte * 8u);
            quad1 |= (unsigned int)grid[replica_base + n1] << (byte * 8u);
        }
        const int integer_dot =
            __dp4a((int)quad0, (int)act0, 0) + __dp4a((int)quad1, (int)act1, 0);
        accumulator += group_scale * activation_scale * (float)integer_dot;
    }
    accumulator = warp_sum(accumulator);
    if (lane == 0u) output[row] = accumulator;
}
