// Experimental Q8_0 weights x Q8_1 activations. One warp owns an output
// row; each lane covers one packed four-byte chunk of the 32-value block.
// Q8_0 stores plain signed bytes plus one F16 scale, so no nibble unpacking
// or sign rebias is needed — the payload words feed __dp4a directly.

// Block payload sits at byte 2 of a 34-byte block, so payload words are
// never guaranteed u32-aligned (34 is not a multiple of 4). Assemble
// weight words from bytes; activation words stay direct u32 loads
// (nine-word Q8_1 blocks are always aligned).
__device__ __forceinline__ unsigned int q8_0_q8_1_load_u32(
    const unsigned char* p
) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8u)
        | ((unsigned int)p[2] << 16u) | ((unsigned int)p[3] << 24u);
}

// Fast F16 -> F32 for quantization parameters: rebias the exponent with
// bit manipulation instead of the loop-based decoder, like the other
// experimental integer-dot kernels. Kept local so the qualified float
// families keep their existing decoder untouched.
__device__ __forceinline__ float q8_0_q8_1_f16_to_f32(unsigned short bits) {
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
extern "C" __global__ void q8_0_q8_1_gemv(
    const unsigned char* weights,
    const unsigned int* input,
    float* output,
    unsigned int input_size,
    unsigned int output_size
) {
    const unsigned int row = blockIdx.x * 4u + threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    if (row >= output_size) return;
    const unsigned int blocks_per_row = input_size / 32u;
    const unsigned int chunk = lane & 7u;
    float accumulator = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        // Q8_0 block: 2-byte F16 scale, then 32 signed bytes. Payload words
        // are assembled from bytes because the 34-byte block stride never
        // guarantees u32 alignment of the payload.
        const unsigned char* block = weights + (row * blocks_per_row + block_index) * 34u;
        const float d = q8_0_q8_1_f16_to_f32(
            (unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned int q_weight =
            q8_0_q8_1_load_u32(block + 2u + chunk * 4u);
        const unsigned int* activation = input + block_index * 9u;
        const int dot = __dp4a((int)q_weight, (int)activation[1u + chunk], 0);
        const float activation_scale = q8_0_q8_1_f16_to_f32((unsigned short)activation[0]);
        accumulator += d * activation_scale * (float)dot;
    }
    accumulator = warp_sum(accumulator);
    if (lane == 0u) output[row] = accumulator;
}

// Batched weights-read-once variant: one warp owns a weight row and reuses
// each decoded weight word against every member's packed activations.
// Packed inputs and float outputs are batch-major ([member][row]); each
// member carries its own activation scales.
extern "C" __global__ void q8_0_q8_1_gemv_batch(
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
    const unsigned int blocks_per_row = input_size / 32u;
    const unsigned int packed_stride = input_size / 32u * 9u;
    const unsigned int chunk = lane & 7u;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char* block = weights + (row * blocks_per_row + block_index) * 34u;
        const float d = q8_0_q8_1_f16_to_f32(
            (unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const unsigned int q_weight =
            q8_0_q8_1_load_u32(block + 2u + chunk * 4u);
        #pragma unroll
        for (unsigned int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
            if (m < members) {
                const unsigned int* activation =
                    input + m * packed_stride + block_index * 9u;
                const int dot = __dp4a((int)q_weight, (int)activation[1u + chunk], 0);
                const float activation_scale =
                    q8_0_q8_1_f16_to_f32((unsigned short)activation[0]);
                acc[m] += d * activation_scale * (float)dot;
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
