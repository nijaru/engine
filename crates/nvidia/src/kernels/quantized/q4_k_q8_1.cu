// Experimental Q4_K weights x Q8_1 activations. One warp owns an output
// row. Eight lanes cover each 32-value group in packed four-byte chunks.
extern "C" __global__ void q4_k_q8_1_gemv(
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
        const unsigned char* block = weights + (row * blocks_per_row + block_index) * 144u;
        const unsigned int header = *(const unsigned int*)block;
        const float d = decode_f16((unsigned short)header);
        const float minimum = decode_f16((unsigned short)(header >> 16));
        #pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const unsigned int group = half * 4u + lane / 8u;
            // All packed loads are four-byte aligned: Q4_K blocks are 144
            // bytes, payload starts at byte 16, each group spans 32 bytes.
            const unsigned int packed_weight = *(const unsigned int*)(
                block + 16u + (group / 2u) * 32u + chunk * 4u);
            const unsigned int q_weight = (packed_weight >> ((group & 1u) * 4u)) & 0x0f0f0f0fu;
            const unsigned int* activation = input + (block_index * 8u + group) * 9u;
            const int dot = __dp4a((int)q_weight, (int)activation[1u + chunk], 0);
            const float activation_scale = decode_f16((unsigned short)activation[0]);
            accumulator += d * (float)scale_value(block, (int)group) * activation_scale * (float)dot;
            if (chunk == 0u) {
                // Q8_1 stores half(sum(original input)), not scale * sum(q).
                // Apply the affine minimum correction once per group.
                const float input_sum = decode_f16((unsigned short)(activation[0] >> 16));
                accumulator -= minimum * (float)minimum_value(block, (int)group) * input_sum;
            }
        }
    }
    accumulator = warp_sum(accumulator);
    if (lane == 0u) output[row] = accumulator;
}
