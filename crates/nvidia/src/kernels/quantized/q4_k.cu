extern "C" __global__ void q4_k_embedding(
    const unsigned char* weights,
    unsigned int token_index,
    float* output,
    int hidden_size,
    int vocabulary_size
) {
    const int hidden_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (hidden_index >= hidden_size || token_index >= (unsigned int)vocabulary_size) {
        return;
    }

    const int blocks_per_token = hidden_size / 256;
    const int block_index = hidden_index / 256;
    const int local = hidden_index & 255;
    const unsigned char* block =
        weights + ((int)token_index * blocks_per_token + block_index) * 144;
    const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
    const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
    const int group = local / 32;
    const int index = local & 31;
    const int data_offset = 16 + (group / 2) * 32;
    const int shift = (group & 1) * 4;
    const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
    output[hidden_index] =
        d * (float)scale_value(block, group) * (float)quantized
        - min * (float)minimum_value(block, group);
}

extern "C" __global__ void q4_k_embedding_batch(
    const unsigned char* weights,
    const unsigned int* token_indices,
    float* output,
    int hidden_size,
    int vocabulary_size,
    int members
) {
    // One thread per (member, hidden element): tokens come from a [members]
    // device array and output rows are batch-major [members][hidden]. The
    // per-element decode math is identical to the single-token kernel.
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= members * hidden_size) {
        return;
    }
    const int member = index / hidden_size;
    const int hidden_index = index - member * hidden_size;
    const unsigned int token_index = token_indices[member];
    if (token_index >= (unsigned int)vocabulary_size) {
        return;
    }

    const int blocks_per_token = hidden_size / 256;
    const int block_index = hidden_index / 256;
    const int local = hidden_index & 255;
    const unsigned char* block =
        weights + ((int)token_index * blocks_per_token + block_index) * 144;
    const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
    const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
    const int group = local / 32;
    const int index_within = local & 31;
    const int data_offset = 16 + (group / 2) * 32;
    const int shift = (group & 1) * 4;
    const int quantized = (int)((block[data_offset + index_within] >> shift) & 0x0fu);
    output[index] =
        d * (float)scale_value(block, group) * (float)quantized
        - min * (float)minimum_value(block, group);
}

extern "C" __global__ void q4_k_gemv(
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
            weights + (output_index * blocks_per_output + block_index) * 144;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        for (int local = 0; local < 256; ++local) {
            const int group = local / 32;
            const int index = local & 31;
            const int data_offset = 16 + (group / 2) * 32;
            const int shift = (group & 1) * 4;
            const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const float value =
                d * (float)scale_value(block, group) * (float)quantized
                - min * (float)minimum_value(block, group);
            accumulator += value * input[block_index * 256 + local];
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q4_k_gemv_warp(
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
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 144;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        const int group = lane >> 2;
        const int index_base = (lane & 3) * 8;
        const int data_offset = 16 + (group / 2) * 32;
        const int shift = (group & 1) * 4;
        const float group_scale = (float)scale_value(block, group);
        const float group_minimum = (float)minimum_value(block, group);
        for (int j = 0; j < 8; ++j) {
            const int index = index_base + j;
            const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const float value =
                d * group_scale * (float)quantized - min * group_minimum;
            accumulator += value * input[block_index * 256 + group * 32 + index];
        }
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) {
        output[row] = total;
    }
}

extern "C" __global__ void q4_k_gemv_warp_batch(
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
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (row * blocks_per_output + block_index) * 144;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        const int group = lane >> 2;
        const int index_base = (lane & 3) * 8;
        const int data_offset = 16 + (group / 2) * 32;
        const int shift = (group & 1) * 4;
        const float group_scale = (float)scale_value(block, group);
        const float group_minimum = (float)minimum_value(block, group);
        for (int j = 0; j < 8; ++j) {
            const int index = index_base + j;
            const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const float value =
                d * group_scale * (float)quantized - min * group_minimum;
            #pragma unroll
            for (int m = 0; m < MAX_BATCH_MEMBERS; ++m) {
                if (m < members) {
                    acc[m] += value * input[(long long)m * input_size + block_index * 256 + group * 32 + index];
                }
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
