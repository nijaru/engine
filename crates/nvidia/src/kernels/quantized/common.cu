// IEEE binary16 to binary32. Every binary16 value is exactly representable
// in binary32, so the single convert instruction is bit-identical to an
// explicit sign/exponent/fraction expansion while avoiding the per-block
// shift loop that dominated the 32-element block families.
extern "C" __device__ __forceinline__ float decode_f16(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

extern "C" __device__ __forceinline__ int scale_value(const unsigned char* block, int group) {
    if (group < 4) {
        return (int)(block[4 + group] & 0x3fu);
    }
    const int index = group - 4;
    return (int)((block[12 + index] & 0x0fu) | ((block[4 + index] >> 2u) & 0x30u));
}

extern "C" __device__ __forceinline__ int minimum_value(const unsigned char* block, int group) {
    if (group < 4) {
        return (int)(block[8 + group] & 0x3fu);
    }
    const int index = group - 4;
    return (int)((block[12 + index] >> 4u) | ((block[8 + index] >> 2u) & 0x30u));
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Four bytes of an encoded block as one word. The 256-element K-quant block
// strides are 110 and 210 bytes, so a word inside a block is 4-byte aligned for
// only half the rows: read the window as two halfwords instead of relying on a
// `unsigned int` load that would fault on the odd rows.
__device__ __forceinline__ unsigned int load_block_word(
    const unsigned char* base,
    int offset
) {
    const unsigned short* halves = (const unsigned short*)(base + offset);
    return (unsigned int)halves[0] | ((unsigned int)halves[1] << 16u);
}
