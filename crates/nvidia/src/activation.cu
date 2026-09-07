// One warp owns each contiguous 32-value Q8_1 block. The first word
// contains half(d), half(sum(x)); the next eight contain signed int8s.
extern "C" __global__ void quantize_q8_1(
    const float* input, unsigned int* output, unsigned int blocks
) {
    const unsigned int block = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    if (block >= blocks) return;
    const float x = input[block * 32 + lane];
    float maximum = fabsf(x);
    float sum = x;
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    const float d = maximum / 127.0f;
    const int q = maximum == 0.0f ? 0 : (int)roundf(x / d);
    unsigned int packed = (unsigned int)(q & 255);
    packed |= __shfl_down_sync(0xffffffff, packed, 1) << 8;
    packed |= __shfl_down_sync(0xffffffff, packed, 2) << 16;
    if ((lane & 3) == 0) output[block * 9 + 1 + lane / 4] = packed;
    if (lane == 0) {
        unsigned short dh, sh;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(dh) : "f"(d));
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(sh) : "f"(sum));
        output[block * 9] = (unsigned int)dh | ((unsigned int)sh << 16);
    }
}
