extern "C" __device__ __forceinline__ float decode_f16(unsigned short bits) {
    const int sign = (bits & 0x8000u) != 0u ? -1 : 1;
    const int exponent = (bits >> 10u) & 0x1fu;
    const int fraction = bits & 0x03ffu;
    if (exponent == 0) {
        return (float)sign * ((float)fraction / 1024.0f) * 0.00006103515625f;
    }
    if (exponent == 31) {
        if (fraction == 0) {
            return sign > 0 ? __int_as_float(0x7f800000) : __int_as_float(0xff800000);
        }
        return __int_as_float(0x7fc00000);
    }
    float scale = 1.0f;
    int shift = exponent - 15;
    if (shift > 0) {
        for (int i = 0; i < shift; ++i) {
            scale *= 2.0f;
        }
    } else {
        for (int i = 0; i > shift; --i) {
            scale *= 0.5f;
        }
    }
    return (float)sign * (1.0f + (float)fraction / 1024.0f) * scale;
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
