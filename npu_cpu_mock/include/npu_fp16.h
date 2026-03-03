#ifndef NPU_FP16_H
#define NPU_FP16_H

#include <stdint.h>
#include <string.h>

/* IEEE 754 fp16 <-> float bit-level conversion, no compiler __fp16 dependency */

static inline uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03FF;

    if (((x >> 23) & 0xFF) == 0xFF) {          /* inf / nan */
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    if (exp <= 0) {                             /* underflow -> zero */
        return (uint16_t)sign;
    }
    if (exp >= 31) {                            /* overflow  -> inf  */
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (exp << 10) | mant);
}

static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t out;

    if (exp == 0) {                             /* zero / subnormal */
        if (mant == 0) { out = sign; }
        else {
            exp = 1;
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {                     /* inf / nan */
        out = sign | 0x7F800000 | (mant << 13);
    } else {                                    /* normal */
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &out, 4);
    return result;
}

#endif /* NPU_FP16_H */
