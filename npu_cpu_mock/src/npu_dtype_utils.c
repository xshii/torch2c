#include "npu_api.h"
#include "npu_fp16.h"
#include <string.h>
#include <stdint.h>

/* Global L1 base pointer — set by model_run before first op call */
unsigned char* npu_l1_base = NULL;

size_t npu_dtype_size(npu_dtype_t dtype) {
    switch (dtype) {
        case NPU_DTYPE_FP16:  return 2;
        case NPU_DTYPE_FP32:  return 4;
        case NPU_DTYPE_BF16:  return 2;
        case NPU_DTYPE_INT8:  return 1;
        case NPU_DTYPE_INT32: return 4;
        case NPU_DTYPE_INT16: return 2;
        default:              return 0;
    }
}

float npu_read_as_float(const void* buf, int index, npu_dtype_t dtype) {
    const uint8_t* p = (const uint8_t*)buf;
    switch (dtype) {
        case NPU_DTYPE_FP16: {
            uint16_t h;
            memcpy(&h, p + index * 2, 2);
            return fp16_to_float(h);
        }
        case NPU_DTYPE_FP32: {
            float f;
            memcpy(&f, p + index * 4, 4);
            return f;
        }
        case NPU_DTYPE_BF16: {
            uint16_t h;
            memcpy(&h, p + index * 2, 2);
            uint32_t bits = (uint32_t)h << 16;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        case NPU_DTYPE_INT8: {
            int8_t v;
            memcpy(&v, p + index * 1, 1);
            return (float)v;
        }
        case NPU_DTYPE_INT32: {
            int32_t v;
            memcpy(&v, p + index * 4, 4);
            return (float)v;
        }
        case NPU_DTYPE_INT16: {
            int16_t v;
            memcpy(&v, p + index * 2, 2);
            return (float)v;
        }
        default: return 0.0f;
    }
}

void npu_write_from_float(void* buf, int index, float value, npu_dtype_t dtype) {
    uint8_t* p = (uint8_t*)buf;
    switch (dtype) {
        case NPU_DTYPE_FP16: {
            uint16_t h = float_to_fp16(value);
            memcpy(p + index * 2, &h, 2);
            break;
        }
        case NPU_DTYPE_FP32:
            memcpy(p + index * 4, &value, 4);
            break;
        case NPU_DTYPE_BF16: {
            uint32_t bits;
            memcpy(&bits, &value, 4);
            uint16_t h = (uint16_t)(bits >> 16);
            memcpy(p + index * 2, &h, 2);
            break;
        }
        case NPU_DTYPE_INT8: {
            float clamped = value < -128.0f ? -128.0f : (value > 127.0f ? 127.0f : value);
            ((int8_t*)p)[index] = (int8_t)clamped;
            break;
        }
        case NPU_DTYPE_INT32: {
            int32_t v = (int32_t)value;
            memcpy(p + index * 4, &v, 4);
            break;
        }
        case NPU_DTYPE_INT16: {
            float clamped = value < -32768.0f ? -32768.0f : (value > 32767.0f ? 32767.0f : value);
            int16_t v = (int16_t)clamped;
            memcpy(p + index * 2, &v, 2);
            break;
        }
        default: break;
    }
}

float npu_round_to_dtype(float value, npu_dtype_t dtype) {
    switch (dtype) {
        case NPU_DTYPE_FP16:
            return fp16_to_float(float_to_fp16(value));
        case NPU_DTYPE_BF16: {
            uint32_t bits;
            memcpy(&bits, &value, 4);
            bits &= 0xFFFF0000u; /* truncate lower 16 mantissa bits */
            float r;
            memcpy(&r, &bits, 4);
            return r;
        }
        default:
            return value; /* FP32/INT types: no rounding needed */
    }
}
