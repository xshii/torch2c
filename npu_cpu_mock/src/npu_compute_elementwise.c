#include "npu_api.h"
#include <math.h>

void npu_add(void* a, void* b, void* out, int count, npu_dtype_t dtype) {
    for (int i = 0; i < count; i++) {
        float va = npu_read_as_float(a, i, dtype);
        float vb = npu_read_as_float(b, i, dtype);
        npu_write_from_float(out, i, va + vb, dtype);
    }
}

void npu_mul(void* a, void* b, void* out, int count, npu_dtype_t dtype) {
    for (int i = 0; i < count; i++) {
        float va = npu_read_as_float(a, i, dtype);
        float vb = npu_read_as_float(b, i, dtype);
        npu_write_from_float(out, i, va * vb, dtype);
    }
}

void npu_mul_scalar(void* input, void* out, float scalar, int count, npu_dtype_t dtype) {
    for (int i = 0; i < count; i++) {
        float v = npu_read_as_float(input, i, dtype);
        npu_write_from_float(out, i, v * scalar, dtype);
    }
}

void npu_gelu(void* input, void* out, int count, npu_dtype_t dtype) {
    const float inv_sqrt2 = (float)(1.0 / sqrt(2.0));
    for (int i = 0; i < count; i++) {
        float x = npu_read_as_float(input, i, dtype);
        float y = x * 0.5f * (1.0f + erff(x * inv_sqrt2));
        npu_write_from_float(out, i, y, dtype);
    }
}
