#include "npu_api.h"
#include <math.h>

void vector_add(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    (void)compute_dtype;
    for (int i = 0; i < count; i++) {
        float va = npu_read_as_float(pa, i, a.dtype);
        float vb = npu_read_as_float(pb, i, b.dtype);
        npu_write_from_float(po, i, va + vb, out.dtype);
    }
}

void vector_mul(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    (void)compute_dtype;
    for (int i = 0; i < count; i++) {
        float va = npu_read_as_float(pa, i, a.dtype);
        float vb = npu_read_as_float(pb, i, b.dtype);
        npu_write_from_float(po, i, va * vb, out.dtype);
    }
}

void vector_mul_scalar(npu_tensor_t input, npu_tensor_t out,
                       float scalar, int count, npu_dtype_t compute_dtype) {
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    (void)compute_dtype;
    for (int i = 0; i < count; i++) {
        float v = npu_read_as_float(pi, i, input.dtype);
        npu_write_from_float(po, i, v * scalar, out.dtype);
    }
}

void vector_gelu(npu_tensor_t input, npu_tensor_t out,
                 int count, npu_dtype_t compute_dtype) {
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    (void)compute_dtype;
    const float inv_sqrt2 = (float)(1.0 / sqrt(2.0));
    for (int i = 0; i < count; i++) {
        float x = npu_read_as_float(pi, i, input.dtype);
        float y = x * 0.5f * (1.0f + erff(x * inv_sqrt2));
        npu_write_from_float(po, i, y, out.dtype);
    }
}
