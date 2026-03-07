#include "npu_api.h"
#include <math.h>

void vector_add(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va + vb, out.dtype, compute_dtype);
    }
}

void vector_mul(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va * vb, out.dtype, compute_dtype);
    }
}

void vector_mul_scalar(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                       float scalar, int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float v = npu_read_compute(pi, i, input.dtype, compute_dtype);
        npu_write_compute(po, i, v * scalar, out.dtype, compute_dtype);
    }
}

void vector_sub(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va - vb, out.dtype, compute_dtype);
    }
}

void vector_div(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va / vb, out.dtype, compute_dtype);
    }
}

void vector_fill(TidInfo tid, npu_tensor_t out, float value,
                 int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* po = npu_t_ptr(out);
    float rounded = npu_round_to_dtype(value, compute_dtype);
    for (int i = 0; i < count; i++)
        npu_write_from_float(po, i, rounded, out.dtype);
}

void vector_dropout(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                    npu_tensor_t mask, int count, float scale,
                    npu_dtype_t compute_dtype) {
    (void)tid;
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    void* pm = npu_t_ptr(mask);
    for (int i = 0; i < count; i++) {
        float m = npu_read_as_float(pm, i, mask.dtype);
        float v = npu_read_compute(pi, i, input.dtype, compute_dtype);
        npu_write_compute(po, i, (m != 0.0f) ? v * scale : 0.0f, out.dtype, compute_dtype);
    }
}

void vector_gelu(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                 int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    const float inv_sqrt2 = (float)(1.0 / sqrt(2.0));
    for (int i = 0; i < count; i++) {
        float x = npu_read_compute(pi, i, input.dtype, compute_dtype);
        float y = x * 0.5f * (1.0f + erff(x * inv_sqrt2));
        npu_write_compute(po, i, y, out.dtype, compute_dtype);
    }
}
