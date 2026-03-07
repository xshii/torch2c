#include "npu_api.h"
#include "npu_debug.h"
#include <math.h>

/* 二元算子通用 trace 宏 */
#define BINOP_TRACE_BEGIN(name, tid, a, b, out, count) \
    npu_debug_tensor_arg_t _dbg[] = { \
        NPU_DBG_T(a, a, count), NPU_DBG_T(b, b, count), \
        NPU_DBG_T(out, out, count) \
    }; \
    NPU_TRACE_BEGIN(name, tid, _dbg, 3)

#define BINOP_TRACE_END(name, tid) \
    NPU_TRACE_END(name, tid, _dbg, 3)

/* 一元算子通用 trace 宏 */
#define UNOP_TRACE_BEGIN(name, tid, in, out, count) \
    npu_debug_tensor_arg_t _dbg[] = { \
        NPU_DBG_T(in, in, count), NPU_DBG_T(out, out, count) \
    }; \
    NPU_TRACE_BEGIN(name, tid, _dbg, 2)

#define UNOP_TRACE_END(name, tid) \
    NPU_TRACE_END(name, tid, _dbg, 2)

void vector_add(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    BINOP_TRACE_BEGIN("vector_add", tid, a, b, out, count);
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va + vb, out.dtype, compute_dtype);
    }
    BINOP_TRACE_END("vector_add", tid);
}

void vector_mul(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    BINOP_TRACE_BEGIN("vector_mul", tid, a, b, out, count);
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va * vb, out.dtype, compute_dtype);
    }
    BINOP_TRACE_END("vector_mul", tid);
}

void vector_mul_scalar(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                       float scalar, int count, npu_dtype_t compute_dtype) {
    UNOP_TRACE_BEGIN("vector_mul_scalar", tid, input, out, count);
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float v = npu_read_compute(pi, i, input.dtype, compute_dtype);
        npu_write_compute(po, i, v * scalar, out.dtype, compute_dtype);
    }
    UNOP_TRACE_END("vector_mul_scalar", tid);
}

void vector_sub(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    BINOP_TRACE_BEGIN("vector_sub", tid, a, b, out, count);
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va - vb, out.dtype, compute_dtype);
    }
    BINOP_TRACE_END("vector_sub", tid);
}

void vector_div(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype) {
    BINOP_TRACE_BEGIN("vector_div", tid, a, b, out, count);
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    for (int i = 0; i < count; i++) {
        float va = npu_read_compute(pa, i, a.dtype, compute_dtype);
        float vb = npu_read_compute(pb, i, b.dtype, compute_dtype);
        npu_write_compute(po, i, va / vb, out.dtype, compute_dtype);
    }
    BINOP_TRACE_END("vector_div", tid);
}

void vector_fill(TidInfo tid, npu_tensor_t out, float value,
                 int count, npu_dtype_t compute_dtype) {
    npu_debug_tensor_arg_t _dbg[] = { NPU_DBG_T(out, out, count) };
    NPU_TRACE_BEGIN("vector_fill", tid, _dbg, 1);
    void* po = npu_t_ptr(out);
    float rounded = npu_round_to_dtype(value, compute_dtype);
    for (int i = 0; i < count; i++)
        npu_write_from_float(po, i, rounded, out.dtype);
    NPU_TRACE_END("vector_fill", tid, _dbg, 1);
}

void vector_dropout(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                    npu_tensor_t mask, int count, float scale,
                    npu_dtype_t compute_dtype) {
    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(input, input, count), NPU_DBG_T(mask, mask, count),
        NPU_DBG_T(out, out, count)
    };
    NPU_TRACE_BEGIN("vector_dropout", tid, _dbg, 3);
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    void* pm = npu_t_ptr(mask);
    for (int i = 0; i < count; i++) {
        float m = npu_read_as_float(pm, i, mask.dtype);
        float v = npu_read_compute(pi, i, input.dtype, compute_dtype);
        npu_write_compute(po, i, (m != 0.0f) ? v * scale : 0.0f, out.dtype, compute_dtype);
    }
    NPU_TRACE_END("vector_dropout", tid, _dbg, 3);
}

void vector_gelu(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                 int count, npu_dtype_t compute_dtype) {
    UNOP_TRACE_BEGIN("vector_gelu", tid, input, out, count);
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    const float inv_sqrt2 = (float)(1.0 / sqrt(2.0));
    for (int i = 0; i < count; i++) {
        float x = npu_read_compute(pi, i, input.dtype, compute_dtype);
        float y = x * 0.5f * (1.0f + erff(x * inv_sqrt2));
        npu_write_compute(po, i, y, out.dtype, compute_dtype);
    }
    UNOP_TRACE_END("vector_gelu", tid);
}
