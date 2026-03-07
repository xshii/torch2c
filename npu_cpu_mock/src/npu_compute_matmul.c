#include "npu_api.h"
#include "npu_debug.h"
#include <stdlib.h>
#include <assert.h>

#define MATMUL_STACK_ELEMS 4096

/* Core matmul: accumulate in float, return results via caller.
   Does NOT write to output tensor — caller decides how to use the float accumulator. */
static void matmul_core(const void* pa, npu_dtype_t a_dt,
                         const void* pb, npu_dtype_t b_dt,
                         float* acc_out,
                         int loop, int m, int n, int k,
                         npu_dtype_t compute_dtype) {
    int mat_a = m * k;
    int mat_b = k * n;
    int mat_o = m * n;
    for (int l = 0; l < loop; l++) {
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n; ni++) {
                float acc = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    float va = npu_read_compute(pa, l * mat_a + mi * k + ki, a_dt, compute_dtype);
                    float vb = npu_read_compute(pb, l * mat_b + ki * n + ni, b_dt, compute_dtype);
                    acc += va * vb;
                }
                acc_out[l * mat_o + mi * n + ni] = npu_round_to_dtype(acc, compute_dtype);
            }
        }
    }
}

static float* matmul_alloc(float* stack_buf, size_t total) {
    if (total <= MATMUL_STACK_ELEMS) return stack_buf;
    float* heap = (float*)malloc(total * sizeof(float));
    assert(heap != NULL);
    return heap;
}

static void matmul_free(float* buf, float* stack_buf) {
    if (buf != stack_buf) free(buf);
}

void cube_matmul(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                 int loop, int m, int n, int k, npu_dtype_t compute_dtype) {
    size_t total = (size_t)loop * (size_t)m * (size_t)n;
    npu_debug_tensor_arg_t dbg[] = {
        NPU_DBG_T(a, a, (int)(loop*m*k)), NPU_DBG_T(b, b, (int)(loop*k*n)),
        NPU_DBG_T(out, out, (int)total)
    };
    NPU_TRACE_BEGIN("cube_matmul", tid, dbg, 3);
    float acc_buf[MATMUL_STACK_ELEMS];
    float* acc = matmul_alloc(acc_buf, total);

    matmul_core(npu_t_ptr(a), a.dtype, npu_t_ptr(b), b.dtype,
                acc, loop, m, n, k, compute_dtype);

    void* po = npu_t_ptr(out);
    for (size_t i = 0; i < total; i++)
        npu_write_from_float(po, (int)i, acc[i], out.dtype);

    matmul_free(acc, acc_buf);
    NPU_TRACE_END("cube_matmul", tid, dbg, 3);
}

void cube_matmul_bias(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t bias, npu_tensor_t out,
                      int loop, int m, int n, int k, npu_dtype_t compute_dtype) {
    size_t total = (size_t)loop * (size_t)m * (size_t)n;
    npu_debug_tensor_arg_t dbg[] = {
        NPU_DBG_T(a, a, (int)(loop*m*k)), NPU_DBG_T(b, b, (int)(loop*k*n)),
        NPU_DBG_T(bias, bias, n), NPU_DBG_T(out, out, (int)total)
    };
    NPU_TRACE_BEGIN("cube_matmul_bias", tid, dbg, 4);
    float acc_buf[MATMUL_STACK_ELEMS];
    float* acc = matmul_alloc(acc_buf, total);

    matmul_core(npu_t_ptr(a), a.dtype, npu_t_ptr(b), b.dtype,
                acc, loop, m, n, k, compute_dtype);

    void* pbias = npu_t_ptr(bias);
    void* po = npu_t_ptr(out);
    int mat_o = m * n;
    for (int l = 0; l < loop; l++) {
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n; ni++) {
                int idx = l * mat_o + mi * n + ni;
                float b_val = npu_read_compute(pbias, ni, bias.dtype, compute_dtype);
                float result = acc[idx] + b_val;
                npu_write_from_float(po, idx, npu_round_to_dtype(result, compute_dtype), out.dtype);
            }
        }
    }

    matmul_free(acc, acc_buf);
    NPU_TRACE_END("cube_matmul_bias", tid, dbg, 4);
}
