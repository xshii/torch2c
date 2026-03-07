#include "npu_api.h"
#include "npu_debug.h"
#include <math.h>
#include <string.h>

void vector_rmsnorm(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t out,
                    int hidden, int seq, float eps, npu_dtype_t compute_dtype) {
    vector_rmsnorm_part1(tid, input, gamma, out, hidden, seq, eps, compute_dtype);
}

void vector_rmsnorm_part1(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t out,
                          int hidden, int seq, float eps, npu_dtype_t compute_dtype) {
    int count = seq * hidden;
    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(input, input, count), NPU_DBG_T(gamma, gamma, hidden),
        NPU_DBG_T(out, out, count)
    };
    NPU_TRACE_BEGIN("vector_rmsnorm_part1", tid, _dbg, 3);

    void* pi = npu_t_ptr(input);
    void* pg = npu_t_ptr(gamma);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    for (int s = 0; s < seq; s++) {
        float sum_sq = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float x = npu_read_compute(pi, s * hidden + h, dt, compute_dtype);
            sum_sq += x * x;
        }
        float rms = sum_sq / (float)hidden;
        rms = npu_round_to_dtype(rms, compute_dtype);
        float inv_rms = 1.0f / sqrtf(rms + eps);
        inv_rms = npu_round_to_dtype(inv_rms, compute_dtype);
        for (int h = 0; h < hidden; h++) {
            float x = npu_read_compute(pi, s * hidden + h, dt, compute_dtype);
            float g = npu_read_compute(pg, h, gamma.dtype, compute_dtype);
            npu_write_compute(po, s * hidden + h, x * inv_rms * g, out.dtype, compute_dtype);
        }
    }

    NPU_TRACE_END("vector_rmsnorm_part1", tid, _dbg, 3);
}

void vector_rmsnorm_part2(TidInfo tid, npu_tensor_t inter, npu_tensor_t orig, npu_tensor_t out,
                          int size, npu_dtype_t compute_dtype) {
    (void)orig;
    (void)compute_dtype;
    int elem_size = (int)npu_dtype_size(inter.dtype);
    int count = (elem_size > 0) ? size / elem_size : 0;
    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(inter, inter, count), NPU_DBG_T(out, out, count)
    };
    NPU_TRACE_BEGIN("vector_rmsnorm_part2", tid, _dbg, 2);
    memcpy(npu_t_ptr(out), npu_t_ptr(inter), (size_t)size);
    NPU_TRACE_END("vector_rmsnorm_part2", tid, _dbg, 2);
}
