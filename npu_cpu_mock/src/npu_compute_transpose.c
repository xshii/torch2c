#include "npu_api.h"
#include "npu_debug.h"
#include <string.h>
#include <assert.h>

void vector_transpose(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                      int ndim, const int* dims, int dim0, int dim1, npu_dtype_t compute_dtype) {
    assert(ndim > 0 && ndim <= NPU_MAX_NDIM);
    assert(dim0 >= 0 && dim0 < ndim);
    assert(dim1 >= 0 && dim1 < ndim);

    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    int total = 1;
    for (int i = 0; i < ndim; i++) total *= dims[i];

    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(input, input, total), NPU_DBG_T(out, out, total)
    };
    NPU_TRACE_BEGIN("vector_transpose", tid, _dbg, 2);

    int odims[NPU_MAX_NDIM];
    for (int i = 0; i < ndim; i++) odims[i] = dims[i];
    int tmp = odims[dim0]; odims[dim0] = odims[dim1]; odims[dim1] = tmp;

    int same_dtype = (dt == out.dtype);
    size_t elem_sz = same_dtype ? npu_dtype_size(dt) : 0;

    for (int idx = 0; idx < total; idx++) {
        int coords[NPU_MAX_NDIM];
        int rem = idx;
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = rem % dims[d];
            rem /= dims[d];
        }
        int t2 = coords[dim0]; coords[dim0] = coords[dim1]; coords[dim1] = t2;
        int oidx = 0;
        for (int d = 0; d < ndim; d++)
            oidx = oidx * odims[d] + coords[d];

        if (same_dtype) {
            memcpy((char*)po + oidx * elem_sz,
                   (const char*)pi + idx * elem_sz, elem_sz);
        } else {
            float v = npu_read_compute(pi, idx, dt, compute_dtype);
            npu_write_compute(po, oidx, v, out.dtype, compute_dtype);
        }
    }

    NPU_TRACE_END("vector_transpose", tid, _dbg, 2);
}

void vector_transpose_2d(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                         int rows, int cols, npu_dtype_t compute_dtype) {
    int total = rows * cols;
    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(input, input, total), NPU_DBG_T(out, out, total)
    };
    NPU_TRACE_BEGIN("vector_transpose_2d", tid, _dbg, 2);

    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    int same_dtype = (dt == out.dtype);
    size_t elem_sz = same_dtype ? npu_dtype_size(dt) : 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (same_dtype) {
                memcpy((char*)po + (c * rows + r) * elem_sz,
                       (const char*)pi + (r * cols + c) * elem_sz, elem_sz);
            } else {
                float v = npu_read_compute(pi, r * cols + c, dt, compute_dtype);
                npu_write_compute(po, c * rows + r, v, out.dtype, compute_dtype);
            }
        }
    }

    NPU_TRACE_END("vector_transpose_2d", tid, _dbg, 2);
}
