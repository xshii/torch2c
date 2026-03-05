#include "npu_api.h"
#include <string.h>

void vector_transpose(npu_tensor_t input, npu_tensor_t out,
                      int ndim, const int* dims, int dim0, int dim1, npu_dtype_t compute_dtype) {
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    (void)compute_dtype;
    int total = 1;
    for (int i = 0; i < ndim; i++) total *= dims[i];

    int odims[16];
    for (int i = 0; i < ndim; i++) odims[i] = dims[i];
    int tmp = odims[dim0]; odims[dim0] = odims[dim1]; odims[dim1] = tmp;

    size_t elem_sz = npu_dtype_size(dt);

    for (int idx = 0; idx < total; idx++) {
        int coords[16];
        int rem = idx;
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = rem % dims[d];
            rem /= dims[d];
        }

        int t2 = coords[dim0]; coords[dim0] = coords[dim1]; coords[dim1] = t2;

        int oidx = 0;
        for (int d = 0; d < ndim; d++)
            oidx = oidx * odims[d] + coords[d];

        memcpy((char*)po + oidx * elem_sz,
               (const char*)pi + idx * elem_sz, elem_sz);
    }
}

void vector_transpose_2d(npu_tensor_t input, npu_tensor_t out,
                         int rows, int cols, npu_dtype_t compute_dtype) {
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    (void)compute_dtype;
    size_t elem_sz = npu_dtype_size(dt);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            memcpy((char*)po + (c * rows + r) * elem_sz,
                   (const char*)pi + (r * cols + c) * elem_sz, elem_sz);
        }
    }
}

void scalar_reshape(npu_tensor_t input, npu_tensor_t out,
                    int size, npu_dtype_t compute_dtype) {
    (void)compute_dtype;
    memcpy(npu_t_ptr(out), npu_t_ptr(input), (size_t)size);
}

void scalar_broadcast(npu_tensor_t input, npu_tensor_t out,
                      int size, npu_dtype_t compute_dtype) {
    (void)compute_dtype;
    memcpy(npu_t_ptr(out), npu_t_ptr(input), (size_t)size);
}

void scalar_copy(npu_tensor_t input, npu_tensor_t out,
                 int size, npu_dtype_t compute_dtype) {
    (void)compute_dtype;
    memcpy(npu_t_ptr(out), npu_t_ptr(input), (size_t)size);
}
