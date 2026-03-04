#include "npu_api.h"
#include <string.h>

void npu_transpose(void* input, void* out, int ndim, const int* dims,
                   int dim0, int dim1, npu_dtype_t dtype) {
    /* compute total element count */
    int total = 1;
    for (int i = 0; i < ndim; i++) total *= dims[i];

    /* build output dims (swap dim0 and dim1) */
    int odims[16];
    for (int i = 0; i < ndim; i++) odims[i] = dims[i];
    int tmp = odims[dim0]; odims[dim0] = odims[dim1]; odims[dim1] = tmp;

    size_t elem_sz = npu_dtype_size(dtype);

    for (int idx = 0; idx < total; idx++) {
        /* flat index -> input coordinates */
        int coords[16];
        int rem = idx;
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = rem % dims[d];
            rem /= dims[d];
        }

        /* swap dim0 and dim1 */
        int t = coords[dim0]; coords[dim0] = coords[dim1]; coords[dim1] = t;

        /* coordinates -> output flat index */
        int oidx = 0;
        for (int d = 0; d < ndim; d++)
            oidx = oidx * odims[d] + coords[d];

        memcpy((char*)out + oidx * elem_sz,
               (const char*)input + idx * elem_sz, elem_sz);
    }
}

void npu_transpose_2d(void* input, void* out, int rows, int cols, npu_dtype_t dtype) {
    size_t elem_sz = npu_dtype_size(dtype);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            memcpy((char*)out + (c * rows + r) * elem_sz,
                   (const char*)input + (r * cols + c) * elem_sz, elem_sz);
        }
    }
}

void npu_reshape(void* input, void* out, int size, npu_dtype_t dtype) {
    (void)dtype;
    memcpy(out, input, (size_t)size);
}
