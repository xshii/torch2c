#include "npu_api.h"
#include <math.h>
#include <string.h>

void vector_softmax(npu_tensor_t input, npu_tensor_t out,
                    int dim, int count, npu_dtype_t compute_dtype) {
    vector_softmax_part1(input, out, dim, count, compute_dtype);
}

void vector_softmax_part1(npu_tensor_t input, npu_tensor_t out,
                          int dim, int count, npu_dtype_t compute_dtype) {
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    (void)compute_dtype;
    int rows = count / dim;
    for (int r = 0; r < rows; r++) {
        int base = r * dim;

        float mx = npu_read_as_float(pi, base, dt);
        for (int c = 1; c < dim; c++) {
            float v = npu_read_as_float(pi, base + c, dt);
            if (v > mx) mx = v;
        }

        float sum = 0.0f;
        for (int c = 0; c < dim; c++) {
            float v = npu_read_as_float(pi, base + c, dt);
            float e = expf(v - mx);
            npu_write_from_float(po, base + c, e, out.dtype);
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (int c = 0; c < dim; c++) {
            float e = npu_read_as_float(po, base + c, out.dtype);
            npu_write_from_float(po, base + c, e * inv_sum, out.dtype);
        }
    }
}

void vector_softmax_part2(npu_tensor_t inter, npu_tensor_t out,
                          int size, npu_dtype_t compute_dtype) {
    (void)compute_dtype;
    memcpy(npu_t_ptr(out), npu_t_ptr(inter), (size_t)size);
}
