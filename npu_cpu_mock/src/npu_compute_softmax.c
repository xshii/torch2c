#include "npu_api.h"
#include <math.h>
#include <string.h>
#include <assert.h>

void vector_softmax(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                    int dim, int count, npu_dtype_t compute_dtype) {
    vector_softmax_part1(tid, input, out, dim, count, compute_dtype);
}

void vector_softmax_part1(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                          int dim, int count, npu_dtype_t compute_dtype) {
    (void)tid;
    assert(dim > 0);
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    int rows = count / dim;
    for (int r = 0; r < rows; r++) {
        int base = r * dim;

        float mx = npu_read_compute(pi, base, dt, compute_dtype);
        for (int c = 1; c < dim; c++) {
            float v = npu_read_compute(pi, base + c, dt, compute_dtype);
            if (v > mx) mx = v;
        }

        float sum = 0.0f;
        for (int c = 0; c < dim; c++) {
            float v = npu_read_compute(pi, base + c, dt, compute_dtype);
            float e = expf(v - mx);
            e = npu_round_to_dtype(e, compute_dtype);
            npu_write_from_float(po, base + c, e, out.dtype);
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        inv_sum = npu_round_to_dtype(inv_sum, compute_dtype);
        for (int c = 0; c < dim; c++) {
            float e = npu_read_as_float(po, base + c, out.dtype);
            npu_write_compute(po, base + c, e * inv_sum, out.dtype, compute_dtype);
        }
    }
}

void vector_softmax_part2(TidInfo tid, npu_tensor_t inter, npu_tensor_t out,
                          int size, npu_dtype_t compute_dtype) {
    (void)tid;
    (void)compute_dtype;
    memcpy(npu_t_ptr(out), npu_t_ptr(inter), (size_t)size);
}
