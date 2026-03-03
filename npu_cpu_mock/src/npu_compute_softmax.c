#include "npu_api.h"
#include <math.h>
#include <string.h>

void npu_softmax_part1(void* input, void* out, int dim, int count, npu_dtype_t dtype) {
    int rows = count / dim;
    for (int r = 0; r < rows; r++) {
        int base = r * dim;

        /* row max */
        float mx = npu_read_as_float(input, base, dtype);
        for (int c = 1; c < dim; c++) {
            float v = npu_read_as_float(input, base + c, dtype);
            if (v > mx) mx = v;
        }

        /* exp(x - max) and sum */
        float sum = 0.0f;
        for (int c = 0; c < dim; c++) {
            float v = npu_read_as_float(input, base + c, dtype);
            float e = expf(v - mx);
            npu_write_from_float(out, base + c, e, dtype);
            sum += e;
        }

        /* normalize */
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < dim; c++) {
            float e = npu_read_as_float(out, base + c, dtype);
            npu_write_from_float(out, base + c, e * inv_sum, dtype);
        }
    }
}

void npu_softmax_part2(void* inter, void* out, int count, npu_dtype_t dtype) {
    (void)dtype;
    memcpy(out, inter, (size_t)count);
}
