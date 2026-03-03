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
    memcpy(out, inter, (size_t)count * npu_dtype_size(dtype));
}

/*
 * Softmax with bitmap mask (attention mask).
 *   mask: bit-packed, LSB-first.  bit i = mask[i/8] & (1 << (i%8))
 *         bit=1 => keep,  bit=0 => masked out (-inf before softmax => 0 after)
 *
 * Row-wise: for each row of length `dim`, apply mask then softmax.
 * If an entire row is masked, output is all zeros.
 */
void npu_softmax_mask(void* input, void* out, const uint8_t* mask,
                      int dim, int count, npu_dtype_t dtype) {
    int rows = count / dim;
    for (int r = 0; r < rows; r++) {
        int base = r * dim;

        /* row max among unmasked positions */
        float mx = -1e30f;
        int has_unmasked = 0;
        for (int c = 0; c < dim; c++) {
            int idx = base + c;
            if (mask[idx / 8] & (1 << (idx % 8))) {
                float v = npu_read_as_float(input, idx, dtype);
                if (!has_unmasked || v > mx) mx = v;
                has_unmasked = 1;
            }
        }

        if (!has_unmasked) {
            /* entire row masked => all zeros */
            for (int c = 0; c < dim; c++)
                npu_write_from_float(out, base + c, 0.0f, dtype);
            continue;
        }

        /* exp(x - max) for unmasked, 0 for masked */
        float sum = 0.0f;
        for (int c = 0; c < dim; c++) {
            int idx = base + c;
            if (mask[idx / 8] & (1 << (idx % 8))) {
                float v = npu_read_as_float(input, idx, dtype);
                float e = expf(v - mx);
                npu_write_from_float(out, idx, e, dtype);
                sum += e;
            } else {
                npu_write_from_float(out, idx, 0.0f, dtype);
            }
        }

        /* normalize unmasked */
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < dim; c++) {
            int idx = base + c;
            if (mask[idx / 8] & (1 << (idx % 8))) {
                float e = npu_read_as_float(out, idx, dtype);
                npu_write_from_float(out, idx, e * inv_sum, dtype);
            }
        }
    }
}
