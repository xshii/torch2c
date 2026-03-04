#include "npu_api.h"
#include <math.h>
#include <string.h>

void npu_layernorm_part1(void* input, void* gamma, void* beta, void* out,
                         int hidden, int seq, float eps, npu_dtype_t dtype) {
    for (int s = 0; s < seq; s++) {
        /* mean */
        float mean = 0.0f;
        for (int h = 0; h < hidden; h++)
            mean += npu_read_as_float(input, s * hidden + h, dtype);
        mean /= (float)hidden;

        /* variance */
        float var = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float d = npu_read_as_float(input, s * hidden + h, dtype) - mean;
            var += d * d;
        }
        var /= (float)hidden;

        /* normalize -> gamma*x + beta */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int h = 0; h < hidden; h++) {
            float x = npu_read_as_float(input, s * hidden + h, dtype);
            float normed = (x - mean) * inv_std;
            float g = npu_read_as_float(gamma, h, dtype);
            float b = npu_read_as_float(beta,  h, dtype);
            npu_write_from_float(out, s * hidden + h, g * normed + b, dtype);
        }
    }
}

void npu_layernorm_part2(void* inter, void* orig, void* out, int size, npu_dtype_t dtype) {
    (void)orig;
    (void)dtype;
    memcpy(out, inter, (size_t)size);
}
