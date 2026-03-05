#include "npu_api.h"
#include <math.h>
#include <string.h>

void vector_layernorm(npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out,
                      int hidden, int seq, float eps, npu_dtype_t compute_dtype) {
    vector_layernorm_part1(input, gamma, beta, out, hidden, seq, eps, compute_dtype);
}

void vector_layernorm_part1(npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out,
                            int hidden, int seq, float eps, npu_dtype_t compute_dtype) {
    void* pi = npu_t_ptr(input);
    void* pg = npu_t_ptr(gamma);
    void* pb = npu_t_ptr(beta);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    (void)compute_dtype;
    for (int s = 0; s < seq; s++) {
        float mean = 0.0f;
        for (int h = 0; h < hidden; h++)
            mean += npu_read_as_float(pi, s * hidden + h, dt);
        mean /= (float)hidden;

        float var = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float d = npu_read_as_float(pi, s * hidden + h, dt) - mean;
            var += d * d;
        }
        var /= (float)hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int h = 0; h < hidden; h++) {
            float x = npu_read_as_float(pi, s * hidden + h, dt);
            float normed = (x - mean) * inv_std;
            float g = npu_read_as_float(pg, h, gamma.dtype);
            float b = npu_read_as_float(pb, h, beta.dtype);
            npu_write_from_float(po, s * hidden + h, g * normed + b, out.dtype);
        }
    }
}

void vector_layernorm_part2(npu_tensor_t inter, npu_tensor_t orig, npu_tensor_t out,
                            int size, npu_dtype_t compute_dtype) {
    (void)orig;
    (void)compute_dtype;
    memcpy(npu_t_ptr(out), npu_t_ptr(inter), (size_t)size);
}
