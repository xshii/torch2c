#include "npu_api.h"
#include <math.h>
#include <string.h>

void vector_layernorm(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out,
                      int hidden, int seq, float eps, npu_dtype_t compute_dtype) {
    vector_layernorm_part1(tid, input, gamma, beta, out, hidden, seq, eps, compute_dtype);
}

void vector_layernorm_part1(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out,
                            int hidden, int seq, float eps, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pi = npu_t_ptr(input);
    void* pg = npu_t_ptr(gamma);
    void* pb = npu_t_ptr(beta);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;
    for (int s = 0; s < seq; s++) {
        float mean = 0.0f;
        for (int h = 0; h < hidden; h++)
            mean += npu_read_compute(pi, s * hidden + h, dt, compute_dtype);
        mean /= (float)hidden;
        mean = npu_round_to_dtype(mean, compute_dtype);

        float var = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float d = npu_read_compute(pi, s * hidden + h, dt, compute_dtype) - mean;
            var += d * d;
        }
        var /= (float)hidden;
        var = npu_round_to_dtype(var, compute_dtype);

        float inv_std = 1.0f / sqrtf(var + eps);
        inv_std = npu_round_to_dtype(inv_std, compute_dtype);
        for (int h = 0; h < hidden; h++) {
            float x = npu_read_compute(pi, s * hidden + h, dt, compute_dtype);
            float normed = (x - mean) * inv_std;
            float g = npu_read_compute(pg, h, gamma.dtype, compute_dtype);
            float b = npu_read_compute(pb, h, beta.dtype, compute_dtype);
            npu_write_compute(po, s * hidden + h, g * normed + b, out.dtype, compute_dtype);
        }
    }
}

void vector_layernorm_part2(TidInfo tid, npu_tensor_t inter, npu_tensor_t orig, npu_tensor_t out,
                            int size, npu_dtype_t compute_dtype) {
    (void)tid;
    (void)orig;
    (void)compute_dtype;
    memcpy(npu_t_ptr(out), npu_t_ptr(inter), (size_t)size);
}
