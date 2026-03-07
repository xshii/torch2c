#include "npu_train_api.h"
#include <math.h>

/* gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi) */
void vector_gelu_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t input,
                          npu_tensor_t grad_in, int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pg = npu_t_ptr(grad_out);
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(grad_in);
    const float inv_sqrt2 = (float)(1.0 / sqrt(2.0));
    const float inv_sqrt2pi = (float)(1.0 / sqrt(2.0 * 3.14159265358979323846));

    for (int i = 0; i < count; i++) {
        float x = npu_read_compute(pi, i, input.dtype, compute_dtype);
        float g = npu_read_compute(pg, i, grad_out.dtype, compute_dtype);
        float cdf = 0.5f * (1.0f + erff(x * inv_sqrt2));
        float pdf = inv_sqrt2pi * expf(-0.5f * x * x);
        npu_write_compute(po, i, g * (cdf + x * pdf), grad_in.dtype, compute_dtype);
    }
}

/* softmax backward: grad_in[i] = out[i] * (grad_out[i] - sum(grad_out * out)) */
void vector_softmax_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t output,
                             npu_tensor_t grad_in, int dim, int count,
                             npu_dtype_t compute_dtype) {
    (void)tid;
    void* pg = npu_t_ptr(grad_out);
    void* py = npu_t_ptr(output);
    void* po = npu_t_ptr(grad_in);
    int rows = count / dim;

    for (int r = 0; r < rows; r++) {
        int base = r * dim;
        float dot = 0.0f;
        for (int c = 0; c < dim; c++) {
            float g = npu_read_compute(pg, base + c, grad_out.dtype, compute_dtype);
            float y = npu_read_compute(py, base + c, output.dtype, compute_dtype);
            dot += g * y;
        }
        dot = npu_round_to_dtype(dot, compute_dtype);
        for (int c = 0; c < dim; c++) {
            float g = npu_read_compute(pg, base + c, grad_out.dtype, compute_dtype);
            float y = npu_read_compute(py, base + c, output.dtype, compute_dtype);
            npu_write_compute(po, base + c, y * (g - dot), grad_in.dtype, compute_dtype);
        }
    }
}

void vector_layernorm_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t input,
                               npu_tensor_t gamma, npu_tensor_t grad_in,
                               npu_tensor_t grad_gamma, npu_tensor_t grad_beta,
                               int hidden, int seq, float eps,
                               npu_dtype_t compute_dtype) {
    (void)tid;
    void* pdy = npu_t_ptr(grad_out);
    void* px  = npu_t_ptr(input);
    void* pg  = npu_t_ptr(gamma);
    void* pdx = npu_t_ptr(grad_in);
    void* pdg = npu_t_ptr(grad_gamma);
    void* pdb = npu_t_ptr(grad_beta);
    npu_dtype_t x_dt = input.dtype;
    npu_dtype_t dy_dt = grad_out.dtype;
    npu_dtype_t g_dt = gamma.dtype;

    /* zero grad_gamma, grad_beta */
    for (int h = 0; h < hidden; h++) {
        npu_write_from_float(pdg, h, 0.0f, grad_gamma.dtype);
        npu_write_from_float(pdb, h, 0.0f, grad_beta.dtype);
    }

    for (int s = 0; s < seq; s++) {
        int base = s * hidden;
        /* forward stats */
        float mean = 0.0f;
        for (int h = 0; h < hidden; h++)
            mean += npu_read_compute(px, base + h, x_dt, compute_dtype);
        mean /= (float)hidden;

        float var = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float d = npu_read_compute(px, base + h, x_dt, compute_dtype) - mean;
            var += d * d;
        }
        var /= (float)hidden;
        float inv_std = 1.0f / sqrtf(var + eps);

        /* accumulate grad_gamma, grad_beta */
        float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float dy = npu_read_compute(pdy, base + h, dy_dt, compute_dtype);
            float x_hat = (npu_read_compute(px, base + h, x_dt, compute_dtype) - mean) * inv_std;
            float gv = npu_read_compute(pg, h, g_dt, compute_dtype);
            float prev_dg = npu_read_as_float(pdg, h, grad_gamma.dtype);
            npu_write_from_float(pdg, h, prev_dg + dy * x_hat, grad_gamma.dtype);
            float prev_db = npu_read_as_float(pdb, h, grad_beta.dtype);
            npu_write_from_float(pdb, h, prev_db + dy, grad_beta.dtype);
            sum_dy += dy * gv;
            sum_dy_xhat += dy * gv * x_hat;
        }

        /* grad_in */
        for (int h = 0; h < hidden; h++) {
            float dy = npu_read_compute(pdy, base + h, dy_dt, compute_dtype);
            float x_hat = (npu_read_compute(px, base + h, x_dt, compute_dtype) - mean) * inv_std;
            float gv = npu_read_compute(pg, h, g_dt, compute_dtype);
            float dx = inv_std / (float)hidden *
                       ((float)hidden * dy * gv - sum_dy - x_hat * sum_dy_xhat);
            npu_write_compute(pdx, base + h, dx, grad_in.dtype, compute_dtype);
        }
    }
}

void vector_rmsnorm_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t input,
                             npu_tensor_t gamma, npu_tensor_t grad_in,
                             npu_tensor_t grad_gamma,
                             int hidden, int seq, float eps,
                             npu_dtype_t compute_dtype) {
    (void)tid;
    void* pdy = npu_t_ptr(grad_out);
    void* px  = npu_t_ptr(input);
    void* pg  = npu_t_ptr(gamma);
    void* pdx = npu_t_ptr(grad_in);
    void* pdg = npu_t_ptr(grad_gamma);
    npu_dtype_t x_dt = input.dtype;
    npu_dtype_t dy_dt = grad_out.dtype;
    npu_dtype_t g_dt = gamma.dtype;

    /* zero grad_gamma */
    for (int h = 0; h < hidden; h++)
        npu_write_from_float(pdg, h, 0.0f, grad_gamma.dtype);

    for (int s = 0; s < seq; s++) {
        int base = s * hidden;
        float sum_sq = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float x = npu_read_compute(px, base + h, x_dt, compute_dtype);
            sum_sq += x * x;
        }
        float rms = sum_sq / (float)hidden;
        float inv_rms = 1.0f / sqrtf(rms + eps);

        /* accumulate grad_gamma, compute intermediate */
        float sum_dy_x = 0.0f;
        for (int h = 0; h < hidden; h++) {
            float dy = npu_read_compute(pdy, base + h, dy_dt, compute_dtype);
            float x = npu_read_compute(px, base + h, x_dt, compute_dtype);
            float gv = npu_read_compute(pg, h, g_dt, compute_dtype);
            float prev_dg = npu_read_as_float(pdg, h, grad_gamma.dtype);
            npu_write_from_float(pdg, h, prev_dg + dy * x * inv_rms, grad_gamma.dtype);
            sum_dy_x += dy * gv * x;
        }

        /* grad_in: dx = inv_rms * (dy*g - x * sum(dy*g*x) / (hidden * (rms+eps))) */
        for (int h = 0; h < hidden; h++) {
            float dy = npu_read_compute(pdy, base + h, dy_dt, compute_dtype);
            float x = npu_read_compute(px, base + h, x_dt, compute_dtype);
            float gv = npu_read_compute(pg, h, g_dt, compute_dtype);
            float dx = inv_rms * (dy * gv - x * sum_dy_x * inv_rms * inv_rms / (float)hidden);
            npu_write_compute(pdx, base + h, dx, grad_in.dtype, compute_dtype);
        }
    }
}