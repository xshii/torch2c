#include "npu_train_api.h"
#include <math.h>

void vector_sgd_step(TidInfo tid, npu_tensor_t param, npu_tensor_t grad,
                     float lr, float momentum, float weight_decay,
                     npu_tensor_t velocity, int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pp = npu_t_ptr(param);
    void* pg = npu_t_ptr(grad);
    void* pv = npu_t_ptr(velocity);

    for (int i = 0; i < count; i++) {
        float p = npu_read_compute(pp, i, param.dtype, compute_dtype);
        float g = npu_read_compute(pg, i, grad.dtype, compute_dtype);

        /* weight decay */
        if (weight_decay != 0.0f)
            g += weight_decay * p;

        /* momentum */
        if (momentum != 0.0f) {
            float v = npu_read_compute(pv, i, velocity.dtype, compute_dtype);
            v = momentum * v + g;
            npu_write_compute(pv, i, v, velocity.dtype, compute_dtype);
            g = v;
        }

        /* update */
        npu_write_compute(pp, i, p - lr * g, param.dtype, compute_dtype);
    }
}

void vector_adam_step(TidInfo tid, npu_tensor_t param, npu_tensor_t grad,
                      npu_tensor_t m, npu_tensor_t v,
                      float lr, float beta1, float beta2, float eps,
                      int step, int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pp = npu_t_ptr(param);
    void* pg = npu_t_ptr(grad);
    void* pm = npu_t_ptr(m);
    void* pv = npu_t_ptr(v);

    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);

    for (int i = 0; i < count; i++) {
        float p = npu_read_compute(pp, i, param.dtype, compute_dtype);
        float g = npu_read_compute(pg, i, grad.dtype, compute_dtype);
        float mi = npu_read_compute(pm, i, m.dtype, compute_dtype);
        float vi = npu_read_compute(pv, i, v.dtype, compute_dtype);

        /* update moments */
        mi = beta1 * mi + (1.0f - beta1) * g;
        vi = beta2 * vi + (1.0f - beta2) * g * g;
        npu_write_compute(pm, i, mi, m.dtype, compute_dtype);
        npu_write_compute(pv, i, vi, v.dtype, compute_dtype);

        /* bias-corrected */
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;

        /* update param */
        npu_write_compute(pp, i, p - lr * m_hat / (sqrtf(v_hat) + eps),
                          param.dtype, compute_dtype);
    }
}

void vector_grad_clip(TidInfo tid, npu_tensor_t grad, float max_norm,
                      int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pg = npu_t_ptr(grad);

    /* compute L2 norm */
    float sum_sq = 0.0f;
    for (int i = 0; i < count; i++) {
        float g = npu_read_compute(pg, i, grad.dtype, compute_dtype);
        sum_sq += g * g;
    }
    float norm = sqrtf(sum_sq);

    /* clip if exceeds */
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < count; i++) {
            float g = npu_read_compute(pg, i, grad.dtype, compute_dtype);
            npu_write_compute(pg, i, g * scale, grad.dtype, compute_dtype);
        }
    }
}
