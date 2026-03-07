#include "npu_train_api.h"
#include <math.h>

void vector_cross_entropy_loss(TidInfo tid, npu_tensor_t logits, npu_tensor_t labels,
                               npu_tensor_t out, int batch, int num_classes,
                               npu_dtype_t compute_dtype) {
    (void)tid;
    void* pl = npu_t_ptr(logits);
    void* plb = npu_t_ptr(labels);
    void* po = npu_t_ptr(out);
    npu_dtype_t l_dt = logits.dtype;
    npu_dtype_t lb_dt = labels.dtype;

    for (int b = 0; b < batch; b++) {
        int base = b * num_classes;
        /* row-wise log-softmax: max → exp(x-max) → sum → log(sum) */
        float max_val = -1e30f;
        for (int c = 0; c < num_classes; c++) {
            float v = npu_read_compute(pl, base + c, l_dt, compute_dtype);
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float v = npu_read_compute(pl, base + c, l_dt, compute_dtype);
            sum_exp += expf(v - max_val);
        }
        float log_sum = logf(sum_exp) + max_val;

        /* label is class index stored as float */
        int target = (int)npu_read_as_float(plb, b, lb_dt);
        float loss = -(npu_read_compute(pl, base + target, l_dt, compute_dtype) - log_sum);
        npu_write_compute(po, b, loss, out.dtype, compute_dtype);
    }
}

void vector_mse_loss(TidInfo tid, npu_tensor_t pred, npu_tensor_t target,
                     npu_tensor_t out, int count, npu_dtype_t compute_dtype) {
    (void)tid;
    void* pp = npu_t_ptr(pred);
    void* pt = npu_t_ptr(target);
    void* po = npu_t_ptr(out);

    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        float p = npu_read_compute(pp, i, pred.dtype, compute_dtype);
        float t = npu_read_compute(pt, i, target.dtype, compute_dtype);
        float d = p - t;
        sum += d * d;
    }
    npu_write_compute(po, 0, sum / (float)count, out.dtype, compute_dtype);
}
