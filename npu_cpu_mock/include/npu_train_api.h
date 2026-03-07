#ifndef NPU_TRAIN_API_H
#define NPU_TRAIN_API_H

#include "npu_api.h"

/* ============================================================
 * Training-only ops (incremental over inference ops)
 * Organized by compute unit: cube → vector
 * ============================================================ */

/* ---- cube training ops ---- */
/* backward matmul: dX = dY @ W^T, dW = X^T @ dY — use cube_matmul + transpose */

/* ---- vector training ops: loss ---- */
void vector_cross_entropy_loss(TidInfo tid, npu_tensor_t logits, npu_tensor_t labels,
                               npu_tensor_t out, int batch, int num_classes,
                               npu_dtype_t compute_dtype);
void vector_mse_loss(TidInfo tid, npu_tensor_t pred, npu_tensor_t target,
                     npu_tensor_t out, int count, npu_dtype_t compute_dtype);

/* ---- vector training ops: activation backward ---- */
void vector_gelu_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t input,
                          npu_tensor_t grad_in, int count, npu_dtype_t compute_dtype);
void vector_softmax_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t output,
                             npu_tensor_t grad_in, int dim, int count,
                             npu_dtype_t compute_dtype);

/* ---- vector training ops: norm backward ---- */
void vector_layernorm_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t input,
                               npu_tensor_t gamma, npu_tensor_t grad_in,
                               npu_tensor_t grad_gamma, npu_tensor_t grad_beta,
                               int hidden, int seq, float eps,
                               npu_dtype_t compute_dtype);
void vector_rmsnorm_backward(TidInfo tid, npu_tensor_t grad_out, npu_tensor_t input,
                             npu_tensor_t gamma, npu_tensor_t grad_in,
                             npu_tensor_t grad_gamma,
                             int hidden, int seq, float eps,
                             npu_dtype_t compute_dtype);

/* ---- vector training ops: optimizer ---- */
void vector_sgd_step(TidInfo tid, npu_tensor_t param, npu_tensor_t grad,
                     float lr, float momentum, float weight_decay,
                     npu_tensor_t velocity, int count, npu_dtype_t compute_dtype);
void vector_adam_step(TidInfo tid, npu_tensor_t param, npu_tensor_t grad,
                      npu_tensor_t m, npu_tensor_t v,
                      float lr, float beta1, float beta2, float eps,
                      int step, int count, npu_dtype_t compute_dtype);
void vector_grad_clip(TidInfo tid, npu_tensor_t grad, float max_norm,
                      int count, npu_dtype_t compute_dtype);

#endif /* NPU_TRAIN_API_H */
