#include "../test_framework.h"
#include "npu_train_api.h"
#include <string.h>
#include <math.h>

#define OFF_PARAM  0
#define OFF_GRAD   4096
#define OFF_VEL    8192
#define OFF_M      12288
#define OFF_V      16384

static int test_sgd_basic(void) {
    L1_INIT();
    float* param = L1_PTR(float, OFF_PARAM);
    float* grad = L1_PTR(float, OFF_GRAD);
    param[0] = 1.0f; param[1] = 2.0f;
    grad[0] = 0.1f;  grad[1] = 0.2f;

    /* no momentum, no weight_decay */
    vector_sgd_step(TID0,
        TENSOR(OFF_PARAM, NPU_DTYPE_FP32), TENSOR(OFF_GRAD, NPU_DTYPE_FP32),
        0.01f, 0.0f, 0.0f,
        TENSOR(OFF_VEL, NPU_DTYPE_FP32), 2, NPU_DTYPE_FP32);

    ASSERT_FLOAT_EQ(param[0], 1.0f - 0.01f * 0.1f, 1e-6f);
    ASSERT_FLOAT_EQ(param[1], 2.0f - 0.01f * 0.2f, 1e-6f);
    return 1;
}

static int test_adam_basic(void) {
    L1_INIT();
    float* param = L1_PTR(float, OFF_PARAM);
    float* grad = L1_PTR(float, OFF_GRAD);
    float* m = L1_PTR(float, OFF_M);
    float* v = L1_PTR(float, OFF_V);
    param[0] = 1.0f;
    grad[0] = 0.5f;
    m[0] = 0.0f;
    v[0] = 0.0f;

    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    vector_adam_step(TID0,
        TENSOR(OFF_PARAM, NPU_DTYPE_FP32), TENSOR(OFF_GRAD, NPU_DTYPE_FP32),
        TENSOR(OFF_M, NPU_DTYPE_FP32), TENSOR(OFF_V, NPU_DTYPE_FP32),
        lr, beta1, beta2, eps, 1, 1, NPU_DTYPE_FP32);

    /* m1 = 0.1*0.5 = 0.05, v1 = 0.001*0.25 = 0.00025 */
    /* m_hat = 0.05/0.1 = 0.5, v_hat = 0.00025/0.001 = 0.25 */
    /* param = 1.0 - 0.001 * 0.5 / (sqrt(0.25)+1e-8) = 1.0 - 0.001 */
    ASSERT_FLOAT_EQ(m[0], 0.05f, 1e-6f);
    ASSERT_FLOAT_EQ(v[0], 0.00025f, 1e-7f);
    float m_hat = 0.05f / (1.0f - 0.9f);
    float v_hat = 0.00025f / (1.0f - 0.999f);
    float expected = 1.0f - lr * m_hat / (sqrtf(v_hat) + eps);
    ASSERT_FLOAT_EQ(param[0], expected, 1e-5f);
    return 1;
}

static int test_grad_clip(void) {
    L1_INIT();
    float* grad = L1_PTR(float, OFF_GRAD);
    grad[0] = 3.0f; grad[1] = 4.0f;
    /* norm = 5.0, max_norm = 2.5 => scale = 0.5 */
    vector_grad_clip(TID0,
        TENSOR(OFF_GRAD, NPU_DTYPE_FP32), 2.5f, 2, NPU_DTYPE_FP32);

    ASSERT_FLOAT_EQ(grad[0], 1.5f, 1e-5f);
    ASSERT_FLOAT_EQ(grad[1], 2.0f, 1e-5f);
    return 1;
}

int main(void) {
    printf("test_train_optimizer:\n");
    RUN_TEST(test_sgd_basic);
    RUN_TEST(test_adam_basic);
    RUN_TEST(test_grad_clip);
    TEST_SUMMARY();
}
