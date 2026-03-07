#include "../test_framework.h"
#include "npu_train_api.h"
#include <string.h>
#include <math.h>

#define OFF_LOGITS  0
#define OFF_LABELS  4096
#define OFF_OUT     8192
#define OFF_TARGET  12288

static int test_cross_entropy_basic(void) {
    L1_INIT();
    /* batch=1, classes=3, logits=[1,2,3], label=2 */
    float* logits = L1_PTR(float, OFF_LOGITS);
    float logit_vals[3] = {1.0f, 2.0f, 3.0f};
    memcpy(logits, logit_vals, sizeof(logit_vals));
    /* label = class 2 */
    npu_write_from_float(L1_PTR(float, OFF_LABELS), 0, 2.0f, NPU_DTYPE_FP32);

    vector_cross_entropy_loss(TID0,
        TENSOR(OFF_LOGITS, NPU_DTYPE_FP32), TENSOR(OFF_LABELS, NPU_DTYPE_FP32),
        TENSOR(OFF_OUT, NPU_DTYPE_FP32), 1, 3, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    /* expected: -log(softmax(logits)[2]) = -log(exp(3)/(exp(1)+exp(2)+exp(3))) */
    float max_v = 3.0f;
    float sum_exp = expf(1.0f - max_v) + expf(2.0f - max_v) + expf(3.0f - max_v);
    float expected = -(3.0f - max_v - logf(sum_exp));
    ASSERT_FLOAT_EQ(out[0], expected, 1e-5f);
    return 1;
}

static int test_mse_loss_basic(void) {
    L1_INIT();
    float* pred = L1_PTR(float, OFF_LOGITS);
    float* target = L1_PTR(float, OFF_TARGET);
    float p[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float t[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    memcpy(pred, p, sizeof(p));
    memcpy(target, t, sizeof(t));

    vector_mse_loss(TID0,
        TENSOR(OFF_LOGITS, NPU_DTYPE_FP32), TENSOR(OFF_TARGET, NPU_DTYPE_FP32),
        TENSOR(OFF_OUT, NPU_DTYPE_FP32), 4, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    /* 4 * 0.25 / 4 = 0.25 */
    ASSERT_FLOAT_EQ(out[0], 0.25f, 1e-6f);
    return 1;
}

int main(void) {
    printf("test_train_loss:\n");
    RUN_TEST(test_cross_entropy_basic);
    RUN_TEST(test_mse_loss_basic);
    TEST_SUMMARY();
}
