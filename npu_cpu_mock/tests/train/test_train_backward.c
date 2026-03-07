#include "../test_framework.h"
#include "npu_train_api.h"
#include <string.h>
#include <math.h>

#define OFF_GRAD_OUT  0
#define OFF_INPUT     4096
#define OFF_GRAD_IN   8192
#define OFF_OUTPUT    12288
#define OFF_GAMMA     16384
#define OFF_GRAD_G    20480
#define OFF_GRAD_B    24576

static int test_gelu_backward_zero(void) {
    L1_INIT();
    /* gelu'(0) = 0.5 */
    float* grad_out = L1_PTR(float, OFF_GRAD_OUT);
    float* input = L1_PTR(float, OFF_INPUT);
    grad_out[0] = 1.0f;
    input[0] = 0.0f;

    vector_gelu_backward(TID0,
        TENSOR(OFF_GRAD_OUT, NPU_DTYPE_FP32), TENSOR(OFF_INPUT, NPU_DTYPE_FP32),
        TENSOR(OFF_GRAD_IN, NPU_DTYPE_FP32), 1, NPU_DTYPE_FP32);

    float* grad_in = L1_PTR(float, OFF_GRAD_IN);
    ASSERT_FLOAT_EQ(grad_in[0], 0.5f, 1e-4f);
    return 1;
}

static int test_softmax_backward_basic(void) {
    L1_INIT();
    /* softmax output = [0.25, 0.75], grad_out = [1, 0]
       dot = 0.25*1 + 0.75*0 = 0.25
       grad_in[0] = 0.25*(1-0.25) = 0.1875
       grad_in[1] = 0.75*(0-0.25) = -0.1875 */
    float* grad_out = L1_PTR(float, OFF_GRAD_OUT);
    float* output = L1_PTR(float, OFF_OUTPUT);
    grad_out[0] = 1.0f; grad_out[1] = 0.0f;
    output[0] = 0.25f; output[1] = 0.75f;

    vector_softmax_backward(TID0,
        TENSOR(OFF_GRAD_OUT, NPU_DTYPE_FP32), TENSOR(OFF_OUTPUT, NPU_DTYPE_FP32),
        TENSOR(OFF_GRAD_IN, NPU_DTYPE_FP32), 2, 2, NPU_DTYPE_FP32);

    float* grad_in = L1_PTR(float, OFF_GRAD_IN);
    ASSERT_FLOAT_EQ(grad_in[0], 0.1875f, 1e-5f);
    ASSERT_FLOAT_EQ(grad_in[1], -0.1875f, 1e-5f);
    return 1;
}

static int test_layernorm_backward_basic(void) {
    L1_INIT();
    /* hidden=2, seq=1, input=[1,3], gamma=[1,1], grad_out=[1,1]
       mean=2, var=1, inv_std=1, x_hat=[-1,1]
       grad_gamma = [dy*x_hat] = [-1, 1]
       grad_beta = [dy] = [1, 1] */
    float* grad_out = L1_PTR(float, OFF_GRAD_OUT);
    float* input = L1_PTR(float, OFF_INPUT);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    grad_out[0] = 1.0f; grad_out[1] = 1.0f;
    input[0] = 1.0f; input[1] = 3.0f;
    gamma[0] = 1.0f; gamma[1] = 1.0f;

    vector_layernorm_backward(TID0,
        TENSOR(OFF_GRAD_OUT, NPU_DTYPE_FP32), TENSOR(OFF_INPUT, NPU_DTYPE_FP32),
        TENSOR(OFF_GAMMA, NPU_DTYPE_FP32), TENSOR(OFF_GRAD_IN, NPU_DTYPE_FP32),
        TENSOR(OFF_GRAD_G, NPU_DTYPE_FP32), TENSOR(OFF_GRAD_B, NPU_DTYPE_FP32),
        2, 1, 1e-5f, NPU_DTYPE_FP32);

    float* grad_gamma = L1_PTR(float, OFF_GRAD_G);
    float* grad_beta = L1_PTR(float, OFF_GRAD_B);
    ASSERT_FLOAT_EQ(grad_gamma[0], -1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(grad_gamma[1], 1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(grad_beta[0], 1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(grad_beta[1], 1.0f, 1e-4f);
    return 1;
}

static int test_rmsnorm_backward_basic(void) {
    L1_INIT();
    /* hidden=2, seq=1, input=[3,4], gamma=[1,1], grad_out=[1,1], eps=1e-5
       rms = sqrt((9+16)/2 + 1e-5) = sqrt(12.5 + 1e-5)
       inv_rms = 1/rms
       x_hat = x * inv_rms = [3/rms, 4/rms]
       grad_gamma[h] = sum_s(dy * x * inv_rms) = [1*3*inv_rms, 1*4*inv_rms] */
    float* grad_out = L1_PTR(float, OFF_GRAD_OUT);
    float* input = L1_PTR(float, OFF_INPUT);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    grad_out[0] = 1.0f; grad_out[1] = 1.0f;
    input[0] = 3.0f; input[1] = 4.0f;
    gamma[0] = 1.0f; gamma[1] = 1.0f;

    vector_rmsnorm_backward(TID0,
        TENSOR(OFF_GRAD_OUT, NPU_DTYPE_FP32), TENSOR(OFF_INPUT, NPU_DTYPE_FP32),
        TENSOR(OFF_GAMMA, NPU_DTYPE_FP32), TENSOR(OFF_GRAD_IN, NPU_DTYPE_FP32),
        TENSOR(OFF_GRAD_G, NPU_DTYPE_FP32),
        2, 1, 1e-5f, NPU_DTYPE_FP32);

    float rms_val = sqrtf(12.5f + 1e-5f);
    float inv_rms = 1.0f / rms_val;

    float* grad_gamma = L1_PTR(float, OFF_GRAD_G);
    ASSERT_FLOAT_EQ(grad_gamma[0], 3.0f * inv_rms, 1e-4f);
    ASSERT_FLOAT_EQ(grad_gamma[1], 4.0f * inv_rms, 1e-4f);

    /* grad_in should be non-zero and finite */
    float* grad_in = L1_PTR(float, OFF_GRAD_IN);
    ASSERT_TRUE(isfinite(grad_in[0]));
    ASSERT_TRUE(isfinite(grad_in[1]));
    return 1;
}

int main(void) {
    printf("test_train_backward:\n");
    RUN_TEST(test_gelu_backward_zero);
    RUN_TEST(test_softmax_backward_basic);
    RUN_TEST(test_layernorm_backward_basic);
    RUN_TEST(test_rmsnorm_backward_basic);
    TEST_SUMMARY();
}
