#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

#define OFF_IN    0
#define OFF_GAMMA 1024
#define OFF_BETA  2048
#define OFF_OUT   3072
#define OFF_ORIG  4096

static int test_layernorm_basic(void) {
    L1_INIT();
    float* input = L1_PTR(float, OFF_IN);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    float* beta  = L1_PTR(float, OFF_BETA);
    float vals_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vals_g[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float vals_b[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    memcpy(input, vals_in, sizeof(vals_in));
    memcpy(gamma, vals_g, sizeof(vals_g));
    memcpy(beta, vals_b, sizeof(vals_b));

    vector_layernorm_part1(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_GAMMA, NPU_DTYPE_FP32),
                           TENSOR(OFF_BETA, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                           4, 1, 1e-5f, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    float mean = 2.5f, var = 1.25f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], (vals_in[i] - mean) * inv_std, 1e-5f);
    return 1;
}

static int test_layernorm_with_gamma_beta(void) {
    L1_INIT();
    float* input = L1_PTR(float, OFF_IN);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    float* beta  = L1_PTR(float, OFF_BETA);
    float vals_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vals_g[4]  = {2.0f, 2.0f, 2.0f, 2.0f};
    float vals_b[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
    memcpy(input, vals_in, sizeof(vals_in));
    memcpy(gamma, vals_g, sizeof(vals_g));
    memcpy(beta, vals_b, sizeof(vals_b));

    vector_layernorm_part1(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_GAMMA, NPU_DTYPE_FP32),
                           TENSOR(OFF_BETA, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                           4, 1, 1e-5f, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    float mean = 2.5f, var = 1.25f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < 4; i++) {
        float expected = 2.0f * (vals_in[i] - mean) * inv_std + 1.0f;
        ASSERT_FLOAT_EQ(out[i], expected, 1e-5f);
    }
    return 1;
}

static int test_layernorm_multi_seq(void) {
    L1_INIT();
    float* input = L1_PTR(float, OFF_IN);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    float* beta  = L1_PTR(float, OFF_BETA);
    float vals_in[4] = {1.0f, 3.0f, 2.0f, 4.0f};
    float vals_g[2]  = {1.0f, 1.0f};
    float vals_b[2]  = {0.0f, 0.0f};
    memcpy(input, vals_in, sizeof(vals_in));
    memcpy(gamma, vals_g, sizeof(vals_g));
    memcpy(beta, vals_b, sizeof(vals_b));

    vector_layernorm_part1(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_GAMMA, NPU_DTYPE_FP32),
                           TENSOR(OFF_BETA, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                           2, 2, 1e-5f, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(out[1],  1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(out[2], -1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(out[3],  1.0f, 1e-4f);
    return 1;
}

static int test_layernorm_part2(void) {
    L1_INIT();
    float* inter = L1_PTR(float, OFF_IN);
    float vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(inter, vals, sizeof(vals));
    int byte_count = 8 * (int)sizeof(float);

    vector_layernorm_part2(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_ORIG, NPU_DTYPE_FP32),
                           TENSOR(OFF_OUT, NPU_DTYPE_FP32), byte_count, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 8; i++)
        ASSERT_FLOAT_EQ(out[i], vals[i], 0.0f);
    return 1;
}

int main(void) {
    printf("test_norm:\n");
    RUN_TEST(test_layernorm_basic);
    RUN_TEST(test_layernorm_with_gamma_beta);
    RUN_TEST(test_layernorm_multi_seq);
    RUN_TEST(test_layernorm_part2);
    TEST_SUMMARY();
}
