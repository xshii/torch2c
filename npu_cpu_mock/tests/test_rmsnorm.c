#include "test_framework.h"
#include "npu_api.h"
#include <string.h>
#include <math.h>

#define OFF_IN    0
#define OFF_GAMMA 1024
#define OFF_OUT   3072

static int test_rmsnorm_basic(void) {
    L1_INIT();
    float* input = L1_PTR(float, OFF_IN);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    float vals_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vals_g[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
    memcpy(input, vals_in, sizeof(vals_in));
    memcpy(gamma, vals_g, sizeof(vals_g));

    vector_rmsnorm_part1(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_GAMMA, NPU_DTYPE_FP32),
                         TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         4, 1, 1e-5f, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    /* RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) */
    float rms = sqrtf(7.5f + 1e-5f);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], vals_in[i] / rms, 1e-5f);
    return 1;
}

static int test_rmsnorm_with_gamma(void) {
    L1_INIT();
    float* input = L1_PTR(float, OFF_IN);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    float vals_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vals_g[4]  = {2.0f, 0.5f, 1.0f, 3.0f};
    memcpy(input, vals_in, sizeof(vals_in));
    memcpy(gamma, vals_g, sizeof(vals_g));

    vector_rmsnorm_part1(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_GAMMA, NPU_DTYPE_FP32),
                         TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         4, 1, 1e-5f, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    float rms = sqrtf(7.5f + 1e-5f);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], vals_in[i] / rms * vals_g[i], 1e-5f);
    return 1;
}

static int test_rmsnorm_multi_seq(void) {
    L1_INIT();
    float* input = L1_PTR(float, OFF_IN);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    float vals_in[4] = {3.0f, 4.0f, 1.0f, 1.0f};
    float vals_g[2]  = {1.0f, 1.0f};
    memcpy(input, vals_in, sizeof(vals_in));
    memcpy(gamma, vals_g, sizeof(vals_g));

    vector_rmsnorm_part1(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_GAMMA, NPU_DTYPE_FP32),
                         TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         2, 2, 1e-5f, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    /* seq 0: RMS = sqrt((9+16)/2) = sqrt(12.5) */
    float rms0 = sqrtf(12.5f + 1e-5f);
    ASSERT_FLOAT_EQ(out[0], 3.0f / rms0, 1e-5f);
    ASSERT_FLOAT_EQ(out[1], 4.0f / rms0, 1e-5f);
    /* seq 1: RMS = sqrt((1+1)/2) = 1.0 */
    float rms1 = sqrtf(1.0f + 1e-5f);
    ASSERT_FLOAT_EQ(out[2], 1.0f / rms1, 1e-5f);
    ASSERT_FLOAT_EQ(out[3], 1.0f / rms1, 1e-5f);
    return 1;
}

static int test_rmsnorm_wrapper(void) {
    L1_INIT();
    float* input = L1_PTR(float, OFF_IN);
    float* gamma = L1_PTR(float, OFF_GAMMA);
    float vals_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vals_g[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
    memcpy(input, vals_in, sizeof(vals_in));
    memcpy(gamma, vals_g, sizeof(vals_g));

    vector_rmsnorm(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_GAMMA, NPU_DTYPE_FP32),
                   TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                   4, 1, 1e-5f, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    float rms = sqrtf(7.5f + 1e-5f);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], vals_in[i] / rms, 1e-5f);
    return 1;
}

static int test_rmsnorm_part2(void) {
    L1_INIT();
    float* inter = L1_PTR(float, OFF_IN);
    float vals[4] = {1, 2, 3, 4};
    memcpy(inter, vals, sizeof(vals));
    int byte_count = 4 * (int)sizeof(float);

    #define OFF_ORIG 4096
    vector_rmsnorm_part2(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_ORIG, NPU_DTYPE_FP32),
                         TENSOR(OFF_OUT, NPU_DTYPE_FP32), byte_count, NPU_DTYPE_FP32);

    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], vals[i], 0.0f);
    return 1;
}

int main(void) {
    printf("test_rmsnorm:\n");
    RUN_TEST(test_rmsnorm_basic);
    RUN_TEST(test_rmsnorm_with_gamma);
    RUN_TEST(test_rmsnorm_multi_seq);
    RUN_TEST(test_rmsnorm_wrapper);
    RUN_TEST(test_rmsnorm_part2);
    TEST_SUMMARY();
}
