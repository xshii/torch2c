#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

#define OFF_IN   0
#define OFF_OUT  1024

static int test_softmax_uniform(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    memcpy(in, vals, sizeof(vals));
    vector_softmax_part1(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         4, 4, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], 0.25f, 1e-5f);
    return 1;
}

static int test_softmax_sum_one(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(in, vals, sizeof(vals));
    vector_softmax_part1(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         4, 4, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    float sum = 0;
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(out[i] > 0.0f);
        sum += out[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f);
    for (int i = 1; i < 4; i++)
        ASSERT_TRUE(out[i] > out[i-1]);
    return 1;
}

static int test_softmax_multi_row(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[6] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    memcpy(in, vals, sizeof(vals));
    vector_softmax_part1(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         3, 6, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    float s0 = out[0] + out[1] + out[2];
    ASSERT_FLOAT_EQ(s0, 1.0f, 1e-5f);
    float s1 = out[3] + out[4] + out[5];
    ASSERT_FLOAT_EQ(s1, 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ(out[3], out[4], 1e-5f);
    ASSERT_FLOAT_EQ(out[4], out[5], 1e-5f);
    return 1;
}

static int test_softmax_part2(void) {
    L1_INIT();
    float* inter = L1_PTR(float, OFF_IN);
    float vals[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    memcpy(inter, vals, sizeof(vals));
    int byte_count = 4 * (int)sizeof(float);
    vector_softmax_part2(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         byte_count, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], vals[i], 0.0f);
    return 1;
}

static int test_softmax_wrapper(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(in, vals, sizeof(vals));
    vector_softmax(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                   4, 4, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    float sum = 0;
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(out[i] > 0.0f);
        sum += out[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f);
    return 1;
}

static int test_softmax_large_values(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[3] = {1000.0f, 1001.0f, 1002.0f};
    memcpy(in, vals, sizeof(vals));
    vector_softmax_part1(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                         3, 3, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    float sum = out[0] + out[1] + out[2];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f);
    for (int i = 0; i < 3; i++)
        ASSERT_TRUE(out[i] > 0.0f && out[i] <= 1.0f);
    return 1;
}

int main(void) {
    printf("test_softmax:\n");
    RUN_TEST(test_softmax_uniform);
    RUN_TEST(test_softmax_sum_one);
    RUN_TEST(test_softmax_multi_row);
    RUN_TEST(test_softmax_part2);
    RUN_TEST(test_softmax_wrapper);
    RUN_TEST(test_softmax_large_values);
    TEST_SUMMARY();
}
