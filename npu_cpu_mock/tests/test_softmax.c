#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

static int test_softmax_uniform(void) {
    /* equal inputs => uniform output */
    float in[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {0};
    npu_softmax_part1(in, out, 4, 4, NPU_DTYPE_FP32);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], 0.25f, 1e-5f);
    return 1;
}

static int test_softmax_sum_one(void) {
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0};
    npu_softmax_part1(in, out, 4, 4, NPU_DTYPE_FP32);
    float sum = 0;
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(out[i] > 0.0f);
        sum += out[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f);
    /* monotonicity: out[0] < out[1] < out[2] < out[3] */
    for (int i = 1; i < 4; i++)
        ASSERT_TRUE(out[i] > out[i-1]);
    return 1;
}

static int test_softmax_multi_row(void) {
    /* 2 rows of dim=3 */
    float in[6] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    float out[6] = {0};
    npu_softmax_part1(in, out, 3, 6, NPU_DTYPE_FP32);

    /* row 0 sums to 1 */
    float s0 = out[0] + out[1] + out[2];
    ASSERT_FLOAT_EQ(s0, 1.0f, 1e-5f);

    /* row 1 is uniform */
    float s1 = out[3] + out[4] + out[5];
    ASSERT_FLOAT_EQ(s1, 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ(out[3], out[4], 1e-5f);
    ASSERT_FLOAT_EQ(out[4], out[5], 1e-5f);
    return 1;
}

static int test_softmax_part2(void) {
    float inter[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float out[4] = {0};
    int byte_count = 4 * (int)sizeof(float);
    npu_softmax_part2(inter, out, byte_count, NPU_DTYPE_FP32);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], inter[i], 0.0f);
    return 1;
}

static int test_softmax_large_values(void) {
    /* numerical stability: large values should not overflow */
    float in[3] = {1000.0f, 1001.0f, 1002.0f};
    float out[3] = {0};
    npu_softmax_part1(in, out, 3, 3, NPU_DTYPE_FP32);
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
    RUN_TEST(test_softmax_large_values);
    TEST_SUMMARY();
}
