#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

static int test_layernorm_basic(void) {
    /* seq=1, hidden=4, gamma=1, beta=0 => standard normalization */
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4]   = {0};
    npu_layernorm_part1(input, gamma, beta, out, 4, 1, 1e-5f, NPU_DTYPE_FP32);

    /* mean=2.5, var=1.25, std=~1.118 */
    /* expected: (x - 2.5) / sqrt(1.25 + 1e-5) */
    float mean = 2.5f, var = 1.25f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], (input[i] - mean) * inv_std, 1e-5f);
    return 1;
}

static int test_layernorm_with_gamma_beta(void) {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float beta[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4]   = {0};
    npu_layernorm_part1(input, gamma, beta, out, 4, 1, 1e-5f, NPU_DTYPE_FP32);

    float mean = 2.5f, var = 1.25f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < 4; i++) {
        float expected = 2.0f * (input[i] - mean) * inv_std + 1.0f;
        ASSERT_FLOAT_EQ(out[i], expected, 1e-5f);
    }
    return 1;
}

static int test_layernorm_multi_seq(void) {
    /* seq=2, hidden=2 */
    float input[4] = {1.0f, 3.0f, 2.0f, 4.0f};
    float gamma[2] = {1.0f, 1.0f};
    float beta[2]  = {0.0f, 0.0f};
    float out[4]   = {0};
    npu_layernorm_part1(input, gamma, beta, out, 2, 2, 1e-5f, NPU_DTYPE_FP32);

    /* seq0: mean=2, var=1, (1-2)/1=-1, (3-2)/1=1 */
    ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(out[1],  1.0f, 1e-4f);
    /* seq1: mean=3, var=1, (2-3)/1=-1, (4-3)/1=1 */
    ASSERT_FLOAT_EQ(out[2], -1.0f, 1e-4f);
    ASSERT_FLOAT_EQ(out[3],  1.0f, 1e-4f);
    return 1;
}

static int test_layernorm_part2(void) {
    float inter[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float orig[8]  = {0};
    float out[8]   = {0};
    int byte_count = 8 * (int)sizeof(float);
    npu_layernorm_part2(inter, orig, out, byte_count, NPU_DTYPE_FP32);
    for (int i = 0; i < 8; i++)
        ASSERT_FLOAT_EQ(out[i], inter[i], 0.0f);
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
