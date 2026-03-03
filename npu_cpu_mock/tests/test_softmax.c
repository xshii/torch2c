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

/* ---- softmax_mask tests ---- */

/* helper: set bit i in bitmap */
static void set_bit(uint8_t* mask, int i) {
    mask[i / 8] |= (uint8_t)(1 << (i % 8));
}

static int test_softmax_mask_no_mask(void) {
    /* all bits=1 => identical to regular softmax */
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_masked[4] = {0}, out_plain[4] = {0};
    uint8_t mask[1] = {0x0F}; /* bits 0-3 all set */

    npu_softmax_part1(in, out_plain, 4, 4, NPU_DTYPE_FP32);
    npu_softmax_mask(in, out_masked, mask, 4, 4, NPU_DTYPE_FP32);

    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out_masked[i], out_plain[i], 1e-6f);
    return 1;
}

static int test_softmax_mask_causal(void) {
    /*
     * Causal mask for 4x4 attention:
     *   row 0: [1,0,0,0]  => only position 0
     *   row 1: [1,1,0,0]  => positions 0,1
     *   row 2: [1,1,1,0]  => positions 0,1,2
     *   row 3: [1,1,1,1]  => all positions
     *
     * Input: all 1.0 => unmasked positions get uniform probability
     */
    float in[16], out[16];
    for (int i = 0; i < 16; i++) in[i] = 1.0f;

    /* build causal bitmap: 16 bits = 2 bytes */
    uint8_t mask[2] = {0, 0};
    /* row 0 (bits 0-3): bit 0 */
    set_bit(mask, 0);
    /* row 1 (bits 4-7): bits 4,5 */
    set_bit(mask, 4); set_bit(mask, 5);
    /* row 2 (bits 8-11): bits 8,9,10 */
    set_bit(mask, 8); set_bit(mask, 9); set_bit(mask, 10);
    /* row 3 (bits 12-15): bits 12,13,14,15 */
    set_bit(mask, 12); set_bit(mask, 13); set_bit(mask, 14); set_bit(mask, 15);

    npu_softmax_mask(in, out, mask, 4, 16, NPU_DTYPE_FP32);

    /* row 0: [1.0, 0, 0, 0] */
    ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ(out[1], 0.0f, 1e-5f);
    ASSERT_FLOAT_EQ(out[2], 0.0f, 1e-5f);
    ASSERT_FLOAT_EQ(out[3], 0.0f, 1e-5f);

    /* row 1: [0.5, 0.5, 0, 0] */
    ASSERT_FLOAT_EQ(out[4], 0.5f, 1e-5f);
    ASSERT_FLOAT_EQ(out[5], 0.5f, 1e-5f);
    ASSERT_FLOAT_EQ(out[6], 0.0f, 1e-5f);
    ASSERT_FLOAT_EQ(out[7], 0.0f, 1e-5f);

    /* row 2: [1/3, 1/3, 1/3, 0] */
    ASSERT_FLOAT_EQ(out[8],  1.0f/3, 1e-5f);
    ASSERT_FLOAT_EQ(out[9],  1.0f/3, 1e-5f);
    ASSERT_FLOAT_EQ(out[10], 1.0f/3, 1e-5f);
    ASSERT_FLOAT_EQ(out[11], 0.0f,   1e-5f);

    /* row 3: [0.25, 0.25, 0.25, 0.25] */
    for (int i = 12; i < 16; i++)
        ASSERT_FLOAT_EQ(out[i], 0.25f, 1e-5f);

    /* all rows sum to 1 */
    for (int r = 0; r < 4; r++) {
        float s = 0;
        for (int c = 0; c < 4; c++) s += out[r * 4 + c];
        ASSERT_FLOAT_EQ(s, 1.0f, 1e-5f);
    }
    return 1;
}

static int test_softmax_mask_all_masked(void) {
    /* entire row masked => all zeros */
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {99, 99, 99, 99};
    uint8_t mask[1] = {0x00}; /* all masked */

    npu_softmax_mask(in, out, mask, 4, 4, NPU_DTYPE_FP32);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], 0.0f, 0.0f);
    return 1;
}

static int test_softmax_mask_partial(void) {
    /* mask out position 1: softmax over {1, _, 3, 4} */
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0};
    uint8_t mask[1] = {0};
    set_bit(mask, 0); /* keep */ set_bit(mask, 2); /* keep */ set_bit(mask, 3); /* keep */
    /* bit 1 not set => masked */

    npu_softmax_mask(in, out, mask, 4, 4, NPU_DTYPE_FP32);

    ASSERT_FLOAT_EQ(out[1], 0.0f, 0.0f); /* masked position is 0 */
    ASSERT_TRUE(out[0] > 0.0f);
    ASSERT_TRUE(out[2] > out[0]); /* 3 > 1 => bigger probability */
    ASSERT_TRUE(out[3] > out[2]); /* 4 > 3 */

    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f);
    return 1;
}

int main(void) {
    printf("test_softmax:\n");
    RUN_TEST(test_softmax_uniform);
    RUN_TEST(test_softmax_sum_one);
    RUN_TEST(test_softmax_multi_row);
    RUN_TEST(test_softmax_part2);
    RUN_TEST(test_softmax_large_values);
    RUN_TEST(test_softmax_mask_no_mask);
    RUN_TEST(test_softmax_mask_causal);
    RUN_TEST(test_softmax_mask_all_masked);
    RUN_TEST(test_softmax_mask_partial);
    TEST_SUMMARY();
}
