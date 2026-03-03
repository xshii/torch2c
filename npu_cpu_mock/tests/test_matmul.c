#include "test_framework.h"
#include "npu_api.h"

static int test_matmul_identity_fp32(void) {
    /* [2x3] * [3x2] = [2x2] */
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {1, 0, 0, 1, 0, 0};
    float out[4] = {0};
    npu_matmul(a, b, out, 2, 2, 3, NPU_DTYPE_FP32, NPU_FORMAT_ND);
    /* row0: 1*1+2*0+3*0=1, 1*0+2*1+3*0=2 */
    /* row1: 4*1+5*0+6*0=4, 4*0+5*1+6*0=5 */
    ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[1], 2.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[2], 4.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[3], 5.0f, 1e-6f);
    return 1;
}

static int test_matmul_2x2_fp32(void) {
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float out[4] = {0};
    npu_matmul(a, b, out, 2, 2, 2, NPU_DTYPE_FP32, NPU_FORMAT_ND);
    /* [1*5+2*7, 1*6+2*8] = [19, 22] */
    /* [3*5+4*7, 3*6+4*8] = [43, 50] */
    ASSERT_FLOAT_EQ(out[0], 19.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[1], 22.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[2], 43.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[3], 50.0f, 1e-6f);
    return 1;
}

static int test_matmul_fp16(void) {
    uint16_t a[4], b[4], out[4];
    float fa[4] = {1, 2, 3, 4};
    float fb[4] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) {
        npu_write_from_float(a, i, fa[i], NPU_DTYPE_FP16);
        npu_write_from_float(b, i, fb[i], NPU_DTYPE_FP16);
    }
    npu_matmul(a, b, out, 2, 2, 2, NPU_DTYPE_FP16, NPU_FORMAT_ND);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_FP16), 19.0f, 1e-1f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 1, NPU_DTYPE_FP16), 22.0f, 1e-1f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 2, NPU_DTYPE_FP16), 43.0f, 1e-1f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 3, NPU_DTYPE_FP16), 50.0f, 1e-1f);
    return 1;
}

static int test_matmul_1x1(void) {
    float a[1] = {3.0f}, b[1] = {4.0f}, out[1] = {0};
    npu_matmul(a, b, out, 1, 1, 1, NPU_DTYPE_FP32, NPU_FORMAT_ND);
    ASSERT_FLOAT_EQ(out[0], 12.0f, 1e-6f);
    return 1;
}

int main(void) {
    printf("test_matmul:\n");
    RUN_TEST(test_matmul_identity_fp32);
    RUN_TEST(test_matmul_2x2_fp32);
    RUN_TEST(test_matmul_fp16);
    RUN_TEST(test_matmul_1x1);
    TEST_SUMMARY();
}
