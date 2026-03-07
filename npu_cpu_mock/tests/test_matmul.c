#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

/* byte offsets into L1 buffer */
#define OFF_A    0
#define OFF_B    1024
#define OFF_BIAS 2048
#define OFF_OUT  3072

static int test_matmul_identity_fp32(void) {
    /* [2x3] * [3x2] = [2x2] */
    L1_INIT();
    float* a = L1_PTR(float, OFF_A);
    float* b = L1_PTR(float, OFF_B);
    float vals_a[6] = {1, 2, 3, 4, 5, 6};
    float vals_b[6] = {1, 0, 0, 1, 0, 0};
    memcpy(a, vals_a, sizeof(vals_a));
    memcpy(b, vals_b, sizeof(vals_b));
    cube_matmul(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_B, NPU_DTYPE_FP32),
                TENSOR(OFF_OUT, NPU_DTYPE_FP32), 1, 2, 2, 3, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[1], 2.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[2], 4.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[3], 5.0f, 1e-6f);
    return 1;
}

static int test_matmul_2x2_fp32(void) {
    L1_INIT();
    float* a = L1_PTR(float, OFF_A);
    float* b = L1_PTR(float, OFF_B);
    float vals_a[4] = {1, 2, 3, 4};
    float vals_b[4] = {5, 6, 7, 8};
    memcpy(a, vals_a, sizeof(vals_a));
    memcpy(b, vals_b, sizeof(vals_b));
    cube_matmul(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_B, NPU_DTYPE_FP32),
                TENSOR(OFF_OUT, NPU_DTYPE_FP32), 1, 2, 2, 2, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    ASSERT_FLOAT_EQ(out[0], 19.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[1], 22.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[2], 43.0f, 1e-6f);
    ASSERT_FLOAT_EQ(out[3], 50.0f, 1e-6f);
    return 1;
}

static int test_matmul_fp16(void) {
    L1_INIT();
    float fa[4] = {1, 2, 3, 4};
    float fb[4] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) {
        npu_write_from_float(L1_PTR(uint16_t, OFF_A), i, fa[i], NPU_DTYPE_FP16);
        npu_write_from_float(L1_PTR(uint16_t, OFF_B), i, fb[i], NPU_DTYPE_FP16);
    }
    cube_matmul(TID0, TENSOR(OFF_A, NPU_DTYPE_FP16), TENSOR(OFF_B, NPU_DTYPE_FP16),
                TENSOR(OFF_OUT, NPU_DTYPE_FP16), 1, 2, 2, 2, NPU_DTYPE_FP16);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), 0, NPU_DTYPE_FP16), 19.0f, 1e-1f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), 1, NPU_DTYPE_FP16), 22.0f, 1e-1f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), 2, NPU_DTYPE_FP16), 43.0f, 1e-1f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), 3, NPU_DTYPE_FP16), 50.0f, 1e-1f);
    return 1;
}

static int test_matmul_1x1(void) {
    L1_INIT();
    *L1_PTR(float, OFF_A) = 3.0f;
    *L1_PTR(float, OFF_B) = 4.0f;
    cube_matmul(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_B, NPU_DTYPE_FP32),
                TENSOR(OFF_OUT, NPU_DTYPE_FP32), 1, 1, 1, 1, NPU_DTYPE_FP32);
    ASSERT_FLOAT_EQ(*L1_PTR(float, OFF_OUT), 12.0f, 1e-6f);
    return 1;
}

static int test_matmul_bias_int8_fp16(void) {
    L1_INIT();
    int8_t a_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(L1_PTR(int8_t, OFF_A), i, (float)a_vals[i], NPU_DTYPE_INT8);
    int8_t b_vals[] = {1, 0, 0, 1, 1, 1};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(L1_PTR(int8_t, OFF_B), i, (float)b_vals[i], NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(uint16_t, OFF_BIAS), 0, 10.0f, NPU_DTYPE_FP16);
    npu_write_from_float(L1_PTR(uint16_t, OFF_BIAS), 1, 20.0f, NPU_DTYPE_FP16);

    cube_matmul_bias(TID0, TENSOR(OFF_A, NPU_DTYPE_INT8), TENSOR(OFF_B, NPU_DTYPE_INT8),
                     TENSOR(OFF_BIAS, NPU_DTYPE_FP16), TENSOR(OFF_OUT, NPU_DTYPE_INT8),
                     1, 2, 2, 3, NPU_DTYPE_FP16);

    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 0, NPU_DTYPE_INT8), 14.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 1, NPU_DTYPE_INT8), 25.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 2, NPU_DTYPE_INT8), 20.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 3, NPU_DTYPE_INT8), 31.0f, 0.0f);
    return 1;
}

static int test_matmul_bias_fp16_precision_loss(void) {
    L1_INIT();
    for (int i = 0; i < 4; i++) {
        npu_write_from_float(L1_PTR(int8_t, OFF_A), i, 127.0f, NPU_DTYPE_INT8);
        npu_write_from_float(L1_PTR(int8_t, OFF_B), i, 127.0f, NPU_DTYPE_INT8);
    }
    npu_write_from_float(L1_PTR(uint16_t, OFF_BIAS), 0, 0.0f, NPU_DTYPE_FP16);

    cube_matmul_bias(TID0, TENSOR(OFF_A, NPU_DTYPE_INT8), TENSOR(OFF_B, NPU_DTYPE_INT8),
                     TENSOR(OFF_BIAS, NPU_DTYPE_FP16), TENSOR(OFF_OUT, NPU_DTYPE_INT8),
                     1, 1, 1, 4, NPU_DTYPE_FP16);

    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 0, NPU_DTYPE_INT8), 127.0f, 0.0f);
    return 1;
}

static int test_matmul_bias_int8_fp32(void) {
    L1_INIT();
    int8_t a_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(L1_PTR(int8_t, OFF_A), i, (float)a_vals[i], NPU_DTYPE_INT8);
    int8_t b_vals[] = {1, 0, 0, 1, 1, 1};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(L1_PTR(int8_t, OFF_B), i, (float)b_vals[i], NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(float, OFF_BIAS), 0, 10.0f, NPU_DTYPE_FP32);
    npu_write_from_float(L1_PTR(float, OFF_BIAS), 1, 20.0f, NPU_DTYPE_FP32);

    cube_matmul_bias(TID0, TENSOR(OFF_A, NPU_DTYPE_INT8), TENSOR(OFF_B, NPU_DTYPE_INT8),
                     TENSOR(OFF_BIAS, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_INT8),
                     1, 2, 2, 3, NPU_DTYPE_FP32);

    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 0, NPU_DTYPE_INT8), 14.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 1, NPU_DTYPE_INT8), 25.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 2, NPU_DTYPE_INT8), 20.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 3, NPU_DTYPE_INT8), 31.0f, 0.0f);
    return 1;
}

static int test_matmul_bias_fp16_io_fp16_compute(void) {
    L1_INIT();
    npu_write_from_float(L1_PTR(uint16_t, OFF_A), 0, 120.0f, NPU_DTYPE_FP16);
    npu_write_from_float(L1_PTR(uint16_t, OFF_A), 1, 100.0f, NPU_DTYPE_FP16);
    npu_write_from_float(L1_PTR(uint16_t, OFF_B), 0, 100.0f, NPU_DTYPE_FP16);
    npu_write_from_float(L1_PTR(uint16_t, OFF_B), 1, 120.0f, NPU_DTYPE_FP16);
    npu_write_from_float(L1_PTR(uint16_t, OFF_BIAS), 0, 0.5f, NPU_DTYPE_FP16);

    cube_matmul_bias(TID0, TENSOR(OFF_A, NPU_DTYPE_FP16), TENSOR(OFF_B, NPU_DTYPE_FP16),
                     TENSOR(OFF_BIAS, NPU_DTYPE_FP16), TENSOR(OFF_OUT, NPU_DTYPE_FP16),
                     1, 1, 1, 2, NPU_DTYPE_FP16);

    float result = npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), 0, NPU_DTYPE_FP16);
    ASSERT_FLOAT_EQ(result, 24000.0f, 16.0f);
    return 1;
}

int main(void) {
    printf("test_matmul:\n");
    RUN_TEST(test_matmul_identity_fp32);
    RUN_TEST(test_matmul_2x2_fp32);
    RUN_TEST(test_matmul_fp16);
    RUN_TEST(test_matmul_1x1);
    RUN_TEST(test_matmul_bias_int8_fp16);
    RUN_TEST(test_matmul_bias_fp16_precision_loss);
    RUN_TEST(test_matmul_bias_int8_fp32);
    RUN_TEST(test_matmul_bias_fp16_io_fp16_compute);
    TEST_SUMMARY();
}
