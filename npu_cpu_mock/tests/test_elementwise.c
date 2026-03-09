#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

#define N 8
#define OFF_A   0
#define OFF_B   1024
#define OFF_OUT 2048

static int test_add_fp32(void) {
    L1_INIT();
    float* a = L1_PTR(float, OFF_A);
    float* b = L1_PTR(float, OFF_B);
    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(i * 2); }
    vector_add(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_B, NPU_DTYPE_FP32),
               TENSOR(OFF_OUT, NPU_DTYPE_FP32), N, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], (float)(i + i * 2), 1e-6f);
    return 1;
}

static int test_add_fp16(void) {
    L1_INIT();
    for (int i = 0; i < N; i++) {
        npu_write_from_float(L1_PTR(uint16_t, OFF_A), i, (float)i, NPU_DTYPE_FP16);
        npu_write_from_float(L1_PTR(uint16_t, OFF_B), i, (float)(i * 2), NPU_DTYPE_FP16);
    }
    vector_add(TID0, TENSOR(OFF_A, NPU_DTYPE_FP16), TENSOR(OFF_B, NPU_DTYPE_FP16),
               TENSOR(OFF_OUT, NPU_DTYPE_FP16), N, NPU_DTYPE_FP16);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), i, NPU_DTYPE_FP16), (float)(i * 3), 1e-2f);
    return 1;
}

static int test_mul_fp32(void) {
    L1_INIT();
    float* a = L1_PTR(float, OFF_A);
    float* b = L1_PTR(float, OFF_B);
    for (int i = 0; i < N; i++) { a[i] = (float)i + 1.0f; b[i] = 2.0f; }
    vector_mul(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_B, NPU_DTYPE_FP32),
               TENSOR(OFF_OUT, NPU_DTYPE_FP32), N, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], ((float)i + 1.0f) * 2.0f, 1e-6f);
    return 1;
}

static int test_mul_scalar_fp32(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_A);
    for (int i = 0; i < N; i++) in[i] = (float)i + 1.0f;
    vector_mul_scalar(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                      3.0f, N, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], ((float)i + 1.0f) * 3.0f, 1e-6f);
    return 1;
}

static int test_sub_fp32(void) {
    L1_INIT();
    float* a = L1_PTR(float, OFF_A);
    float* b = L1_PTR(float, OFF_B);
    for (int i = 0; i < N; i++) { a[i] = (float)(i * 3); b[i] = (float)i; }
    vector_sub(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_B, NPU_DTYPE_FP32),
               TENSOR(OFF_OUT, NPU_DTYPE_FP32), N, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], (float)(i * 2), 1e-6f);
    return 1;
}

static int test_div_fp32(void) {
    L1_INIT();
    float* a = L1_PTR(float, OFF_A);
    float* b = L1_PTR(float, OFF_B);
    for (int i = 0; i < N; i++) { a[i] = (float)(i + 1) * 6.0f; b[i] = 3.0f; }
    vector_div(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_B, NPU_DTYPE_FP32),
               TENSOR(OFF_OUT, NPU_DTYPE_FP32), N, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], (float)(i + 1) * 2.0f, 1e-6f);
    return 1;
}

static int test_fill_fp32(void) {
    L1_INIT();
    vector_fill(TID0, TENSOR(OFF_OUT, NPU_DTYPE_FP32), -1e30f, 4, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(out[i], -1e30f, 0.0f);
    return 1;
}

#define OFF_MASK 3072
static int test_dropout_fp32(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_A);
    float* mask = L1_PTR(float, OFF_MASK);
    for (int i = 0; i < 4; i++) in[i] = (float)(i + 1);
    /* mask: 1,0,1,0 → keep odds, drop evens */
    mask[0] = 1.0f; mask[1] = 0.0f; mask[2] = 1.0f; mask[3] = 0.0f;
    float scale = 2.0f; /* 1/(1-p) where p=0.5 */
    vector_dropout(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                   TENSOR(OFF_MASK, NPU_DTYPE_FP32), 4, scale, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    ASSERT_FLOAT_EQ(out[0], 2.0f, 1e-6f);  /* 1 * 2 */
    ASSERT_FLOAT_EQ(out[1], 0.0f, 1e-6f);  /* dropped */
    ASSERT_FLOAT_EQ(out[2], 6.0f, 1e-6f);  /* 3 * 2 */
    ASSERT_FLOAT_EQ(out[3], 0.0f, 1e-6f);  /* dropped */
    return 1;
}

static int test_gelu_fp32(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_A);
    float vals[4] = {0.0f, 1.0f, -1.0f, 2.0f};
    memcpy(in, vals, sizeof(vals));
    vector_gelu(TID0, TENSOR(OFF_A, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                4, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-4f);
    ASSERT_FLOAT_EQ(out[1], 0.8413f, 1e-3f);
    ASSERT_FLOAT_EQ(out[2], -0.1587f, 1e-3f);
    ASSERT_FLOAT_EQ(out[3], 1.9545f, 1e-3f);
    return 1;
}

static int test_gelu_fp16(void) {
    L1_INIT();
    float vals[] = {0.0f, 1.0f, -1.0f, 2.0f};
    float expected[] = {0.0f, 0.8413f, -0.1587f, 1.9545f};
    for (int i = 0; i < 4; i++)
        npu_write_from_float(L1_PTR(uint16_t, OFF_A), i, vals[i], NPU_DTYPE_FP16);
    vector_gelu(TID0, TENSOR(OFF_A, NPU_DTYPE_FP16), TENSOR(OFF_OUT, NPU_DTYPE_FP16),
                4, NPU_DTYPE_FP16);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), i, NPU_DTYPE_FP16), expected[i], 5e-3f);
    return 1;
}

static int test_add_int8(void) {
    L1_INIT();
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 0, 10.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 1, 50.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 2, -10.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 3, 100.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 0, 20.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 1, 60.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 2, -20.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 3, 100.0f, NPU_DTYPE_INT8);
    vector_add(TID0, TENSOR(OFF_A, NPU_DTYPE_INT8), TENSOR(OFF_B, NPU_DTYPE_INT8),
               TENSOR(OFF_OUT, NPU_DTYPE_INT8), 4, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 0, NPU_DTYPE_INT8), 30.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 1, NPU_DTYPE_INT8), 110.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 2, NPU_DTYPE_INT8), -30.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 3, NPU_DTYPE_INT8), 127.0f, 0.0f);
    return 1;
}

static int test_add_int16(void) {
    L1_INIT();
    npu_write_from_float(L1_PTR(int16_t, OFF_A), 0, 1000.0f, NPU_DTYPE_INT16);
    npu_write_from_float(L1_PTR(int16_t, OFF_A), 1, -500.0f, NPU_DTYPE_INT16);
    npu_write_from_float(L1_PTR(int16_t, OFF_B), 0, 2000.0f, NPU_DTYPE_INT16);
    npu_write_from_float(L1_PTR(int16_t, OFF_B), 1, -300.0f, NPU_DTYPE_INT16);
    vector_add(TID0, TENSOR(OFF_A, NPU_DTYPE_INT16), TENSOR(OFF_B, NPU_DTYPE_INT16),
               TENSOR(OFF_OUT, NPU_DTYPE_INT16), 2, NPU_DTYPE_INT16);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int16_t, OFF_OUT), 0, NPU_DTYPE_INT16), 3000.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int16_t, OFF_OUT), 1, NPU_DTYPE_INT16), -800.0f, 0.0f);
    return 1;
}

static int test_mul_int8(void) {
    L1_INIT();
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 0, 5.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 1, -3.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 0, 4.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 1, 7.0f, NPU_DTYPE_INT8);
    vector_mul(TID0, TENSOR(OFF_A, NPU_DTYPE_INT8), TENSOR(OFF_B, NPU_DTYPE_INT8),
               TENSOR(OFF_OUT, NPU_DTYPE_INT8), 2, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 0, NPU_DTYPE_INT8), 20.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 1, NPU_DTYPE_INT8), -21.0f, 0.0f);
    return 1;
}

static int test_matmul_int8(void) {
    /* [1x2] * [2x1] = [1x1]: 3*4 + 5*6 = 42 */
    L1_INIT();
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 0, 3.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_A), 1, 5.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 0, 4.0f, NPU_DTYPE_INT8);
    npu_write_from_float(L1_PTR(int8_t, OFF_B), 1, 6.0f, NPU_DTYPE_INT8);
    cube_matmul(TID0, TENSOR(OFF_A, NPU_DTYPE_INT8), TENSOR(OFF_B, NPU_DTYPE_INT8),
                TENSOR(OFF_OUT, NPU_DTYPE_INT8), 1, 1, 1, 2, 0, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(int8_t, OFF_OUT), 0, NPU_DTYPE_INT8), 42.0f, 0.0f);
    return 1;
}

int main(void) {
    printf("test_elementwise:\n");
    RUN_TEST(test_add_fp32);
    RUN_TEST(test_add_fp16);
    RUN_TEST(test_sub_fp32);
    RUN_TEST(test_div_fp32);
    RUN_TEST(test_add_int8);
    RUN_TEST(test_add_int16);
    RUN_TEST(test_mul_fp32);
    RUN_TEST(test_mul_int8);
    RUN_TEST(test_mul_scalar_fp32);
    RUN_TEST(test_fill_fp32);
    RUN_TEST(test_dropout_fp32);
    RUN_TEST(test_gelu_fp32);
    RUN_TEST(test_gelu_fp16);
    RUN_TEST(test_matmul_int8);
    TEST_SUMMARY();
}
