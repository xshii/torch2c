#include "test_framework.h"
#include "npu_api.h"

#define N 8

static int test_add_fp32(void) {
    float a[N], b[N], out[N];
    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(i * 2); }
    npu_add(a, b, out, N, NPU_DTYPE_FP32);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], (float)(i + i * 2), 1e-6f);
    return 1;
}

static int test_add_fp16(void) {
    uint16_t a[N], b[N], out[N];
    for (int i = 0; i < N; i++) {
        npu_write_from_float(a, i, (float)i, NPU_DTYPE_FP16);
        npu_write_from_float(b, i, (float)(i * 2), NPU_DTYPE_FP16);
    }
    npu_add(a, b, out, N, NPU_DTYPE_FP16);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(npu_read_as_float(out, i, NPU_DTYPE_FP16), (float)(i * 3), 1e-2f);
    return 1;
}

static int test_mul_fp32(void) {
    float a[N], b[N], out[N];
    for (int i = 0; i < N; i++) { a[i] = (float)i + 1.0f; b[i] = 2.0f; }
    npu_mul(a, b, out, N, NPU_DTYPE_FP32);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], ((float)i + 1.0f) * 2.0f, 1e-6f);
    return 1;
}

static int test_mul_scalar_fp32(void) {
    float in[N], out[N];
    for (int i = 0; i < N; i++) in[i] = (float)i + 1.0f;
    npu_mul_scalar(in, out, 3.0f, N, NPU_DTYPE_FP32);
    for (int i = 0; i < N; i++)
        ASSERT_FLOAT_EQ(out[i], ((float)i + 1.0f) * 3.0f, 1e-6f);
    return 1;
}

static int test_gelu_fp32(void) {
    float in[4] = {0.0f, 1.0f, -1.0f, 2.0f};
    float out[4];
    npu_gelu(in, out, 4, NPU_DTYPE_FP32);
    /* gelu(0) = 0, gelu(1) ≈ 0.8413, gelu(-1) ≈ -0.1587, gelu(2) ≈ 1.9545 */
    ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-4f);
    ASSERT_FLOAT_EQ(out[1], 0.8413f, 1e-3f);
    ASSERT_FLOAT_EQ(out[2], -0.1587f, 1e-3f);
    ASSERT_FLOAT_EQ(out[3], 1.9545f, 1e-3f);
    return 1;
}

static int test_gelu_fp16(void) {
    uint16_t in[4], out[4];
    float vals[] = {0.0f, 1.0f, -1.0f, 2.0f};
    float expected[] = {0.0f, 0.8413f, -0.1587f, 1.9545f};
    for (int i = 0; i < 4; i++)
        npu_write_from_float(in, i, vals[i], NPU_DTYPE_FP16);
    npu_gelu(in, out, 4, NPU_DTYPE_FP16);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(npu_read_as_float(out, i, NPU_DTYPE_FP16), expected[i], 5e-3f);
    return 1;
}

static int test_add_int8(void) {
    int8_t a[4], b[4], out[4];
    /* 10+20=30, 50+60=110, (-10)+(-20)=-30, 100+100=127 (clamped) */
    npu_write_from_float(a, 0, 10.0f, NPU_DTYPE_INT8);
    npu_write_from_float(a, 1, 50.0f, NPU_DTYPE_INT8);
    npu_write_from_float(a, 2, -10.0f, NPU_DTYPE_INT8);
    npu_write_from_float(a, 3, 100.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 0, 20.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 1, 60.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 2, -20.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 3, 100.0f, NPU_DTYPE_INT8);
    npu_add(a, b, out, 4, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_INT8), 30.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 1, NPU_DTYPE_INT8), 110.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 2, NPU_DTYPE_INT8), -30.0f, 0.0f);
    /* 100+100=200 => clamped to 127 */
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 3, NPU_DTYPE_INT8), 127.0f, 0.0f);
    return 1;
}

static int test_add_int16(void) {
    int16_t a[4], b[4], out[4];
    npu_write_from_float(a, 0, 1000.0f, NPU_DTYPE_INT16);
    npu_write_from_float(a, 1, -500.0f, NPU_DTYPE_INT16);
    npu_write_from_float(b, 0, 2000.0f, NPU_DTYPE_INT16);
    npu_write_from_float(b, 1, -300.0f, NPU_DTYPE_INT16);
    npu_add(a, b, out, 2, NPU_DTYPE_INT16);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_INT16), 3000.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 1, NPU_DTYPE_INT16), -800.0f, 0.0f);
    return 1;
}

static int test_mul_int8(void) {
    int8_t a[2], b[2], out[2];
    npu_write_from_float(a, 0, 5.0f, NPU_DTYPE_INT8);
    npu_write_from_float(a, 1, -3.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 0, 4.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 1, 7.0f, NPU_DTYPE_INT8);
    npu_mul(a, b, out, 2, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_INT8), 20.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 1, NPU_DTYPE_INT8), -21.0f, 0.0f);
    return 1;
}

static int test_matmul_int8(void) {
    /* [1x2] * [2x1] = [1x1]: 3*4 + 5*6 = 42 */
    int8_t a[2], b[2], out[1];
    npu_write_from_float(a, 0, 3.0f, NPU_DTYPE_INT8);
    npu_write_from_float(a, 1, 5.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 0, 4.0f, NPU_DTYPE_INT8);
    npu_write_from_float(b, 1, 6.0f, NPU_DTYPE_INT8);
    npu_matmul(a, b, out, 1, 1, 2, NPU_DTYPE_INT8, NPU_FORMAT_ND);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_INT8), 42.0f, 0.0f);
    return 1;
}

int main(void) {
    printf("test_elementwise:\n");
    RUN_TEST(test_add_fp32);
    RUN_TEST(test_add_fp16);
    RUN_TEST(test_add_int8);
    RUN_TEST(test_add_int16);
    RUN_TEST(test_mul_fp32);
    RUN_TEST(test_mul_int8);
    RUN_TEST(test_mul_scalar_fp32);
    RUN_TEST(test_gelu_fp32);
    RUN_TEST(test_gelu_fp16);
    RUN_TEST(test_matmul_int8);
    TEST_SUMMARY();
}
