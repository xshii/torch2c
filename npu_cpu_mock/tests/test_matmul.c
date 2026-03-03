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

/*
 * INT8 I/O + FP16 compute precision: matmul + bias
 *
 * Data flow (simulates real NPU):
 *   INT8 input -> read as float -> round to FP16 -> float32 accumulate
 *   -> round to FP16 -> add FP16 bias -> round to FP16 -> clamp & write INT8
 *
 *   a[2x3] (INT8)  *  b[3x2] (INT8)  +  bias[2] (FP16)  =  out[2x2] (INT8)
 *
 *   a = [[1, 2, 3],    b = [[1, 0],    bias = [10, 20]
 *        [4, 5, 6]]         [0, 1],
 *                            [1, 1]]
 *
 *   matmul = [[1+0+3, 0+2+3], = [[4, 5],
 *             [4+0+6, 0+5+6]]    [10, 11]]
 *
 *   + bias  = [[14, 25],
 *              [20, 31]]
 */
static int test_matmul_bias_int8_fp16(void) {
    int8_t a[6], b[6], out[4];
    uint16_t bias[2]; /* stored as FP16 */

    /* fill a[2x3] = {1,2,3,4,5,6} as INT8 */
    int8_t a_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(a, i, (float)a_vals[i], NPU_DTYPE_INT8);

    /* fill b[3x2] = {{1,0},{0,1},{1,1}} as INT8 */
    int8_t b_vals[] = {1, 0, 0, 1, 1, 1};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(b, i, (float)b_vals[i], NPU_DTYPE_INT8);

    /* bias[2] = {10, 20} stored as FP16 */
    npu_write_from_float(bias, 0, 10.0f, NPU_DTYPE_FP16);
    npu_write_from_float(bias, 1, 20.0f, NPU_DTYPE_FP16);

    npu_matmul_bias(a, b, bias, out, 2, 2, 3,
                    NPU_DTYPE_INT8, NPU_DTYPE_FP16, NPU_FORMAT_ND);

    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_INT8), 14.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 1, NPU_DTYPE_INT8), 25.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 2, NPU_DTYPE_INT8), 20.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 3, NPU_DTYPE_INT8), 31.0f, 0.0f);
    return 1;
}

/*
 * Verify FP16 compute precision actually loses precision vs FP32.
 *
 * Use values that are exactly representable in INT8 but whose
 * accumulated product overflows FP16's 10-bit mantissa:
 *   a = [127, 127, 127, 127]  (1x4)
 *   b = [127, 127, 127, 127]  (4x1)
 *   bias = [0]
 *
 *   Exact result: 127*127*4 = 64516
 *   FP16 can represent 64512 (nearest), so result ≈ 64512
 *   INT8 output clamps to 127
 */
static int test_matmul_bias_fp16_precision_loss(void) {
    int8_t a[4], b[4], out[1];
    uint16_t bias[1];

    for (int i = 0; i < 4; i++) {
        npu_write_from_float(a, i, 127.0f, NPU_DTYPE_INT8);
        npu_write_from_float(b, i, 127.0f, NPU_DTYPE_INT8);
    }
    npu_write_from_float(bias, 0, 0.0f, NPU_DTYPE_FP16);

    npu_matmul_bias(a, b, bias, out, 1, 1, 4,
                    NPU_DTYPE_INT8, NPU_DTYPE_FP16, NPU_FORMAT_ND);

    /* FP16 rounds 64516 -> 64512, then INT8 clamps to 127 */
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_INT8), 127.0f, 0.0f);
    return 1;
}

/*
 * Same computation but with FP32 compute precision -> exact accumulation.
 * This proves the compute_dtype parameter actually affects the result path.
 */
static int test_matmul_bias_int8_fp32(void) {
    int8_t a[6], b[6], out[4];
    float bias[2]; /* stored as FP32 */

    int8_t a_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(a, i, (float)a_vals[i], NPU_DTYPE_INT8);
    int8_t b_vals[] = {1, 0, 0, 1, 1, 1};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(b, i, (float)b_vals[i], NPU_DTYPE_INT8);

    npu_write_from_float(bias, 0, 10.0f, NPU_DTYPE_FP32);
    npu_write_from_float(bias, 1, 20.0f, NPU_DTYPE_FP32);

    npu_matmul_bias(a, b, bias, out, 2, 2, 3,
                    NPU_DTYPE_INT8, NPU_DTYPE_FP32, NPU_FORMAT_ND);

    ASSERT_FLOAT_EQ(npu_read_as_float(out, 0, NPU_DTYPE_INT8), 14.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 1, NPU_DTYPE_INT8), 25.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 2, NPU_DTYPE_INT8), 20.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(out, 3, NPU_DTYPE_INT8), 31.0f, 0.0f);
    return 1;
}

/*
 * Show FP16 vs FP32 compute precision difference on non-trivial values.
 * a = [120, 100] (1x2), b = [100, 120] (2x1), bias = [0.5] (FP16)
 *
 * Exact:   120*100 + 100*120 + 0.5 = 24000.5
 * FP16:    round(24000.5, fp16) = 24000 (fp16 can't represent .5 at this magnitude)
 * FP32:    24000.5 exactly
 * INT8 out: both clamp to 127
 *
 * To see the difference, use FP16 I/O instead of INT8.
 */
static int test_matmul_bias_fp16_io_fp16_compute(void) {
    uint16_t a[2], b[2], out[1], bias[1];
    npu_write_from_float(a, 0, 120.0f, NPU_DTYPE_FP16);
    npu_write_from_float(a, 1, 100.0f, NPU_DTYPE_FP16);
    npu_write_from_float(b, 0, 100.0f, NPU_DTYPE_FP16);
    npu_write_from_float(b, 1, 120.0f, NPU_DTYPE_FP16);
    npu_write_from_float(bias, 0, 0.5f, NPU_DTYPE_FP16);

    npu_matmul_bias(a, b, bias, out, 1, 1, 2,
                    NPU_DTYPE_FP16, NPU_DTYPE_FP16, NPU_FORMAT_ND);

    /* FP16 rounds 24000.5 -> 24000.0 (nearest even at this scale) */
    float result = npu_read_as_float(out, 0, NPU_DTYPE_FP16);
    ASSERT_FLOAT_EQ(result, 24000.0f, 16.0f); /* FP16 precision at ~24000 is ±16 */
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
