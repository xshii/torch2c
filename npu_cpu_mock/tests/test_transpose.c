#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

#define OFF_IN  0
#define OFF_OUT 4096

static int test_transpose_2d_fp32(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[6] = {1, 2, 3, 4, 5, 6};
    memcpy(in, vals, sizeof(vals));
    vector_transpose_2d(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                        2, 3, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    float expected[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(out[i], expected[i], 1e-6f);
    return 1;
}

static int test_transpose_2d_fp16(void) {
    L1_INIT();
    float vals[6] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(L1_PTR(uint16_t, OFF_IN), i, vals[i], NPU_DTYPE_FP16);
    vector_transpose_2d(TENSOR(OFF_IN, NPU_DTYPE_FP16), TENSOR(OFF_OUT, NPU_DTYPE_FP16),
                        2, 3, NPU_DTYPE_FP16);
    float expected[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_OUT), i, NPU_DTYPE_FP16), expected[i], 1e-3f);
    return 1;
}

static int test_transpose_nd_3d(void) {
    /* [2,3,4] swap dim 0 and 2 => [4,3,2] */
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    int dims[3] = {2, 3, 4};
    int total = 24;
    for (int i = 0; i < total; i++) in[i] = (float)i;
    vector_transpose(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                     3, dims, 0, 2, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 3; b++)
            for (int c = 0; c < 4; c++) {
                float v_in  = in[a * 12 + b * 4 + c];
                float v_out = out[c * 6 + b * 2 + a];
                ASSERT_FLOAT_EQ(v_in, v_out, 1e-6f);
            }
    return 1;
}

static int test_transpose_nd_swap_adjacent(void) {
    /* [2,3,4] swap dim 1 and 2 => [2,4,3] */
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    int dims[3] = {2, 3, 4};
    int total = 24;
    for (int i = 0; i < total; i++) in[i] = (float)(i + 1);
    vector_transpose(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                     3, dims, 1, 2, NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 3; b++)
            for (int c = 0; c < 4; c++) {
                float v_in  = in[a * 12 + b * 4 + c];
                float v_out = out[a * 12 + c * 3 + b];
                ASSERT_FLOAT_EQ(v_in, v_out, 1e-6f);
            }
    return 1;
}

static int test_reshape(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[6] = {1, 2, 3, 4, 5, 6};
    memcpy(in, vals, sizeof(vals));
    scalar_reshape(TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                   6 * (int)sizeof(float), NPU_DTYPE_FP32);
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(out[i], vals[i], 0.0f);
    return 1;
}

static int test_dma_ops(void) {
    float src[4] = {10, 20, 30, 40};
    float dst[4] = {0};
    npu_dma_load(dst, src, (int)sizeof(src), NPU_FORMAT_ND, NPU_FORMAT_ND);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(dst[i], src[i], 0.0f);

    float dst2[4] = {0};
    npu_dma_store(dst2, dst, (int)sizeof(dst), NPU_FORMAT_ND, NPU_FORMAT_ND);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(dst2[i], src[i], 0.0f);

    npu_dma_barrier();
    npu_set_dependency(0, 1);
    npu_barrier();
    return 1;
}

int main(void) {
    printf("test_transpose:\n");
    RUN_TEST(test_transpose_2d_fp32);
    RUN_TEST(test_transpose_2d_fp16);
    RUN_TEST(test_transpose_nd_3d);
    RUN_TEST(test_transpose_nd_swap_adjacent);
    RUN_TEST(test_reshape);
    RUN_TEST(test_dma_ops);
    TEST_SUMMARY();
}
