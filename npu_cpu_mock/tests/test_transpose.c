#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

static int test_transpose_2d_fp32(void) {
    /* [2x3] -> [3x2] */
    float in[6]  = {1, 2, 3, 4, 5, 6};
    float out[6] = {0};
    npu_transpose_2d(in, out, 2, 3, NPU_DTYPE_FP32);
    /* expected: col-major read: [1,4], [2,5], [3,6] */
    float expected[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(out[i], expected[i], 1e-6f);
    return 1;
}

static int test_transpose_2d_fp16(void) {
    uint16_t in[6], out[6];
    float vals[6] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
        npu_write_from_float(in, i, vals[i], NPU_DTYPE_FP16);
    npu_transpose_2d(in, out, 2, 3, NPU_DTYPE_FP16);
    float expected[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(npu_read_as_float(out, i, NPU_DTYPE_FP16), expected[i], 1e-3f);
    return 1;
}

static int test_transpose_nd_3d(void) {
    /* [2,3,4] swap dim 0 and 2 => [4,3,2] */
    int dims[3] = {2, 3, 4};
    int total = 24;
    float in[24], out[24];
    for (int i = 0; i < total; i++) in[i] = (float)i;
    npu_transpose(in, out, 3, dims, 0, 2, NPU_DTYPE_FP32);

    /* verify: in[a][b][c] = out[c][b][a] */
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
    int dims[3] = {2, 3, 4};
    int total = 24;
    float in[24], out[24];
    for (int i = 0; i < total; i++) in[i] = (float)(i + 1);
    npu_transpose(in, out, 3, dims, 1, 2, NPU_DTYPE_FP32);

    /* verify: in[a][b][c] = out[a][c][b] */
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
    float in[6] = {1, 2, 3, 4, 5, 6};
    float out[6] = {0};
    npu_reshape(in, out, 6 * (int)sizeof(float), NPU_DTYPE_FP32);
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(out[i], in[i], 0.0f);
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

    /* barrier and sync are no-ops, just verify they don't crash */
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
