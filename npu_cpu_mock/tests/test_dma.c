#include "test_framework.h"
#include "npu_api.h"
#include <string.h>

#define OFF_IN  0
#define OFF_OUT 4096

static int test_dma_move_same_dtype(void) {
    L1_INIT();
    float* src = L1_PTR(float, OFF_IN);
    float vals[4] = {10, 20, 30, 40};
    memcpy(src, vals, sizeof(vals));
    dma_move(TID0, TENSOR(OFF_OUT, NPU_DTYPE_FP32), TENSOR(OFF_IN, NPU_DTYPE_FP32), 4);
    float* dst = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(dst[i], vals[i], 0.0f);
    return 1;
}

static int test_dma_move_dtype_convert(void) {
    L1_INIT();
    float* src = L1_PTR(float, OFF_IN);
    float vals[4] = {10, 20, 30, 40};
    memcpy(src, vals, sizeof(vals));
    #define OFF_FP16 8192
    dma_move(TID0, TENSOR(OFF_FP16, NPU_DTYPE_FP16), TENSOR(OFF_IN, NPU_DTYPE_FP32), 4);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(npu_read_as_float(L1_PTR(uint16_t, OFF_FP16), i, NPU_DTYPE_FP16), vals[i], 1e-2f);
    return 1;
}

static int test_idma_move(void) {
    L1_INIT();
    float* src = L1_PTR(float, OFF_IN);
    float vals[4] = {10, 20, 30, 40};
    memcpy(src, vals, sizeof(vals));
    idma_move(TID0, TENSOR(OFF_OUT, NPU_DTYPE_FP32), TENSOR(OFF_IN, NPU_DTYPE_FP32), 4);
    float* dst = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(dst[i], vals[i], 0.0f);
    return 1;
}

static int test_idma_reshape(void) {
    L1_INIT();
    float* in = L1_PTR(float, OFF_IN);
    float vals[6] = {1, 2, 3, 4, 5, 6};
    memcpy(in, vals, sizeof(vals));
    idma_reshape(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_OUT, NPU_DTYPE_FP32),
                 6 * (int)sizeof(float));
    float* out = L1_PTR(float, OFF_OUT);
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(out[i], vals[i], 0.0f);
    return 1;
}

#define OFF_BCAST 8192
static int test_idma_broadcast(void) {
    L1_INIT();
    float* src = L1_PTR(float, OFF_IN);
    src[0] = 42.0f;
    idma_broadcast(TID0, TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_BCAST, NPU_DTYPE_FP32), 5);
    float* out = L1_PTR(float, OFF_BCAST);
    for (int i = 0; i < 5; i++)
        ASSERT_FLOAT_EQ(out[i], 42.0f, 0.0f);
    return 1;
}

#define OFF_B_CONCAT 8192
static int test_idma_concat(void) {
    L1_INIT();
    float* a = L1_PTR(float, OFF_IN);
    float* b = L1_PTR(float, OFF_B_CONCAT);
    float va[3] = {1, 2, 3};
    float vb[2] = {4, 5};
    memcpy(a, va, sizeof(va));
    memcpy(b, vb, sizeof(vb));

    npu_tensor_t inputs[2] = {TENSOR(OFF_IN, NPU_DTYPE_FP32), TENSOR(OFF_B_CONCAT, NPU_DTYPE_FP32)};
    int counts[2] = {3, 2};
    idma_concat(TID0, inputs, counts, 2, TENSOR(OFF_OUT, NPU_DTYPE_FP32));

    float* out = L1_PTR(float, OFF_OUT);
    float expected[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++)
        ASSERT_FLOAT_EQ(out[i], expected[i], 0.0f);
    return 1;
}

int main(void) {
    printf("test_dma:\n");
    RUN_TEST(test_dma_move_same_dtype);
    RUN_TEST(test_dma_move_dtype_convert);
    RUN_TEST(test_idma_move);
    RUN_TEST(test_idma_reshape);
    RUN_TEST(test_idma_broadcast);
    RUN_TEST(test_idma_concat);
    TEST_SUMMARY();
}
