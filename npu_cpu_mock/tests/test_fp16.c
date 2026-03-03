#include "test_framework.h"
#include "npu_fp16.h"
#include "npu_api.h"

static int test_fp16_roundtrip(void) {
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f, 0.00006103515625f};
    int n = (int)(sizeof(vals) / sizeof(vals[0]));
    for (int i = 0; i < n; i++) {
        uint16_t h = float_to_fp16(vals[i]);
        float    r = fp16_to_float(h);
        ASSERT_FLOAT_EQ(r, vals[i], 1e-4f);
    }
    return 1;
}

static int test_fp16_inf(void) {
    float inf = 1.0f / 0.0f;
    uint16_t h = float_to_fp16(inf);
    ASSERT_TRUE((h & 0x7FFF) == 0x7C00); /* positive inf */
    float r = fp16_to_float(h);
    ASSERT_TRUE(r > 1e30f);
    return 1;
}

static int test_fp16_zero_sign(void) {
    uint16_t pz = float_to_fp16(0.0f);
    uint16_t nz = float_to_fp16(-0.0f);
    ASSERT_FLOAT_EQ(fp16_to_float(pz), 0.0f, 0.0f);
    ASSERT_FLOAT_EQ(fp16_to_float(nz), 0.0f, 0.0f);
    ASSERT_TRUE((nz & 0x8000) != 0); /* sign bit set */
    return 1;
}

static int test_dtype_size(void) {
    ASSERT_INT_EQ((int)npu_dtype_size(NPU_DTYPE_FP16), 2);
    ASSERT_INT_EQ((int)npu_dtype_size(NPU_DTYPE_FP32), 4);
    ASSERT_INT_EQ((int)npu_dtype_size(NPU_DTYPE_BF16), 2);
    ASSERT_INT_EQ((int)npu_dtype_size(NPU_DTYPE_INT8),  1);
    ASSERT_INT_EQ((int)npu_dtype_size(NPU_DTYPE_INT32), 4);
    ASSERT_INT_EQ((int)npu_dtype_size(NPU_DTYPE_INT16), 2);
    return 1;
}

static int test_read_write_fp32(void) {
    float buf[4];
    npu_write_from_float(buf, 0, 3.14f, NPU_DTYPE_FP32);
    npu_write_from_float(buf, 1, -2.71f, NPU_DTYPE_FP32);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 0, NPU_DTYPE_FP32), 3.14f, 1e-6f);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 1, NPU_DTYPE_FP32), -2.71f, 1e-6f);
    return 1;
}

static int test_read_write_fp16(void) {
    uint16_t buf[4];
    npu_write_from_float(buf, 0, 1.5f, NPU_DTYPE_FP16);
    npu_write_from_float(buf, 1, -0.25f, NPU_DTYPE_FP16);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 0, NPU_DTYPE_FP16), 1.5f, 1e-3f);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 1, NPU_DTYPE_FP16), -0.25f, 1e-3f);
    return 1;
}

static int test_read_write_int8(void) {
    int8_t buf[4];
    npu_write_from_float(buf, 0, 42.0f, NPU_DTYPE_INT8);
    npu_write_from_float(buf, 1, -100.0f, NPU_DTYPE_INT8);
    npu_write_from_float(buf, 2, 0.0f, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 0, NPU_DTYPE_INT8), 42.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 1, NPU_DTYPE_INT8), -100.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 2, NPU_DTYPE_INT8), 0.0f, 0.0f);
    return 1;
}

static int test_read_write_int16(void) {
    int16_t buf[4];
    npu_write_from_float(buf, 0, 1000.0f, NPU_DTYPE_INT16);
    npu_write_from_float(buf, 1, -2000.0f, NPU_DTYPE_INT16);
    npu_write_from_float(buf, 2, 0.0f, NPU_DTYPE_INT16);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 0, NPU_DTYPE_INT16), 1000.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 1, NPU_DTYPE_INT16), -2000.0f, 0.0f);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 2, NPU_DTYPE_INT16), 0.0f, 0.0f);
    return 1;
}

static int test_int8_clamp(void) {
    int8_t buf[2];
    /* overflow: 200 > 127 => clamp to 127 */
    npu_write_from_float(buf, 0, 200.0f, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 0, NPU_DTYPE_INT8), 127.0f, 0.0f);
    /* underflow: -200 < -128 => clamp to -128 */
    npu_write_from_float(buf, 1, -200.0f, NPU_DTYPE_INT8);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 1, NPU_DTYPE_INT8), -128.0f, 0.0f);
    return 1;
}

static int test_int16_clamp(void) {
    int16_t buf[2];
    /* overflow: 40000 > 32767 => clamp to 32767 */
    npu_write_from_float(buf, 0, 40000.0f, NPU_DTYPE_INT16);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 0, NPU_DTYPE_INT16), 32767.0f, 0.0f);
    /* underflow: -40000 < -32768 => clamp to -32768 */
    npu_write_from_float(buf, 1, -40000.0f, NPU_DTYPE_INT16);
    ASSERT_FLOAT_EQ(npu_read_as_float(buf, 1, NPU_DTYPE_INT16), -32768.0f, 0.0f);
    return 1;
}

int main(void) {
    printf("test_fp16:\n");
    RUN_TEST(test_fp16_roundtrip);
    RUN_TEST(test_fp16_inf);
    RUN_TEST(test_fp16_zero_sign);
    RUN_TEST(test_dtype_size);
    RUN_TEST(test_read_write_fp32);
    RUN_TEST(test_read_write_fp16);
    RUN_TEST(test_read_write_int8);
    RUN_TEST(test_read_write_int16);
    RUN_TEST(test_int8_clamp);
    RUN_TEST(test_int16_clamp);
    TEST_SUMMARY();
}
