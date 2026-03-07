#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define RUN_TEST(fn) do { \
    printf("  %-40s", #fn); \
    g_tests_run++; \
    int _ok = fn(); \
    if (_ok) { g_tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

#define TEST_SUMMARY() do { \
    printf("\n%d/%d tests passed\n", g_tests_passed, g_tests_run); \
    return (g_tests_passed == g_tests_run) ? 0 : 1; \
} while(0)

#define ASSERT_TRUE(cond) do { if (!(cond)) { \
    printf("  ASSERT_TRUE failed: %s (line %d)\n", #cond, __LINE__); return 0; } } while(0)

#define ASSERT_FLOAT_EQ(a, b, tol) do { \
    float _a=(a), _b=(b); \
    if (fabsf(_a - _b) > (tol)) { \
        printf("  ASSERT_FLOAT_EQ failed: %.6f vs %.6f (tol %.6f, line %d)\n", \
               (double)_a, (double)_b, (double)(tol), __LINE__); return 0; } } while(0)

#define ASSERT_INT_EQ(a, b) do { \
    int _a=(a), _b=(b); \
    if (_a != _b) { \
        printf("  ASSERT_INT_EQ failed: %d vs %d (line %d)\n", _a, _b, __LINE__); return 0; } } while(0)

/* ---- L1 buffer helpers for npu_tensor_t tests ---- */
#define L1_BUF_SIZE 65536
static unsigned char g_l1_buf[L1_BUF_SIZE];

/* Initialize L1 buffer — call at start of each test */
#define L1_INIT() do { \
    extern unsigned char* npu_l1_base; \
    npu_l1_base = g_l1_buf; \
    memset(g_l1_buf, 0, L1_BUF_SIZE); \
} while(0)

/* Create a tensor descriptor pointing into g_l1_buf (byte offset → shifted addr) */
#define TENSOR(offset, dt) ((npu_tensor_t){((uint32_t)(offset) >> NPU_ADDR_SHIFT), (dt), NPU_FORMAT_ND})

/* Get a typed pointer into L1 buffer at byte offset */
#define L1_PTR(type, offset) ((type*)(g_l1_buf + (offset)))

/* Default TidInfo for tests (no scheduling, no dependencies) */
#define TID0 ((TidInfo){0, 0, 0, 0, 0})

#endif /* TEST_FRAMEWORK_H */
