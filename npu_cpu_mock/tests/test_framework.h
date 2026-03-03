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

#endif /* TEST_FRAMEWORK_H */
