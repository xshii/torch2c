#include "npu_api.h"

void npu_matmul(void* a, void* b, void* out,
                int M, int N, int K, npu_dtype_t dtype, npu_format_t fmt) {
    (void)fmt; /* CPU mock ignores format, treats as ND */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float va = npu_read_as_float(a, m * K + k, dtype);
                float vb = npu_read_as_float(b, k * N + n, dtype);
                acc += va * vb;
            }
            npu_write_from_float(out, m * N + n, acc, dtype);
        }
    }
}
