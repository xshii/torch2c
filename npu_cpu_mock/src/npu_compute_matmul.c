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

/*
 * Mixed-precision matmul + bias:
 *   a, b, out  are stored in io_dtype      (e.g. INT8)
 *   bias       is stored in compute_dtype  (e.g. FP16)
 *
 * Simulates real NPU behaviour:
 *   1. Read inputs from io_dtype -> float32
 *   2. Round each operand to compute_dtype precision (INT8 -> FP16 on hardware)
 *   3. Multiply-accumulate in float32 (hardware uses wider accumulator)
 *   4. Round accumulated result to compute_dtype
 *   5. Add bias (stored in compute_dtype)
 *   6. Round final result to compute_dtype
 *   7. Write output in io_dtype (clamp if integer)
 */
void npu_matmul_bias(void* a, void* b, void* bias, void* out,
                     int M, int N, int K,
                     npu_dtype_t io_dtype, npu_dtype_t compute_dtype,
                     npu_format_t fmt) {
    (void)fmt;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float va = npu_read_as_float(a, m * K + k, io_dtype);
                float vb = npu_read_as_float(b, k * N + n, io_dtype);
                va = npu_round_to_dtype(va, compute_dtype);
                vb = npu_round_to_dtype(vb, compute_dtype);
                acc += va * vb;
            }
            /* round accumulator to compute precision */
            acc = npu_round_to_dtype(acc, compute_dtype);
            /* add bias (stored in compute_dtype) */
            float b_val = npu_read_as_float(bias, n, compute_dtype);
            acc += b_val;
            acc = npu_round_to_dtype(acc, compute_dtype);
            /* write output in io_dtype */
            npu_write_from_float(out, m * N + n, acc, io_dtype);
        }
    }
}
