#include "npu_api.h"

void cube_matmul(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                 int loop, int m, int n, int k, npu_dtype_t compute_dtype) {
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = a.dtype;
    (void)compute_dtype;
    int mat_a = m * k;
    int mat_b = k * n;
    int mat_o = m * n;
    for (int l = 0; l < loop; l++) {
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n; ni++) {
                float acc = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    float va = npu_read_as_float(pa, l * mat_a + mi * k + ki, dt);
                    float vb = npu_read_as_float(pb, l * mat_b + ki * n + ni, dt);
                    acc += va * vb;
                }
                npu_write_from_float(po, l * mat_o + mi * n + ni, acc, out.dtype);
            }
        }
    }
}

void cube_matmul_bias(npu_tensor_t a, npu_tensor_t b, npu_tensor_t bias, npu_tensor_t out,
                      int loop, int m, int n, int k, npu_dtype_t compute_dtype) {
    void* pa = npu_t_ptr(a);
    void* pb = npu_t_ptr(b);
    void* pbias = npu_t_ptr(bias);
    void* po = npu_t_ptr(out);
    npu_dtype_t io_dt = a.dtype;
    int mat_a = m * k;
    int mat_b = k * n;
    int mat_o = m * n;
    for (int l = 0; l < loop; l++) {
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n; ni++) {
                float acc = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    float va = npu_read_as_float(pa, l * mat_a + mi * k + ki, io_dt);
                    float vb = npu_read_as_float(pb, l * mat_b + ki * n + ni, io_dt);
                    va = npu_round_to_dtype(va, compute_dtype);
                    vb = npu_round_to_dtype(vb, compute_dtype);
                    acc += va * vb;
                }
                acc = npu_round_to_dtype(acc, compute_dtype);
                float b_val = npu_read_as_float(pbias, ni, bias.dtype);
                acc += b_val;
                acc = npu_round_to_dtype(acc, compute_dtype);
                npu_write_from_float(po, l * mat_o + mi * n + ni, acc, out.dtype);
            }
        }
    }
}
