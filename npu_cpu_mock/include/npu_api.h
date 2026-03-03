#ifndef NPU_API_H
#define NPU_API_H

#include <stdint.h>
#include <stddef.h>

/* ---- enums ---- */
typedef enum {
    NPU_DTYPE_FP16  = 0,
    NPU_DTYPE_FP32,
    NPU_DTYPE_BF16,
    NPU_DTYPE_INT8,
    NPU_DTYPE_INT32,
    NPU_DTYPE_INT16
} npu_dtype_t;

typedef enum {
    NPU_FORMAT_ND = 0,
    NPU_FORMAT_NZ,
    NPU_FORMAT_NC1HWC0
} npu_format_t;

/* ---- dtype helpers ---- */
size_t npu_dtype_size(npu_dtype_t dtype);
float  npu_read_as_float(const void* buf, int index, npu_dtype_t dtype);
void   npu_write_from_float(void* buf, int index, float value, npu_dtype_t dtype);

/* ---- compute ops (12) ---- */
void npu_matmul(void* a, void* b, void* out,
                int M, int N, int K, npu_dtype_t dtype, npu_format_t fmt);

void npu_add(void* a, void* b, void* out, int count, npu_dtype_t dtype);
void npu_mul(void* a, void* b, void* out, int count, npu_dtype_t dtype);
void npu_mul_scalar(void* input, void* out, float scalar, int count, npu_dtype_t dtype);
void npu_gelu(void* input, void* out, int count, npu_dtype_t dtype);

void npu_layernorm_part1(void* input, void* gamma, void* beta, void* out,
                         int hidden, int seq, float eps, npu_dtype_t dtype);
void npu_layernorm_part2(void* inter, void* orig, void* out, int hidden, npu_dtype_t dtype);

void npu_softmax_part1(void* input, void* out, int dim, int count, npu_dtype_t dtype);
void npu_softmax_part2(void* inter, void* out, int count, npu_dtype_t dtype);

void npu_transpose(void* input, void* out, int ndim, const int* dims,
                   int dim0, int dim1, npu_dtype_t dtype);
void npu_transpose_2d(void* input, void* out, int rows, int cols, npu_dtype_t dtype);
void npu_reshape(void* input, void* out, int count);

/* ---- DMA ops (3) ---- */
void npu_dma_load(void* l1_dst, void* hbm_src, int size,
                  npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_store(void* hbm_dst, void* l1_src, int size,
                   npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_barrier(void);

/* ---- sync ops (2) ---- */
void npu_set_dependency(int src_id, int dst_id);
void npu_barrier(void);

#endif /* NPU_API_H */
