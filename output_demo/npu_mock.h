/* 自动生成 — 仅用于编译验证，不含实现 */
#ifndef NPU_MOCK_H
#define NPU_MOCK_H
#include <stddef.h>
#include <stdint.h>

typedef enum { NPU_DTYPE_FP16=0, NPU_DTYPE_FP32, NPU_DTYPE_BF16, NPU_DTYPE_INT8, NPU_DTYPE_INT32, NPU_DTYPE_INT16 } npu_dtype_t;
typedef enum { NPU_FORMAT_ND=0, NPU_FORMAT_NZ, NPU_FORMAT_NC1HWC0 } npu_format_t;

typedef struct { void* ptr; npu_dtype_t dtype; npu_format_t format; } npu_tensor_t;

static inline void* npu_t_ptr(npu_tensor_t t) {
    return t.ptr;
}

/* compute_ops */
void cube_matmul(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int loop, int m, int n, int k, npu_dtype_t compute_dtype);
void cube_matmul_bias(npu_tensor_t a, npu_tensor_t b, npu_tensor_t bias, npu_tensor_t out, int loop, int m, int n, int k, npu_dtype_t compute_dtype);
void vector_add(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_mul(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_mul_scalar(npu_tensor_t input, npu_tensor_t out, float scalar, int count, npu_dtype_t compute_dtype);
void vector_gelu(npu_tensor_t input, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_softmax(npu_tensor_t input, npu_tensor_t out, int dim, int count, npu_dtype_t compute_dtype);
void vector_layernorm(npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out, int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_layernorm_part1(npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out, int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_layernorm_part2(npu_tensor_t inter, npu_tensor_t orig, npu_tensor_t out, int size, npu_dtype_t compute_dtype);
void vector_softmax_part1(npu_tensor_t input, npu_tensor_t out, int dim, int count, npu_dtype_t compute_dtype);
void vector_softmax_part2(npu_tensor_t inter, npu_tensor_t out, int size, npu_dtype_t compute_dtype);
void vector_transpose(npu_tensor_t input, npu_tensor_t out, int ndim, const int* dims, int dim0, int dim1, npu_dtype_t compute_dtype);
void vector_transpose_2d(npu_tensor_t input, npu_tensor_t out, int rows, int cols, npu_dtype_t compute_dtype);
void scalar_reshape(npu_tensor_t input, npu_tensor_t out, int size, npu_dtype_t compute_dtype);
void scalar_broadcast(npu_tensor_t input, npu_tensor_t out, int size, npu_dtype_t compute_dtype);
void scalar_copy(npu_tensor_t input, npu_tensor_t out, int size, npu_dtype_t compute_dtype);
void dma_reformat(npu_tensor_t input, npu_tensor_t out, int count);

/* dma_ops */
void npu_dma_load(void* l1_dst, void* hbm_src, int size, npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_store(void* hbm_dst, void* l1_src, int size, npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_barrier(void);

/* sync_ops */
void npu_set_dependency(int src_id, int dst_id);
void npu_barrier(void);

#ifdef NPU_DEBUG_DUMP
#include <stdio.h>
#define NPU_LOG(fmt, ...) printf("[NPU] " fmt "\n", ##__VA_ARGS__)
#else
#define NPU_LOG(fmt, ...) ((void)0)
#endif

#endif
