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

/* ---- tensor descriptor ---- */
#ifndef NPU_ADDR_SHIFT
#define NPU_ADDR_SHIFT 5  /* log2(32): L1 32-byte alignment */
#endif

typedef struct {
    uint32_t      addr;    /* L1 address >> NPU_ADDR_SHIFT */
    npu_dtype_t   dtype;
    npu_format_t  format;
} npu_tensor_t;

/* Global L1 base pointer — set by model_run before first op call */
extern unsigned char* npu_l1_base;

/* Reconstruct actual memory pointer from tensor descriptor */
static inline void* npu_t_ptr(npu_tensor_t t) {
    return (void*)(npu_l1_base + ((size_t)t.addr << NPU_ADDR_SHIFT));
}

/* ---- dtype helpers ---- */
size_t npu_dtype_size(npu_dtype_t dtype);
float  npu_read_as_float(const void* buf, int index, npu_dtype_t dtype);
void   npu_write_from_float(void* buf, int index, float value, npu_dtype_t dtype);
float  npu_round_to_dtype(float value, npu_dtype_t dtype);

/* ---- cube compute ops ---- */
void cube_matmul(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                 int loop, int m, int n, int k, npu_dtype_t compute_dtype);

void cube_matmul_bias(npu_tensor_t a, npu_tensor_t b, npu_tensor_t bias, npu_tensor_t out,
                      int loop, int m, int n, int k, npu_dtype_t compute_dtype);

/* ---- vector compute ops ---- */
void vector_add(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype);
void vector_mul(npu_tensor_t a, npu_tensor_t b, npu_tensor_t out,
                int count, npu_dtype_t compute_dtype);
void vector_mul_scalar(npu_tensor_t input, npu_tensor_t out,
                       float scalar, int count, npu_dtype_t compute_dtype);
void vector_gelu(npu_tensor_t input, npu_tensor_t out,
                 int count, npu_dtype_t compute_dtype);

void vector_layernorm(npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out,
                      int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_layernorm_part1(npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out,
                            int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_layernorm_part2(npu_tensor_t inter, npu_tensor_t orig, npu_tensor_t out,
                            int size, npu_dtype_t compute_dtype);

void vector_softmax(npu_tensor_t input, npu_tensor_t out,
                    int dim, int count, npu_dtype_t compute_dtype);
void vector_softmax_part1(npu_tensor_t input, npu_tensor_t out,
                          int dim, int count, npu_dtype_t compute_dtype);
void vector_softmax_part2(npu_tensor_t inter, npu_tensor_t out,
                          int size, npu_dtype_t compute_dtype);

void vector_transpose(npu_tensor_t input, npu_tensor_t out,
                      int ndim, const int* dims, int dim0, int dim1, npu_dtype_t compute_dtype);
void vector_transpose_2d(npu_tensor_t input, npu_tensor_t out,
                         int rows, int cols, npu_dtype_t compute_dtype);

/* ---- scalar compute ops ---- */
void scalar_reshape(npu_tensor_t input, npu_tensor_t out,
                    int size, npu_dtype_t compute_dtype);
void scalar_broadcast(npu_tensor_t input, npu_tensor_t out,
                      int size, npu_dtype_t compute_dtype);
void scalar_copy(npu_tensor_t input, npu_tensor_t out,
                 int size, npu_dtype_t compute_dtype);

/* ---- DMA ops ---- */
void npu_dma_load(void* l1_dst, void* hbm_src, int size,
                  npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_store(void* hbm_dst, void* l1_src, int size,
                   npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_barrier(void);

/* ---- sync ops ---- */
void npu_set_dependency(int src_id, int dst_id);
void npu_barrier(void);

#endif /* NPU_API_H */
