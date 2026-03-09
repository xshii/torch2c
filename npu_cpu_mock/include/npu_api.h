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

typedef enum {
    NPU_UNIT_CUBE   = 0,
    NPU_UNIT_VECTOR,
    NPU_UNIT_DMA,
    NPU_UNIT_IDMA
} npu_compute_unit_t;

/* ---- pipeline task descriptor ---- */
/* dep_*_tid = 0 means no dependency on that unit. */
typedef struct {
    int task_id;        /* current task id (software-assigned) */
    int dep_cube_tid;   /* dependency on cube unit */
    int dep_vector_tid; /* dependency on vector unit */
    int dep_dma_tid;    /* dependency on dma unit */
    int dep_idma_tid;   /* dependency on idma unit */
} TidInfo;

/* ---- constants ---- */
#define NPU_MAX_NDIM 16   /* max supported tensor dimensions */

/* ---- tensor descriptor ---- */
typedef struct {
    void*         ptr;     /* direct pointer to L1 memory */
    npu_dtype_t   dtype;
    npu_format_t  format;
} npu_tensor_t;

/* Extract pointer from tensor descriptor */
static inline void* npu_t_ptr(npu_tensor_t t) {
    return t.ptr;
}

/* ---- dtype helpers ---- */
size_t npu_dtype_size(npu_dtype_t dtype);
float  npu_read_as_float(const void* buf, int index, npu_dtype_t dtype);
void   npu_write_from_float(void* buf, int index, float value, npu_dtype_t dtype);
float  npu_round_to_dtype(float value, npu_dtype_t dtype);

/* ---- compute-precision helpers ---- */
/* Read from storage dtype, then round to compute precision */
static inline float npu_read_compute(const void* buf, int index,
                                     npu_dtype_t storage_dtype,
                                     npu_dtype_t compute_dtype) {
    float v = npu_read_as_float(buf, index, storage_dtype);
    return npu_round_to_dtype(v, compute_dtype);
}

/* Round to compute precision, then write to storage dtype */
static inline void npu_write_compute(void* buf, int index, float value,
                                     npu_dtype_t storage_dtype,
                                     npu_dtype_t compute_dtype) {
    float rounded = npu_round_to_dtype(value, compute_dtype);
    npu_write_from_float(buf, index, rounded, storage_dtype);
}

/* ---- cube compute ops ---- */
void cube_matmul(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int loop, int m, int n, int k, int transpose_b, npu_dtype_t compute_dtype);
void cube_matmul_bias(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t bias, npu_tensor_t out, int loop, int m, int n, int k, int transpose_b, npu_dtype_t compute_dtype);

/* ---- vector: arithmetic ---- */
void vector_add(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_sub(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_mul(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_div(TidInfo tid, npu_tensor_t a, npu_tensor_t b, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_mul_scalar(TidInfo tid, npu_tensor_t input, npu_tensor_t out, float scalar, int count, npu_dtype_t compute_dtype);
void vector_fill(TidInfo tid, npu_tensor_t out, float value, int count, npu_dtype_t compute_dtype);

/* ---- vector: activation & regularization ---- */
void vector_gelu(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
void vector_dropout(TidInfo tid, npu_tensor_t input, npu_tensor_t out, npu_tensor_t mask, int count, float scale, npu_dtype_t compute_dtype);

/* ---- vector: normalization ---- */
void vector_softmax(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int dim, int count, npu_dtype_t compute_dtype);
void vector_softmax_part1(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int dim, int count, npu_dtype_t compute_dtype);
void vector_softmax_part2(TidInfo tid, npu_tensor_t inter, npu_tensor_t out, int size, npu_dtype_t compute_dtype);
void vector_layernorm(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out, int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_layernorm_part1(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t beta, npu_tensor_t out, int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_layernorm_part2(TidInfo tid, npu_tensor_t inter, npu_tensor_t orig, npu_tensor_t out, int size, npu_dtype_t compute_dtype);
void vector_rmsnorm(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t out, int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_rmsnorm_part1(TidInfo tid, npu_tensor_t input, npu_tensor_t gamma, npu_tensor_t out, int hidden, int seq, float eps, npu_dtype_t compute_dtype);
void vector_rmsnorm_part2(TidInfo tid, npu_tensor_t inter, npu_tensor_t orig, npu_tensor_t out, int size, npu_dtype_t compute_dtype);

/* ---- vector: shape ---- */
void vector_transpose(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int ndim, const int* dims, int dim0, int dim1, npu_dtype_t compute_dtype);
void vector_transpose_2d(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int rows, int cols, npu_dtype_t compute_dtype);

/* ---- dma (HBM ↔ L1) ---- */
void dma_move(TidInfo tid, npu_tensor_t dst, npu_tensor_t src, int size_bytes);
void dma_reformat(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int size_bytes);

/* ---- idma (L1 → pipe) ---- */
void idma_move(TidInfo tid, npu_tensor_t dst, npu_tensor_t src, int size_bytes);
void idma_reshape(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int size);
void idma_broadcast(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int src_count, int dst_count);
void idma_concat(TidInfo tid, const npu_tensor_t* inputs, const int* counts, int num_inputs, npu_tensor_t out);

#endif /* NPU_API_H */
