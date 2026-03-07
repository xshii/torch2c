#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <stddef.h>
#include <string.h>

static inline size_t dtype_size(const char* dtype) {
    if (strncmp(dtype, "fp32", 4) == 0 || strncmp(dtype, "int32", 5) == 0) return 4;
    return 2; /* fp16, bf16, int8, int16 default */
}

static inline size_t elem_count(const int* shape, int ndim) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
    return n;
}

#endif
