#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stddef.h>

typedef struct {
    int shape[8];
    int ndim;
    char dtype[16];
    char format[16];
    size_t total_bytes;
} tensor_desc_t;

int parse_desc(const char* desc_path, tensor_desc_t* desc);
int load_tensor(void* hbm_base, size_t offset,
                const char* bin_path, const char* desc_path);

#endif
