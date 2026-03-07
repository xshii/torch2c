#ifndef DATA_DUMPER_H
#define DATA_DUMPER_H

#include <stddef.h>

int dump_tensor(const void* hbm_base, size_t offset,
                size_t size, const char* path);

#endif
