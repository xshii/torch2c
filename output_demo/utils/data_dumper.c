#include "data_dumper.h"
#include <stdio.h>

int dump_tensor(const void* hbm_base, size_t offset,
                size_t size, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fwrite((const unsigned char*)hbm_base + offset, 1, size, f);
    fclose(f);
    return 0;
}
