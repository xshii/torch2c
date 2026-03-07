#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parse_desc(const char* desc_path, tensor_desc_t* desc) {
    FILE* f = fopen(desc_path, "r");
    if (!f) return -1;
    memset(desc, 0, sizeof(*desc));

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char key[64], val[128];
        if (sscanf(line, "%63[^:]: %127s", key, val) != 2) continue;

        if (strcmp(key, "shape") == 0) {
            desc->ndim = 0;
            char* tok = strtok(val, ",");
            while (tok && desc->ndim < 8) {
                desc->shape[desc->ndim++] = atoi(tok);
                tok = strtok(NULL, ",");
            }
        } else if (strcmp(key, "dtype") == 0) {
            strncpy(desc->dtype, val, 15);
        } else if (strcmp(key, "format") == 0) {
            strncpy(desc->format, val, 15);
        } else if (strcmp(key, "total_bytes") == 0) {
            desc->total_bytes = (size_t)atol(val);
        }
    }
    fclose(f);
    return 0;
}

int load_tensor(void* hbm_base, size_t offset,
                const char* bin_path, const char* desc_path) {
    tensor_desc_t desc;
    if (parse_desc(desc_path, &desc) != 0) return -1;

    FILE* f = fopen(bin_path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if ((size_t)file_size != desc.total_bytes) {
        fclose(f);
        return -2;
    }

    size_t read = fread((unsigned char*)hbm_base + offset, 1, desc.total_bytes, f);
    fclose(f);
    return (read == desc.total_bytes) ? 0 : -3;
}
