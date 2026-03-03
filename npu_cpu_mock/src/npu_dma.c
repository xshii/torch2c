#include "npu_api.h"
#include <string.h>

void npu_dma_load(void* l1_dst, void* hbm_src, int size,
                  npu_format_t src_fmt, npu_format_t dst_fmt) {
    (void)src_fmt; (void)dst_fmt;
    memcpy(l1_dst, hbm_src, (size_t)size);
}

void npu_dma_store(void* hbm_dst, void* l1_src, int size,
                   npu_format_t src_fmt, npu_format_t dst_fmt) {
    (void)src_fmt; (void)dst_fmt;
    memcpy(hbm_dst, l1_src, (size_t)size);
}

void npu_dma_barrier(void) {
    /* no-op on CPU */
}
