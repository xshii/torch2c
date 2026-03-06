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

void dma_reformat(npu_tensor_t input, npu_tensor_t out, int count) {
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    /* CPU mock: element-wise copy (format conversion is transparent) */
    for (int i = 0; i < count; i++) {
        float v = npu_read_as_float(pi, i, input.dtype);
        npu_write_from_float(po, i, v, out.dtype);
    }
}
