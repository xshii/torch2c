#include "npu_api.h"
#include <string.h>

static void raw_move(void* pd, npu_dtype_t dst_dt, const void* ps, npu_dtype_t src_dt, int count) {
    if (src_dt == dst_dt) {
        memcpy(pd, ps, (size_t)count * npu_dtype_size(src_dt));
    } else {
        for (int i = 0; i < count; i++) {
            float v = npu_read_as_float(ps, i, src_dt);
            npu_write_from_float(pd, i, v, dst_dt);
        }
    }
}

static void tensor_move(npu_tensor_t dst, npu_tensor_t src, int count) {
    raw_move(npu_t_ptr(dst), dst.dtype, npu_t_ptr(src), src.dtype, count);
}

/* ---- DMA: HBM ↔ L1 ---- */
void dma_move(TidInfo tid, npu_tensor_t dst, npu_tensor_t src, int count) {
    (void)tid;
    tensor_move(dst, src, count);
}

/* ---- iDMA: L1 → pipe ---- */
void idma_move(TidInfo tid, npu_tensor_t dst, npu_tensor_t src, int count) {
    (void)tid;
    tensor_move(dst, src, count);
}

void idma_reshape(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int size) {
    (void)tid;
    memcpy(npu_t_ptr(out), npu_t_ptr(input), (size_t)size);
}

void idma_broadcast(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int count) {
    (void)tid;
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    float val = npu_read_as_float(pi, 0, input.dtype);
    for (int i = 0; i < count; i++)
        npu_write_from_float(po, i, val, out.dtype);
}

void idma_concat(TidInfo tid, const npu_tensor_t* inputs, const int* counts,
                 int num_inputs, npu_tensor_t out) {
    (void)tid;
    char* po = (char*)npu_t_ptr(out);
    size_t byte_offset = 0;
    for (int n = 0; n < num_inputs; n++) {
        raw_move(po + byte_offset, out.dtype,
                 npu_t_ptr(inputs[n]), inputs[n].dtype, counts[n]);
        byte_offset += (size_t)counts[n] * npu_dtype_size(out.dtype);
    }
}
