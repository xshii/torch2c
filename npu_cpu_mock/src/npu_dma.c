#include "npu_api.h"
#include "npu_debug.h"
#include <string.h>
#include <stdio.h>

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
void dma_move(TidInfo tid, npu_tensor_t dst, npu_tensor_t src, int size_bytes) {
    int count = size_bytes / (int)npu_dtype_size(src.dtype);
    npu_debug_tensor_arg_t args[] = {
        NPU_DBG_T(src, src, count), NPU_DBG_T(dst, dst, count)
    };
    NPU_TRACE_BEGIN("dma_move", tid, args, 2);

    if (dst.dtype == src.dtype) {
        memcpy(npu_t_ptr(dst), npu_t_ptr(src), (size_t)size_bytes);
    } else {
        tensor_move(dst, src, count);
    }

    args[1].count = count;  /* dst 元素数 = src 元素数 */
    NPU_TRACE_END("dma_move", tid, args, 2);
}

void dma_reformat(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int size_bytes) {
    int count = size_bytes / (int)npu_dtype_size(input.dtype);
    npu_debug_tensor_arg_t args[] = {
        NPU_DBG_T(input, input, count), NPU_DBG_T(out, out, count)
    };
    NPU_TRACE_BEGIN("dma_reformat", tid, args, 2);

    if (input.dtype == out.dtype) {
        memcpy(npu_t_ptr(out), npu_t_ptr(input), (size_t)size_bytes);
    } else {
        tensor_move(out, input, count);
    }

    NPU_TRACE_END("dma_reformat", tid, args, 2);
}

/* ---- iDMA: L1 → pipe ---- */
void idma_move(TidInfo tid, npu_tensor_t dst, npu_tensor_t src, int size_bytes) {
    int count = size_bytes / (int)npu_dtype_size(src.dtype);
    npu_debug_tensor_arg_t args[] = {
        NPU_DBG_T(src, src, count), NPU_DBG_T(dst, dst, count)
    };
    NPU_TRACE_BEGIN("idma_move", tid, args, 2);

    if (dst.dtype == src.dtype) {
        memcpy(npu_t_ptr(dst), npu_t_ptr(src), (size_t)size_bytes);
    } else {
        tensor_move(dst, src, count);
    }

    NPU_TRACE_END("idma_move", tid, args, 2);
}

void idma_reshape(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int size) {
    int count = size / (int)npu_dtype_size(input.dtype);
    npu_debug_tensor_arg_t args[] = {
        NPU_DBG_T(input, input, count), NPU_DBG_T(out, out, count)
    };
    NPU_TRACE_BEGIN("idma_reshape", tid, args, 2);

    if (input.dtype == out.dtype) {
        memcpy(npu_t_ptr(out), npu_t_ptr(input), (size_t)size);
    } else {
        tensor_move(out, input, count);
    }

    NPU_TRACE_END("idma_reshape", tid, args, 2);
}

void idma_broadcast(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int src_count, int dst_count) {
    npu_debug_tensor_arg_t args[] = {
        NPU_DBG_T(input, input, src_count), NPU_DBG_T(out, out, dst_count)
    };
    NPU_TRACE_BEGIN("idma_broadcast", tid, args, 2);

    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    if (src_count == 1) {
        /* 标量广播：读 1 个元素填充整个输出 */
        float val = npu_read_as_float(pi, 0, input.dtype);
        for (int i = 0; i < dst_count; i++)
            npu_write_from_float(po, i, val, out.dtype);
    } else {
        /* tile 广播：将 src_count 个元素重复铺满 dst_count */
        for (int i = 0; i < dst_count; i++) {
            float val = npu_read_as_float(pi, i % src_count, input.dtype);
            npu_write_from_float(po, i, val, out.dtype);
        }
    }

    NPU_TRACE_END("idma_broadcast", tid, args, 2);
}

void idma_concat(TidInfo tid, const npu_tensor_t* inputs, const int* counts,
                 int num_inputs, npu_tensor_t out) {
    /* concat 比较特殊，trace 只记录 output */
    int total = 0;
    for (int n = 0; n < num_inputs; n++) total += counts[n];
    npu_debug_tensor_arg_t args[] = { NPU_DBG_T(out, out, total) };
    NPU_TRACE_BEGIN("idma_concat", tid, args, 1);

    char* po = (char*)npu_t_ptr(out);
    size_t byte_offset = 0;
    for (int n = 0; n < num_inputs; n++) {
        raw_move(po + byte_offset, out.dtype,
                 npu_t_ptr(inputs[n]), inputs[n].dtype, counts[n]);
        byte_offset += (size_t)counts[n] * npu_dtype_size(out.dtype);
    }

    NPU_TRACE_END("idma_concat", tid, args, 1);
}

void idma_slice(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                int dim, int start, int size) {
    /* Simplified: assume contiguous slice on last dim or memcpy for now.
       For dim=-1 or dim=last: copy 'size' elements starting from 'start'. */
    int out_count = size;
    npu_debug_tensor_arg_t args[] = {
        NPU_DBG_T(input, input, out_count), NPU_DBG_T(out, out, out_count)
    };
    NPU_TRACE_BEGIN("idma_slice", tid, args, 2);

    size_t elem_sz = npu_dtype_size(input.dtype);
    const char* src = (const char*)npu_t_ptr(input) + (size_t)start * elem_sz;
    memcpy(npu_t_ptr(out), src, (size_t)size * elem_sz);

    NPU_TRACE_END("idma_slice", tid, args, 2);
}
