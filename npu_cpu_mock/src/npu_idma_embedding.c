#include "npu_api.h"
#include "npu_debug.h"
#include <string.h>

/* idma_embedding: 按索引从权重表中取行 (gather) */
void idma_embedding(TidInfo tid, npu_tensor_t weight, npu_tensor_t indices,
                    npu_tensor_t out, int vocab_size, int dim) {
    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(weight, weight, vocab_size * dim),
        NPU_DBG_T(indices, indices, vocab_size),
        NPU_DBG_T(out, out, vocab_size * dim),
    };
    NPU_TRACE_BEGIN("idma_embedding", tid, _dbg, 3);

    _Float16 *w = (_Float16 *)weight.ptr;
    _Float16 *o = (_Float16 *)out.ptr;

    /* indices 可能是 int32 或 int64，这里按 int32 处理 */
    int *idx = (int *)indices.ptr;

    /* 逐索引 gather */
    int out_offset = 0;
    /* 遍历到遇到无效索引为止，最多 vocab_size 个 */
    for (int i = 0; i < vocab_size * 16; i++) {  /* 上限保护 */
        int id = idx[i];
        if (id < 0 || id >= vocab_size) break;
        memcpy(o + out_offset, w + id * dim, dim * sizeof(_Float16));
        out_offset += dim;
    }

    NPU_TRACE_END("idma_embedding", tid, _dbg, 3);
}
