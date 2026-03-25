#include "npu_api.h"
#include "npu_debug.h"
#include <string.h>

/* idma_embedding: 按索引从权重表中取行 (gather) */
void idma_embedding(TidInfo tid, npu_tensor_t weight, npu_tensor_t indices,
                    npu_tensor_t out, int vocab_size, int dim) {
    NPU_DEBUG_TRACE_T *_dbg = NULL;
    NPU_TRACE_BEGIN("idma_embedding", tid, _dbg, dim);

    _Float16 *w = (_Float16 *)weight.addr;
    _Float16 *o = (_Float16 *)out.addr;

    /* indices 可能是 int32 或 int64，这里按 int32 处理 */
    int *idx = (int *)indices.addr;
    int num_indices = 1;
    /* 简单计算：out 总元素 / dim = 索引个数 */
    _Float16 *out_end = o;
    (void)out_end;

    /* 逐索引 gather */
    int out_offset = 0;
    /* 假设 indices 是连续的 int32 数组，长度 = out_size / dim */
    int total_out_elems = 0;
    /* 用 out 的 hbm_size 推算，但 mock 里没有，用 vocab_size 兜底 */
    /* 简单实现：遍历到遇到无效索引为止，最多 vocab_size 个 */
    for (int i = 0; i < vocab_size * 16; i++) {  /* 上限保护 */
        int id = idx[i];
        if (id < 0 || id >= vocab_size) break;
        memcpy(o + out_offset, w + id * dim, dim * sizeof(_Float16));
        out_offset += dim;
    }

    NPU_TRACE_END("idma_embedding", tid, _dbg, dim);
}
