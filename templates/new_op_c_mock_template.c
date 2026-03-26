/*
 * TODO_OP_NAME — NPU CPU mock 实现
 *
 * 这个文件实现 TODO_OP_NAME 算子的 CPU 模拟版本，
 * 用于在 CPU 上验证计算正确性。
 *
 * !!!! 常见陷阱 !!!!
 * 1. 获取 tensor 数据指针用 npu_t_ptr(t)，不要用 t.addr
 *    - .addr 是 HBM 地址（uint64_t），CPU mock 中无意义
 *    - npu_t_ptr() 返回 CPU 可访问的 void*
 *
 * 2. 读写数据用精度转换函数，不要直接 cast 指针：
 *    - 读: npu_read_compute(ptr, idx, storage_dtype, compute_dtype)
 *         → 从 storage_dtype 读出，转成 compute_dtype 精度的 float
 *    - 写: npu_write_compute(po, idx, value, storage_dtype, compute_dtype)
 *         → 先 round 到 compute_dtype，再存为 storage_dtype
 *    - 另有 npu_read_as_float / npu_write_from_float（不经过 compute_dtype）
 *
 * 3. NPU_DBG_T 宏的参数：NPU_DBG_T(变量名标签, 变量, 元素数)
 *    - 第一个参数是调试输出中显示的名字
 *    - 第二个参数是 npu_tensor_t 变量本身
 *    - 第三个参数是元素数量（用于 dump 数据）
 *
 * 4. TRACE 宏必须成对出现：BEGIN 和 END 参数一致
 */

#include "npu_api.h"
#include "npu_debug.h"
#include <math.h>

/* ── TODO: 选择合适的 trace 模式 ──────────────────────────────────────
 *
 * 一元算子（1 input + 1 output）：
 *   用 UNOP_TRACE_BEGIN / UNOP_TRACE_END 宏（见 npu_compute_elementwise.c）
 *   或者手写 NPU_DBG_T 数组（2 个元素）
 *
 * 二元算子（2 inputs + 1 output）：
 *   用 BINOP_TRACE_BEGIN / BINOP_TRACE_END 宏
 *   或者手写 NPU_DBG_T 数组（3 个元素）
 *
 * 其他（N inputs + M outputs）：
 *   手写 NPU_DBG_T 数组，元素数 = N + M
 *
 * 下面展示手写模式（最通用）。如果是简单一元/二元，
 * 可以改用上面的宏简化代码。
 */

/* ====================================================================
 * TODO_OP_NAME
 * ==================================================================== */

/* TODO: 修改函数签名，匹配 npu_api.h 中的声明
 *
 * 常见签名模式：
 *
 * 一元:  void vector_xxx(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
 *                        int count, npu_dtype_t compute_dtype);
 *
 * 二元:  void vector_xxx(TidInfo tid, npu_tensor_t a, npu_tensor_t b,
 *                        npu_tensor_t out, int count, npu_dtype_t compute_dtype);
 *
 * 带标量: void vector_xxx(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
 *                         float scalar, int count, npu_dtype_t compute_dtype);
 *
 * 矩阵:  void cube_xxx(TidInfo tid, npu_tensor_t a, npu_tensor_t b,
 *                       npu_tensor_t out, int M, int N, int K,
 *                       npu_dtype_t compute_dtype);
 */
void TODO_OP_NAME(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                  int count, npu_dtype_t compute_dtype) {

    /* ── trace 开始 ── */
    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(input, input, count),
        NPU_DBG_T(out, out, count)
        /* TODO: 如果有更多 tensor 参数，继续添加 NPU_DBG_T 行 */
    };
    NPU_TRACE_BEGIN("TODO_OP_NAME", tid, _dbg, 2);
    /* TODO: 第 4 个参数 = _dbg 数组长度，与 tensor 参数数量一致 */

    /* ── 获取数据指针（用 npu_t_ptr，不要用 .addr！）── */
    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    /* TODO: 如果有更多 tensor，继续获取指针 */

    /* ── 计算循环 ── */
    for (int i = 0; i < count; i++) {
        /* 读取输入（自动处理 dtype 转换） */
        float x = npu_read_compute(pi, i, input.dtype, compute_dtype);

        /* TODO: 替换为实际的计算逻辑 */
        float result = x;  /* TODO: 例如 silu = x / (1.0f + expf(-x)) */

        /* 写入输出（自动处理 dtype 转换） */
        npu_write_compute(po, i, result, out.dtype, compute_dtype);
    }

    /* ── trace 结束（参数必须和 BEGIN 一致）── */
    NPU_TRACE_END("TODO_OP_NAME", tid, _dbg, 2);
}

/* TODO: 如果需要添加多个相关算子，在下面继续写
 *
 * void TODO_OP_NAME_variant(...) {
 *     ...
 * }
 */
