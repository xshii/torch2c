#ifndef NPU_DEBUG_H
#define NPU_DEBUG_H

/*
 * NPU CPU Mock 维测系统
 *
 * 编译时控制:
 *   -DNPU_DEBUG_LEVEL=0  关闭 (默认)
 *   -DNPU_DEBUG_LEVEL=1  仅地址/dtype 信息
 *   -DNPU_DEBUG_LEVEL=2  地址 + min/max/nan/inf 统计
 *
 * 运行时控制:
 *   npu_debug_enable(level)  动态开关
 *
 * 日志集中收集到环形缓冲区, 调用 npu_debug_flush() 统一输出。
 * 默认在程序退出时自动 flush (通过 atexit)。
 */

#include "npu_api.h"

#ifndef NPU_DEBUG_LEVEL
#define NPU_DEBUG_LEVEL 0
#endif

/* ---- 运行时控制 ---- */
void npu_debug_enable(int level);
int  npu_debug_level(void);
void npu_debug_flush(void);

/* ---- 日志记录 ---- */

/* 记录一个 tensor 参数的维测信息 */
typedef struct {
    const char*  name;       /* 参数名, e.g. "input", "out" */
    const void*  ptr;        /* 数据指针 */
    npu_dtype_t  dtype;
    int          count;      /* 元素数 */
} npu_debug_tensor_arg_t;

/*
 * 记录一次算子调用:
 *   op_name  — 算子名
 *   tid      — TidInfo
 *   tensors  — tensor 参数数组
 *   n_tensors — tensor 参数个数
 */
void npu_debug_trace_op(const char* op_name, TidInfo tid,
                        const npu_debug_tensor_arg_t* tensors,
                        int n_tensors);

/* ---- 便捷宏 ---- */

#define NPU_DBG_T(var_name, tensor, elem_count) \
    { #var_name, npu_t_ptr(tensor), (tensor).dtype, (elem_count) }

/*
 * NPU_TRACE_BEGIN(op_name, tid, ...tensor_args)
 * NPU_TRACE_END(op_name, tid, ...tensor_args)
 *
 * 在算子入口记录输入状态, 出口记录输出状态。
 * 当 NPU_DEBUG_LEVEL==0 且未 runtime enable 时, 编译为空。
 */
#if NPU_DEBUG_LEVEL > 0

#define NPU_TRACE_BEGIN(op, tid, arr, n) \
    npu_debug_trace_op(op " [BEGIN]", tid, arr, n)

#define NPU_TRACE_END(op, tid, arr, n) \
    npu_debug_trace_op(op " [END]", tid, arr, n)

#else

#define NPU_TRACE_BEGIN(op, tid, arr, n) \
    do { if (npu_debug_level() > 0) npu_debug_trace_op(op " [BEGIN]", tid, arr, n); } while(0)

#define NPU_TRACE_END(op, tid, arr, n) \
    do { if (npu_debug_level() > 0) npu_debug_trace_op(op " [END]", tid, arr, n); } while(0)

#endif

#endif /* NPU_DEBUG_H */
