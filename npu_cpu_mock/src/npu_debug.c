#include "npu_debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

/* ---- 环形日志缓冲区 ---- */

#define LOG_BUF_LINES  4096
#define LOG_LINE_LEN   256

static char   g_log_buf[LOG_BUF_LINES][LOG_LINE_LEN];
static int    g_log_head = 0;      /* 下一个写入位置 */
static int    g_log_count = 0;     /* 已写入条数 */
static int    g_level = NPU_DEBUG_LEVEL;
static int    g_atexit_registered = 0;
static int    g_seq = 0;           /* 全局调用序号 */

/* 高精度时间戳 (微秒) */
static double get_timestamp_us(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    return (double)t * tb.numer / tb.denom / 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
#endif
}

static double g_start_us = 0.0;   /* 首次调用时间 */

static const char* dtype_str(npu_dtype_t dt) {
    switch (dt) {
        case NPU_DTYPE_FP16:  return "fp16";
        case NPU_DTYPE_FP32:  return "fp32";
        case NPU_DTYPE_BF16:  return "bf16";
        case NPU_DTYPE_INT8:  return "i8";
        case NPU_DTYPE_INT32: return "i32";
        case NPU_DTYPE_INT16: return "i16";
        default:              return "???";
    }
}

static void log_line(const char* line) {
    strncpy(g_log_buf[g_log_head], line, LOG_LINE_LEN - 1);
    g_log_buf[g_log_head][LOG_LINE_LEN - 1] = '\0';
    g_log_head = (g_log_head + 1) % LOG_BUF_LINES;
    if (g_log_count < LOG_BUF_LINES) g_log_count++;
}

/* ---- tensor 统计 ---- */

typedef struct {
    float min_val;
    float max_val;
    int   nan_count;
    int   inf_count;
    int   zero_count;
} tensor_stats_t;

static tensor_stats_t compute_stats(const void* ptr, npu_dtype_t dtype, int count) {
    tensor_stats_t s;
    s.min_val = 1e30f;
    s.max_val = -1e30f;
    s.nan_count = 0;
    s.inf_count = 0;
    s.zero_count = 0;

    for (int i = 0; i < count; i++) {
        float v = npu_read_as_float(ptr, i, dtype);
        if (isnan(v)) { s.nan_count++; continue; }
        if (isinf(v)) { s.inf_count++; continue; }
        if (v == 0.0f) s.zero_count++;
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
    }
    if (s.min_val > 1e29f) s.min_val = 0.0f;  /* all nan/inf */
    if (s.max_val < -1e29f) s.max_val = 0.0f;
    return s;
}

/* ---- public API ---- */

void npu_debug_flush(void) {
    if (g_log_count == 0) return;
    fprintf(stderr, "\n===== NPU DEBUG TRACE (%d entries) =====\n", g_log_count);
    int start = (g_log_count < LOG_BUF_LINES)
                ? 0
                : g_log_head;  /* ring buffer wrapped */
    for (int i = 0; i < g_log_count; i++) {
        int idx = (start + i) % LOG_BUF_LINES;
        fprintf(stderr, "%s\n", g_log_buf[idx]);
    }
    fprintf(stderr, "===== END NPU DEBUG TRACE =====\n\n");
    g_log_count = 0;
    g_log_head = 0;
}

static void atexit_flush(void) {
    npu_debug_flush();
}

void npu_debug_enable(int level) {
    g_level = level;
    if (level > 0 && !g_atexit_registered) {
        atexit(atexit_flush);
        g_atexit_registered = 1;
    }
}

int npu_debug_level(void) {
    return g_level;
}

void npu_debug_trace_op(const char* op_name, TidInfo tid,
                        const npu_debug_tensor_arg_t* tensors,
                        int n_tensors) {
    int lvl = g_level;
    if (lvl <= 0) return;

    if (!g_atexit_registered) {
        atexit(atexit_flush);
        g_atexit_registered = 1;
    }

    /* 时间戳 */
    double now = get_timestamp_us();
    if (g_start_us == 0.0) g_start_us = now;
    double elapsed_us = now - g_start_us;
    int seq = g_seq++;

    /* op header: 序号 + 时间戳 + tid + op名 */
    char line[LOG_LINE_LEN];
    snprintf(line, LOG_LINE_LEN,
             "#%-4d %10.1fus [tid=%d dep=C%d/V%d/D%d/I%d] %s",
             seq, elapsed_us,
             tid.task_id, tid.dep_cube_tid, tid.dep_vector_tid,
             tid.dep_dma_tid, tid.dep_idma_tid, op_name);
    log_line(line);

    for (int i = 0; i < n_tensors; i++) {
        const npu_debug_tensor_arg_t* t = &tensors[i];
        if (lvl == 1) {
            /* level 1: 地址 + dtype + count */
            snprintf(line, LOG_LINE_LEN,
                     "  %-10s ptr=%p  dtype=%-4s  count=%d",
                     t->name, t->ptr, dtype_str(t->dtype), t->count);
            log_line(line);
        } else {
            /* level 2: 地址 + dtype + count + min/max/nan/inf */
            tensor_stats_t s = compute_stats(t->ptr, t->dtype, t->count);
            snprintf(line, LOG_LINE_LEN,
                     "  %-10s ptr=%p  dtype=%-4s  count=%-6d  "
                     "min=%+.4e  max=%+.4e  nan=%d  inf=%d  zero=%d",
                     t->name, t->ptr, dtype_str(t->dtype), t->count,
                     s.min_val, s.max_val,
                     s.nan_count, s.inf_count, s.zero_count);
            log_line(line);
        }
    }
}
