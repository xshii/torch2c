# npu_cpu_mock — NPU C API 的 CPU 模拟实现

## 职责

用纯 C (C99) 实现 NPU C API，在 CPU 上模拟 NPU 行为。用于：
- 生成的 C 工程在无真实 NPU 硬件时编译运行
- 与 PyTorch golden 数据做精度比对

## 与 Python 侧无依赖

本模块是纯 C 代码，不依赖 `torch2c/` 下任何 Python 模块，可独立编译测试。

## 核心类型

```c
/* 数据类型 */
typedef enum { NPU_DTYPE_FP16=0, NPU_DTYPE_FP32, NPU_DTYPE_BF16,
               NPU_DTYPE_INT8, NPU_DTYPE_INT32, NPU_DTYPE_INT16 } npu_dtype_t;

/* 数据格式 */
typedef enum { NPU_FORMAT_ND=0, NPU_FORMAT_NZ, NPU_FORMAT_NC1HWC0 } npu_format_t;

/* 计算单元 */
typedef enum { NPU_UNIT_CUBE=0, NPU_UNIT_VECTOR, NPU_UNIT_SCALAR } npu_compute_unit_t;

/* 流水线任务描述符 */
typedef struct {
    int                 task_id;      /* 本任务 ID（软件分配） */
    int                 dep_task_id;  /* 依赖的任务 ID */
    npu_compute_unit_t  dep_unit;     /* 依赖任务所在计算单元 */
} TidInfo;

/* Tensor 描述符（L1 地址 + dtype + format） */
typedef struct {
    uint32_t      addr;    /* L1 地址 >> NPU_ADDR_SHIFT */
    npu_dtype_t   dtype;
    npu_format_t  format;
} npu_tensor_t;
```

所有 op 统一以 `TidInfo tid` 作为第一个参数，用于流水线调度（CPU mock 上为 no-op）。

## API 总览

### Cube 计算 ops

| 函数 | 说明 |
|------|------|
| `cube_matmul` | 批量矩阵乘，float32 累加，支持 compute_dtype 精度控制 |
| `cube_matmul_bias` | 矩阵乘 + 融合 bias，累加器在 float 精度上加 bias 后再写出 |

### Vector 计算 ops

| 函数 | 说明 |
|------|------|
| `vector_add` | 逐元素加 |
| `vector_mul` | 逐元素乘 |
| `vector_mul_scalar` | 逐元素标量乘 |
| `vector_gelu` | `x*0.5*(1+erf(x/sqrt(2)))`，用 C99 `erff()` |
| `vector_layernorm` / `_part1` | mean → var → normalize → gamma*x+beta |
| `vector_layernorm_part2` | identity copy |
| `vector_softmax` / `_part1` | row-wise: max → exp(x-max) → sum → normalize |
| `vector_softmax_part2` | identity copy |
| `vector_transpose` | N 维转置 (flat index → coords → swap dims → output index) |
| `vector_transpose_2d` | 2D 转置 out[c][r] = in[r][c] |

### Scalar 计算 ops

| 函数 | 说明 |
|------|------|
| `scalar_reshape` | memcpy（reshape 只改逻辑形状） |
| `scalar_broadcast` | memcpy |
| `scalar_copy` | memcpy |

### DMA ops

| 函数 | 说明 |
|------|------|
| `dma_move` | tensor 间数据搬运，支持随路 dtype 转换 |
| `dma_reformat` | 格式转换（CPU mock 上等价于 dtype 转换） |
| `idma_move` | L1 → pipe 搬运，TidInfo 携带调度信息 |

## 关键设计

1. **compute_dtype 精度控制**：所有计算 op 接受 `compute_dtype` 参数。输入通过 `npu_read_compute()` 提升到 float32 后截断到计算精度；输出通过 `npu_write_compute()` 先截断到计算精度再写入存储 dtype。当 `compute_dtype == FP32` 时零开销。
2. **TidInfo 统一调度**：每个 op 的第一个参数为 `TidInfo`，包含本任务 ID、依赖任务 ID 和依赖计算单元。CPU mock 上为 no-op，但保证接口与真实硬件一致。
3. **DMA 随路类型转换**：`dma_move` / `idma_move` 在 src/dst dtype 不同时逐元素经 float32 中转转换；dtype 相同时走 memcpy 快速路径。
4. **matmul+bias 融合**：`cube_matmul_bias` 与 `cube_matmul` 共享 `matmul_core()` 内核，bias 在 float 累加器上直接相加，不经过存储 dtype 截断，保证精度。
5. **fp16 纯位运算**：IEEE 754 位操作实现 fp16↔float，不依赖编译器 `__fp16` 扩展。
6. **Part1/Part2 拆分**：part1 执行完整计算，part2 为 identity copy，匹配 NPU 双 buffer ping-pong 调度。
7. **格式忽略**：CPU mock 忽略 NZ/NC1HWC0，全部按 ND 处理。

## 目录结构

```
npu_cpu_mock/
├── CMakeLists.txt
├── include/
│   ├── npu_api.h              # 公共头文件（类型定义 + 全部 API 声明 + inline helpers）
│   └── npu_fp16.h             # fp16 <-> float 位运算转换
├── src/
│   ├── npu_dtype_utils.c      # npu_l1_base + read_as_float / write_from_float / dtype_size
│   ├── npu_compute_elementwise.c  # add, mul, mul_scalar, gelu
│   ├── npu_compute_matmul.c   # matmul_core, matmul, matmul_bias
│   ├── npu_compute_norm.c     # layernorm, layernorm_part1, layernorm_part2
│   ├── npu_compute_softmax.c  # softmax, softmax_part1, softmax_part2
│   ├── npu_compute_transpose.c # transpose, transpose_2d, reshape, broadcast, copy
│   └── npu_dma.c              # dma_move, dma_reformat, idma_move
└── tests/
    ├── test_framework.h       # 测试宏（RUN_TEST, ASSERT_*, L1_INIT, TID0）
    ├── test_fp16.c
    ├── test_elementwise.c
    ├── test_matmul.c
    ├── test_norm.c
    ├── test_softmax.c
    ├── test_transpose.c
    └── CMakeLists.txt
```

## 构建与测试

```bash
cd npu_cpu_mock
cmake -B build
cmake --build build
cd build && ctest --output-on-failure
```

## 精度要求

- fp16 round-trip: 可表示范围内值不变
- 计算精度: max_abs_diff < 1e-3 (FP16), cosine > 0.999
