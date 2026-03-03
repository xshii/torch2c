# npu_cpu_mock — NPU C API 的 CPU 模拟实现

## 职责

用纯 C (C99) 实现 17 个 NPU C API 函数，在 CPU 上模拟 NPU 行为。用于：
- 生成的 C 工程在无真实 NPU 硬件时编译运行
- 与 PyTorch golden 数据做精度比对

## 与 Python 侧无依赖

本模块是纯 C 代码，不依赖 `npu_compiler/` 下任何 Python 模块，可独立编译测试。

## 17 个 NPU API

### 计算 ops（12 个）

| 函数 | 计算单元 | CPU 实现 |
|------|---------|---------|
| `npu_matmul` | Cube | 三重循环，float32 累加 |
| `npu_add` | Vector | 逐元素 a[i]+b[i] |
| `npu_mul` | Vector | 逐元素 a[i]*b[i] |
| `npu_mul_scalar` | Vector | input[i]*scalar |
| `npu_gelu` | Vector | `x*0.5*(1+erf(x/sqrt(2)))`，用 C99 `erff()` |
| `npu_layernorm_part1` | Vector | mean -> var -> normalize -> gamma*x+beta |
| `npu_layernorm_part2` | Vector | identity: memcpy(out, inter) |
| `npu_softmax_part1` | Vector | row-wise: max -> exp(x-max) -> sum -> normalize |
| `npu_softmax_part2` | Vector | identity: memcpy(out, inter) |
| `npu_transpose` | Vector | N 维转置：flat index -> coords -> swap dims -> output index |
| `npu_transpose_2d` | Vector | out[c][r] = in[r][c] |
| `npu_reshape` | Scalar | memcpy |

### DMA ops（3 个）

| 函数 | CPU 实现 |
|------|---------|
| `npu_dma_load` | memcpy |
| `npu_dma_store` | memcpy |
| `npu_dma_barrier` | no-op |

### Sync ops（2 个）

| 函数 | CPU 实现 |
|------|---------|
| `npu_set_dependency` | no-op |
| `npu_barrier` | no-op |

## 关键设计

1. **混合精度**：所有计算在 float32 上执行，输入/输出通过 `npu_read_as_float` / `npu_write_from_float` 做 dtype 转换
2. **fp16 纯位运算**：IEEE 754 位操作实现 fp16<->float，不依赖编译器 `__fp16` 扩展
3. **Part1/Part2 拆分**：part1 执行完整数学计算，part2 为 identity copy
4. **格式忽略**：CPU mock 忽略 NZ/NC1HWC0，全部按 ND 处理

## C 函数签名参考

```c
typedef enum { NPU_DTYPE_FP16=0, NPU_DTYPE_FP32, NPU_DTYPE_BF16,
               NPU_DTYPE_INT8, NPU_DTYPE_INT32, NPU_DTYPE_INT16 } npu_dtype_t;
typedef enum { NPU_FORMAT_ND=0, NPU_FORMAT_NZ, NPU_FORMAT_NC1HWC0 } npu_format_t;

void npu_matmul(void* a, void* b, void* out, int M, int N, int K, npu_dtype_t dtype, npu_format_t fmt);
void npu_add(void* a, void* b, void* out, int count, npu_dtype_t dtype);
void npu_mul(void* a, void* b, void* out, int count, npu_dtype_t dtype);
void npu_mul_scalar(void* input, void* out, float scalar, int count, npu_dtype_t dtype);
void npu_gelu(void* input, void* out, int count, npu_dtype_t dtype);
void npu_layernorm_part1(void* input, void* gamma, void* beta, void* out,
                         int hidden, int seq, float eps, npu_dtype_t dtype);
void npu_layernorm_part2(void* inter, void* orig, void* out, int hidden, npu_dtype_t dtype);
void npu_softmax_part1(void* input, void* out, int dim, int count, npu_dtype_t dtype);
void npu_softmax_part2(void* inter, void* out, int count, npu_dtype_t dtype);
void npu_transpose(void* input, void* out, int ndim, const int* dims, int dim0, int dim1, npu_dtype_t dtype);
void npu_transpose_2d(void* input, void* out, int rows, int cols, npu_dtype_t dtype);
void npu_reshape(void* input, void* out, int count);

void npu_dma_load(void* l1_dst, void* hbm_src, int size, npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_store(void* hbm_dst, void* l1_src, int size, npu_format_t src_fmt, npu_format_t dst_fmt);
void npu_dma_barrier(void);
void npu_set_dependency(int src_id, int dst_id);
void npu_barrier(void);
```

## 目录结构

```
npu_cpu_mock/
├── CMakeLists.txt
├── include/
│   ├── npu_api.h              # 公共头文件（枚举 + 17 函数声明 + dtype helpers）
│   └── npu_fp16.h             # fp16 <-> float 位运算转换
├── src/
│   ├── npu_dtype_utils.c      # read_as_float / write_from_float / dtype_size
│   ├── npu_compute_elementwise.c  # add, mul, mul_scalar, gelu
│   ├── npu_compute_matmul.c   # matmul
│   ├── npu_compute_norm.c     # layernorm_part1, layernorm_part2
│   ├── npu_compute_softmax.c  # softmax_part1, softmax_part2
│   ├── npu_compute_transpose.c # transpose, transpose_2d, reshape
│   ├── npu_dma.c              # DMA ops（memcpy）
│   └── npu_sync.c             # Sync ops（noop）
└── tests/
    ├── test_framework.h       # 测试宏（RUN_TEST, ASSERT_*）
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

## 交付物

1. `include/` 下两个头文件
2. `src/` 下 8 个源文件
3. `tests/` 下 6 个测试文件 + test_framework.h
4. 根目录和 tests/ 各一个 CMakeLists.txt
5. `cmake --build build && ctest` 全绿
