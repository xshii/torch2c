# 新增 NPU 算子

当需要添加一个新的 NPU 算子（如 vector_silu、cube_conv2d 等）时，按以下清单逐步完成。
每一步都有明确的文件路径和代码模式，严格按顺序执行。

## 前置信息收集

在开始前确认以下信息：
- **ATen 算子名**：如 `aten.silu.default`（先跑 `torch.export` 确认实际名称）
- **NPU 算子名**：如 `vector_silu`（前缀=计算单元: cube_/vector_/idma_/dma_）
- **计算单元**：cube / vector / idma
- **C 函数签名**：参数列表（参考已有算子）
- **输出 shape 规则**：与输入相同？还是有变化？

## 步骤清单

### 1. 算子映射 — `torch2c/integration/config/direct_mappings.yaml`

```yaml
aten.silu.default:
  npu_op: vector_silu
  compute_unit: vector
```

### 2. C API 签名 — `torch2c/integration/config/c_api_signatures.yaml`

在对应的 section（cube_ops / vector_ops / idma_ops）下添加：

```yaml
  vector_silu:
    unit: vector
    params:
      - { name: "input",        type: "tensor_desc", source: "tensor.input_0" }
      - { name: "out",          type: "tensor_desc", source: "tensor.output_0" }
      - { name: "count",        type: "int",         source: "tensor.input_0.count" }
      - { name: "compute_dtype", type: "dtype_enum",  source: "param.compute_dtype", default: "fp16" }
```

source 类型说明：
- `tensor.input_N` / `tensor.output_N` → tensor descriptor
- `tensor.input_N.count` → 元素总数
- `tensor.input_N.ndim` → 维度数
- `tensor.input_N.shape` → shape 数组（type 用 int_array）
- `tensor.input_N.hbm_size` → 字节数
- `param.xxx` → node.params 中的值
- `param.compute_dtype` → 计算精度

### 3. Tiling 配置 — `torch2c/integration/config/tiling_config.yaml`

```yaml
vector_silu:
  tile_dim: -2          # tiling 维度（-2 = 倒数第二维）
  min_tile: 1
```

### 4. 命名规则 — `torch2c/integration/config/naming_rules.yaml`

```yaml
vector_silu:
  short_name: silu      # codegen 中的简短名
```

### 5. 代价模型 — `torch2c/integration/config/cost_model_config.yaml`

```yaml
vector_silu:
  flops_per_element: 4  # 每元素浮点运算数估算
```

### 6. C Mock 声明 — `npu_cpu_mock/include/npu_api.h`

在对应 section 添加函数声明：

```c
void vector_silu(TidInfo tid, npu_tensor_t input, npu_tensor_t out, int count, npu_dtype_t compute_dtype);
```

### 7. C Mock 实现 — `npu_cpu_mock/src/npu_compute_xxx.c`

新建或追加到已有文件：

```c
#include "npu_api.h"
#include "npu_debug.h"
#include <math.h>

void vector_silu(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                 int count, npu_dtype_t compute_dtype) {
    npu_debug_tensor_arg_t _dbg[] = {
        NPU_DBG_T(input, input, count), NPU_DBG_T(out, out, count)
    };
    NPU_TRACE_BEGIN("vector_silu", tid, _dbg, 2);

    void* pi = npu_t_ptr(input);
    void* po = npu_t_ptr(out);
    npu_dtype_t dt = input.dtype;

    for (int i = 0; i < count; i++) {
        float x = npu_read_compute(pi, i, dt, compute_dtype);
        float result = x / (1.0f + expf(-x));  // silu = x * sigmoid(x)
        npu_write_store(po, i, result, compute_dtype, dt);
    }

    NPU_TRACE_END("vector_silu", tid, _dbg, 2);
}
```

注意：
- 用 `.ptr` 访问 tensor 数据（不是 `.addr`）
- 用 `npu_t_ptr()` 获取指针
- 用 `npu_read_compute()` / `npu_write_store()` 做精度转换
- 用 `NPU_DBG_T()` 构建 debug 参数

### 8. 验证一致性

```bash
.venv/bin/pytest torch2c/integration/tests/test_config_consistency.py -v
```

这个测试会检查所有配置表的一致性。如果报错，说明某个表漏了新算子。

### 9. 单元测试

在相关模块的 `tests/` 下添加测试，确保：
- mapping 能正确映射
- codegen 能生成正确的 C 调用
- C mock 能编译通过

### 10. 全量回归

```bash
.venv/bin/pytest --tb=short -q
```

## 如果算子需要裂解

有些复合算子（如 layernorm → part1 + part2）需要裂解规则：

在 `torch2c/integration/config/decompositions.yaml` 中添加：

```yaml
decompositions:
  vector_layernorm:               # key = 裂解前的 npu_op
    steps:
      - npu_op: vector_layernorm_part1
        compute_unit: vector
      - npu_op: vector_layernorm_part2
        compute_unit: vector
```

裂解后的每个子算子也需要完成步骤 2-8。
