# /t2c-scaffold — 脚手架引擎

> 合并自: add-op + add-pass + adapt-format
> 用法: `/t2c-scaffold [type] [name] [details]`
> 类型: `op` | `pass` | `format` | `dtype`
> 示例: `/t2c-scaffold op vector_silu` | `/t2c-scaffold pass cd_my_fuser` | `/t2c-scaffold format bfloat16`

你是 torch2c NPU 编译器的脚手架引擎。根据 type 自动生成所有必需文件并验证一致性。
$ARGUMENTS 包含 type 和 name。若为空，询问用户要添加什么。

---

## 0. 类型路由

| 关键词 | 类型 | 产出 |
|--------|------|------|
| `op` / `算子` | 新增 NPU 算子 | 10 个文件/配置变更 |
| `pass` | 新增优化 Pass | 模块目录 + pipeline 接入 |
| `format` / `dtype` / `格式` | 格式/类型适配 | 配置 + sizing + codegen 变更 |

---

## 1. 新增 NPU 算子 (`/add op <name>`)

### 1.0 信息收集

开始前确认（不确定则询问用户）：

| 信息 | 示例 | 如何确认 |
|------|------|----------|
| ATen 算子名 | `aten.silu.default` | 跑 `torch.export` 确认 |
| NPU 算子名 | `vector_silu` | 前缀=计算单元 |
| 计算单元 | `cube` / `vector` / `idma` | 矩阵乘=cube，逐元素=vector，搬运=idma |
| C 函数参数 | 参考已有同类算子 | 查 `c_api_signatures.yaml` |
| 输出 shape | 与输入同 / 有变化 | 查 ATen 文档 |
| 是否需要裂解 | layernorm → part1+part2 | 复合算子需要 |

### 1.1 算子映射 — `torch2c/integration/config/direct_mappings.yaml`

```yaml
aten.silu.default:
  npu_op: vector_silu
  compute_unit: vector
```

### 1.2 C API 签名 — `torch2c/integration/config/c_api_signatures.yaml`

在对应 section（cube_ops / vector_ops / idma_ops）下添加。

source 类型：
- `tensor.input_N` / `tensor.output_N` → tensor descriptor
- `tensor.input_N.count` → 元素总数
- `tensor.input_N.ndim` / `.shape` / `.hbm_size` → 维度/shape/字节
- `param.xxx` → node.params 中的值
- `param.compute_dtype` → 计算精度

### 1.3 Tiling 配置 — `torch2c/integration/config/tiling_config.yaml`

```yaml
vector_silu:
  tile_dim: -2
  min_tile: 1
```

### 1.4 命名规则 — `torch2c/integration/config/naming_rules.yaml`

```yaml
vector_silu:
  short_name: silu
```

### 1.5 代价模型 — `torch2c/integration/config/cost_model_config.yaml`

```yaml
vector_silu:
  flops_per_element: 4
```

对于需要精确成本的算子，可注册 Python cost function：

```python
# torch2c/optpass/cd_roofline/_builtin_costs.py
from torch2c.optpass.cd_roofline.roofline_analyzer import register_cost_fn, CostContext, CostResult

@register_cost_fn("vector_silu")
def _cost_silu(ctx: CostContext) -> CostResult:
    flops = ctx.elem_count * 4
    return CostResult(compute_cycles=flops / ctx.hw.vector_throughput,
                      launch_cycles=ctx.hw.vector_launch)
```

### 1.6 C Mock 声明 — `npu_cpu_mock/include/npu_api.h`

```c
void vector_silu(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                 int count, npu_dtype_t compute_dtype);
```

### 1.7 C Mock 实现 — `npu_cpu_mock/src/npu_compute_xxx.c`

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
        float result = x / (1.0f + expf(-x));
        npu_write_store(po, i, result, compute_dtype, dt);
    }

    NPU_TRACE_END("vector_silu", tid, _dbg, 2);
}
```

**C Mock 要点**：
- 用 `.ptr` 不用 `.addr`
- 用 `npu_t_ptr()` 获取指针
- 用 `npu_read_compute()` / `npu_write_store()` 做精度转换
- 用 `NPU_DBG_T()` 构建 debug 参数

### 1.8 验证一致性

```bash
.venv/bin/pytest torch2c/integration/tests/test_config_consistency.py -v
```

### 1.9 单元测试

在相关模块 `tests/` 下添加测试，确保 mapping → codegen → C mock 全链路正确。

### 1.10 全量回归

```bash
.venv/bin/pytest --tb=short -q
```

### 裂解算子附加步骤

复合算子（如 layernorm → part1 + part2）还需要：

1. 在 `decompositions.yaml` 添加裂解规则：
```yaml
decompositions:
  vector_layernorm:
    steps:
      - npu_op: vector_layernorm_part1
        compute_unit: vector
      - npu_op: vector_layernorm_part2
        compute_unit: vector
```
2. 裂解后的每个子算子都完成步骤 1.2-1.8

---

## 2. 新增优化 Pass (`/add pass <prefix_name>`)

### 2.0 前置决策

| 决策 | 选项 |
|------|------|
| 必须 vs 可选 | 必须 → `a_capture/b_lowering/c_backend/d_emission`；可选 → `optpass/` |
| 前缀 | `bc_`=B→C间, `c_`=C内, `cd_`=C→D间, `d_`=D内 |
| 接口 | 统一 `run(graph, config) -> Graph` + `post_validate(graph) -> list[str]` |

### 2.1 创建模块目录

```
torch2c/optpass/{prefix}_{name}/
├── __init__.py
├── {name}.py          # 核心逻辑（< 300 行）
└── tests/
    ├── __init__.py
    └── test_{name}.py
```

### 2.2 pass 入口 — `{name}.py`

```python
"""pass_name — 一句话说明。"""
from __future__ import annotations
from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)

def run(graph: Graph, config: dict) -> Graph:
    # 1. 从 config 读取参数
    # 2. 遍历 graph.nodes / graph.tensors
    # 3. 执行变换，每步 log_opt(node, "pass_name", "动作", "原因")
    logger.info("pass_name 完成: %d 处变换", count)
    return graph

def post_validate(graph: Graph) -> list[str]:
    errors: list[str] = []
    # 校验变换后的图是否满足不变量
    return errors
```

### 2.3 `__init__.py`

```python
"""pass_name — 一句话说明。"""
from torch2c.optpass.{prefix}_{name} import {name}  # noqa: F401
```

### 2.4 注册 toggle — `torch2c/common/pass_config.py`

```python
class OptionalPass(Enum):
    MY_PASS = auto()     # 新增

class PassConfig:
    my_pass: bool = True  # 默认启用（或 False 如果开发中）
```

### 2.5 接入 pipeline — `torch2c/integration/pipeline.py`

```python
from torch2c.optpass.{prefix}_{name} import {name}

# 在 _OPTIMIZATION_PASSES 中按执行顺序插入
_PassDesc("name", "编号", {name}.run, "config_key",
          {name}.post_validate, toggle=OptionalPass.MY_PASS),
```

### 2.6 测试

必须包含的测试：

```python
class TestRun:
    def test_basic_transform(self): ...       # 正常变换
    def test_no_op_when_nothing(self): ...     # 无匹配时不变换
    def test_idempotent(self): ...             # 跑两次结果相同

class TestPostValidate:
    def test_clean_after_run(self): ...        # post_validate 返回 []
```

### 2.7 验证

```bash
.venv/bin/pytest torch2c/optpass/{prefix}_{name}/tests/ -v
.venv/bin/pytest --tb=short -q
```

### Pass 编写规范

1. 只改 graph 不改 config
2. 用 log_opt 记录每个决策
3. 保持幂等
4. post_validate 校验不变量
5. 函数 < 50 行

---

## 3. 格式/类型适配 (`/add format|dtype <name>`)

### 3.1 新增 dtype

| 步骤 | 文件 |
|------|------|
| 1 | `hardware_config.yaml` → `block_pad.alignment` 每个 format 下加新 dtype |
| 2 | `torch2c/common/sizing.py` → `_DEFAULT_ALIGNMENT` 添加（必须与 YAML 同步） |
| 3 | `torch2c/common/dtypes.py` → dtype info（bytes、C enum、numpy 映射） |
| 4 | `hardware_config.yaml` → `fractal.c0_by_dtype` 添加 c0 值 |
| 5 | 测试: `pytest torch2c/optpass/c_block_pad/tests/ -v` |

### 3.2 新增 format

| 步骤 | 文件 |
|------|------|
| 1 | `hardware_config.yaml` → `block_pad.alignment` 加新 format section |
| 2 | `hardware_config.yaml` → `format_capabilities` 加各单元对新 format 的支持 |
| 3 | `torch2c/common/sizing.py` → `_DEFAULT_ALIGNMENT` + `calc_padded_size` 分形判断 |
| 4 | `torch2c/d_emission/codegen/_helpers.py` → `FORMAT_MAP` 加 format → C enum |
| 5 | `npu_cpu_mock/include/npu_api.h` → `npu_format_t` 枚举 |
| 6 | 全量回归测试 |

### 3.3 修改对齐规则

1. 改 `hardware_config.yaml` → `block_pad.alignment`
2. **同步改** `torch2c/common/sizing.py` → `_DEFAULT_ALIGNMENT`（两者必须一致！）
3. `pytest torch2c/optpass/c_block_pad/tests/ -v`
4. 全量回归

### 关键约束

- `_DEFAULT_ALIGNMENT`(sizing.py) 与 `block_pad.alignment`(hardware_config.yaml) **必须同步**
- 所有 `calc_padded_size` 调用必须用 `get_dim_align(t.format, t.dtype)`，不硬编码
- DMA 随路转换 = HBM 格式和 L1 格式可以不同
- 修改 format 后检查 `format_capabilities` 是否需要更新

---

## 4. 完成后自动检查

无论哪种 type，完成后必须：

```bash
# 1. 配置一致性
.venv/bin/pytest torch2c/integration/tests/test_config_consistency.py -v

# 2. 全量回归
.venv/bin/pytest --tb=short -q
```

如有失败，自动进入 `/dev fix` 模式修复。
