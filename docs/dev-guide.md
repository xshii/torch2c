# 开发指南

> 面向日常开发的实操手册：如何扩展算子、定位问题、编写测试。

## 1. 新增算子（配置驱动）

### 1.1 添加 1:1 直接映射算子

**场景**：ATen 算子与 NPU 算子一一对应（如 `aten.relu` → `vector_relu`）。

**步骤**：

1. **确认 ATen 算子全名**：用 `torch.export` 导出模型，查看 FX graph 中的实际名称
   ```python
   ep = torch.export.export(model, (dummy_input,))
   print(ep.graph)  # 找到如 "aten.relu.default"
   ```

2. **添加映射** — 编辑 `integration/config/direct_mappings.yaml`：
   ```yaml
   mappings:
     aten.relu.default:
       npu_op: vector_relu
       compute_unit: vector
   ```

3. **添加 C API 签名** — 编辑 `integration/config/c_api_signatures.yaml`：
   ```yaml
   compute_ops:
     vector_relu:
       - { name: input, type: tensor_desc, source: "tensor.input_0" }
       - { name: output, type: tensor_desc, source: "tensor.output_0" }
       - { name: tid, type: tid_info, source: "tid" }
   ```

4. **实现 C mock** — `npu_cpu_mock/src/` 中添加实现

5. **同步模块局部 config**（如有）— 确保 `op_mapping/config/` 等副本一致

6. **测试**：
   ```bash
   pytest torch2c/op_mapping/tests/ -v
   pytest torch2c/integration/tests/test_op_semantics.py -v
   ```

### 1.2 添加 1:N 裂解算子

**场景**：ATen 算子需拆分为多个 NPU 算子（如 `aten.batch_norm` → part1 + part2）。

编辑 `integration/config/decompositions.yaml`：
```yaml
decompositions:
  aten.native_batch_norm.default:
    steps:
      - npu_op: vector_batchnorm_part1
        compute_unit: vector
      - npu_op: vector_batchnorm_part2
        compute_unit: vector
```

**注意**：中间 tensor shape = 第一个输入的 shape（§16.5 设计决策）。

### 1.3 添加参数吸收

编辑 `integration/config/absorptions.yaml`：
```yaml
absorptions:
  - absorbed_op: npu_add
    target_op: npu_softmax_part1
    param_name: mask
    absorbed_input_index: 1    # add 的第几个输入被吸收
    passthrough_input_index: 0 # add 的第几个输入传递给 target
```

## 2. 问题定位

### 2.1 定位流程

```
观察现象
    ↓
确定出问题的 Pass（看日志 "Pass ① 开始" / "Pass ② 完成"）
    ↓
dump 该 Pass 前后的 Graph IR（用 graph.to_dict() 或 graph.summary()）
    ↓
对比字段变化（参考字段所有权表 → docs/architecture.md §3.4）
    ↓
定位到具体模块代码
```

### 2.2 日志系统

```python
from torch2c.common import get_logger
logger = get_logger(__name__)
# 格式: [2026-03-08 12:34:56] [INFO] [module.name] message
```

- 设置日志级别：`export NPU_LOG_LEVEL=DEBUG`
- 输出到文件：`setup_logging(log_file="compile.log")`

### 2.3 快速诊断工具

```python
from torch2c.integration.pipeline import inspect

# 快速查看模型标注是否正确传播（不执行编译）
graph = inspect(model, dummy_input, mask=mask)
```

### 2.4 Graph IR 序列化调试

```python
import json

# 任意 Pass 前后 dump Graph
d = graph.to_dict()
with open("graph_after_pass3.json", "w") as f:
    json.dump(d, f, indent=2)

# 查看标注分布
print(graph.format_npu_annotations())

# 查看图摘要
print(graph.summary())
```

### 2.5 可视化产物

编译完成后在 `output/viz/` 自动生成：
- `graph.dot` — Graphviz 算子依赖图
- `graph.txt` — ASCII 算子依赖图
- `lifetime_hbm.txt` — HBM 生命周期图
- `lifetime_l1.txt` — L1 生命周期图

### 2.6 常见问题速查

| 现象 | 可能原因 | 检查点 |
|------|---------|--------|
| `ValidationError: unsupported op` | 新算子未加映射/裂解 | `direct_mappings.yaml` 或 `decompositions.yaml` |
| C 精度偏差大 | compute_dtype 不对 | `format_annotation.compute_dtype`，检查 `model_config` 规则 |
| HBM 溢出 | tensor 未复用 | `lifetime_hbm.txt`，检查 liveness 分析 |
| L1 溢出 | 单算子输入太多/太大 | `_l1_alloc.py` 的 per-op 分配日志 |
| DMA format 错误 | format_annotation 不一致 | `format_annotator` 输出 vs `c_api_signatures` |
| 权重形状不对 | 缺少 weight transpose | `graph_capture._handle_call()` 的 transpose 逻辑 |
| 多输出算子输出丢失 | getitem 处理有误 | `graph_capture._handle_call()` 的 multi-output 分支 |

### 2.7 调试配置 (`debug.yaml`)

```yaml
torch_trace:
  enabled: true       # Hook PyTorch forward 追踪
  leaf_only: true
memory_layout:
  enabled: true       # 打印内存分配详情
c_mock_trace:
  compile_level: 2    # C 编译期调试级别 (0-2)
  runtime_level: 2    # C 运行时调试级别 (0-2)
```

## 3. 测试策略

### 3.1 测试分层

```
┌────────────────────────────────────────┐
│ ST (System Test)                       │  ← 端到端编译+C执行+精度比对
│   integration/demo/demo_st/            │     pytest -m "not slow"
├────────────────────────────────────────┤
│ IT (Integration Test)                  │  ← 多 Pass 串联语义验证
│   integration/tests/test_pipeline.py   │
│   integration/tests/test_op_semantics  │
│   integration/tests/test_mha_semantics │
├────────────────────────────────────────┤
│ UT (Unit Test)                         │  ← 单模块隔离测试
│   <module>/tests/test_*.py             │     pytest torch2c/<module>/
├────────────────────────────────────────┤
│ C UT                                   │  ← C mock 函数级测试
│   npu_cpu_mock/tests/test_*.c          │     ctest
└────────────────────────────────────────┘
```

### 3.2 编写单元测试

**模式**：使用 Graph 工厂函数构造测试图

```python
from torch2c.common.testing import make_linear_chain

def test_basic_mapping():
    graph = make_linear_chain(n_ops=3, ops=[
        ("n0", "aten.mm.default", None),
        ("n1", "aten.add.Tensor", None),
        ("n2", "aten.gelu.default", None),
    ])
    config = load_config("integration/config/direct_mappings.yaml")
    result = op_mapping.run(graph, config)

    assert result.nodes["n0"].npu_op == "cube_matmul"
    assert result.nodes["n0"].is_mapped is True
```

**后置校验**：每个模块提供 `post_validate(graph) -> list[str]`

```python
def test_post_validate_clean():
    graph = _make_valid_graph()
    errors = my_module.post_validate(graph)
    assert errors == []
```

### 3.3 精度标准

| 精度模式 | max_abs_diff | cosine_similarity |
|---------|-------------|-------------------|
| FP32 | < 1e-3 | > 0.9999 |
| FP16 | < 1e-2 | > 0.999 |
| Mixed (cube FP32 + vector FP16) | < 2e-2 | > 0.999 |

### 3.4 运行测试

```bash
# 全部 Python UT
pytest torch2c/ -v

# 单模块
pytest torch2c/memory_planner/tests/ -v

# 集成测试（含 C 编译）
pytest torch2c/integration/tests/ -v

# 系统测试
pytest torch2c/integration/demo/demo_st/ -v

# C mock 测试
cd npu_cpu_mock && cmake -B build && cmake --build build && cd build && ctest -V

# 语义测试（单算子精度）
pytest torch2c/integration/tests/test_op_semantics.py -v
```

### 3.5 测试覆盖清单

每个新模块/功能需覆盖：

- [ ] 正常路径（happy path）
- [ ] 边界条件（空输入、单元素、最大尺寸）
- [ ] 幂等性（同一 Pass 执行两次结果不变）
- [ ] Graph IR 一致性（`graph.validate()` 无错误）
- [ ] post_validate 无错误
- [ ] 错误输入抛出正确异常

## 4. 代码规范

### 4.1 模块结构

```
torch2c/<module>/
├── <module>.py        # 核心逻辑 (< 300 行)
├── _*.py              # 内部辅助模块
├── config/            # YAML 配置（integration/config/ 的副本）
├── demo/              # 独立运行演示
│   └── run_demo.py
├── tests/             # pytest 测试
│   └── test_<module>.py
└── README.md          # 模块说明
```

### 4.2 函数规范

- 每个函数 < 50 行
- 使用 `torch2c.common` 的 `get_logger`, `load_config`, 异常类
- 日志：INFO 级别记录 Pass 开始/结束，DEBUG 级别记录详细处理

### 4.3 配置管理

- **权威来源**：`integration/config/` 目录
- **模块局部 config/**：仅供模块独立测试/demo 使用，是副本
- 新增配置项需同步两处
