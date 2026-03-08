# 架构设计文档

> 本文档描述 torch2c 编译器的整体架构、核心数据结构、模块职责与接口契约。
> 目标读者：新成员上手、问题定位、代码审查参考。

## 1. 系统概览

torch2c 是一个 **PyTorch → NPU C 工程的纯离线编译器**。输入 `nn.Module`，输出可在 NPU 工具链编译执行的完整 C 项目（含源码、权重、golden 数据、CMake 构建）。

```
PyTorch nn.Module + dummy_input
        │
        ▼
  ┌────────────────────────────────┐
  │     9-Pass 编译管线 (Python)     │
  │  graph_capture → ... → codegen  │
  └────────────────────────────────┘
        │
        ▼
  完整 C 工程 (output/)
  ├── src/model_graph.c  (算子调用)
  ├── src/model_weights.h (权重数组)
  ├── golden/*.bin        (精度比对)
  ├── npu_mock.h          (CPU mock)
  └── CMakeLists.txt
```

## 2. 编译管线 (9 Pass)

```
① graph_capture      torch.export → Graph IR (ATen ops)
        ↓
② op_mapping         ATen → NPU 1:1 直接映射
        ↓
③ op_decomposition   ATen → NPU 1:N 固定裂解 + broadcast 插入
        ↓
④ op_absorption      独立算子 → 参数吸收 (mask → softmax)
        ↓
⑤ format_annotator   标注 format/dtype/compute_dtype
        ↓
⑤a reformat_inserter 插入 dma_reformat (format 不匹配时)
        ↓
⑤b storage_assigner  分配 tensor 存储类型 (hbm/local/pipe)
        ↓
⑥ validator          校验所有 npu_op 是否受支持
        ↓
⑦ memory_planner     HBM 分配 + L1 布局 + DMA 计划
        ↓
⑧ scheduler          拓扑排序 + 依赖分析 + task_id
        ↓
⑨ codegen            生成完整 C 工程
```

### 管线编排 (`integration/pipeline.py`)

| 阶段 | 调用方式 | 特殊之处 |
|------|---------|---------|
| Pass ① | `graph_capture.capture(model, input)` | 不走标准 `run(graph, config)` |
| Pass ②-⑤b | `_run_middle_passes()` 声明式循环 | 统一 `_PassDesc` 描述 |
| Pass ⑥-⑧ | `_run_late_passes()` | validator 抛异常中止；memory_planner 返回 tuple |
| Pass ⑨ | `_run_codegen()` | 需要 model/weights/golden 等额外上下文 |

每个 Pass 执行后调用 `_run_post_validation()` 收集诊断，`DiagnosticCollector` 汇总所有 warning/error。

## 3. 核心数据结构 (Graph IR)

位于 `torch2c/common/graph_ir.py`，三个 dataclass 贯穿整条管线：

### 3.1 Tensor

```python
@dataclass
class Tensor:
    id: str                          # 唯一标识 "t_0", "weight_linear_0"
    shape: list[int]                 # [1, 32, 64]
    dtype: str                       # "fp16", "fp32" (NPU 目标精度)
    format: str = "nd"               # "nd" | "nz" (HBM 存储格式)
    src_dtype: str | None = None     # 原始精度 (.pth 文件)
    hbm_offset: int | None = None    # HBM 地址 (memory_planner 填入)
    hbm_size: int | None = None      # HBM 大小 bytes
    l1_offset: int | None = None     # L1 地址
    is_weight: bool = False
    is_model_input: bool = False
    is_model_output: bool = False
    name: str | None = None          # 权重名 (state_dict key)
    storage: str = "hbm"             # "hbm" | "local" | "pipe"
    producer_node_id: str | None     # 生产者节点
    consumer_node_ids: list[str]     # 消费者节点列表
```

### 3.2 Node

```python
@dataclass
class Node:
    id: str                              # "node_0"
    op_type: str                         # "aten.mm.default" → "cube_matmul"
    inputs: list[str]                    # Tensor ID 列表
    outputs: list[str]                   # Tensor ID 列表
    params: dict                         # 算子参数 (dim, stride, etc.)
    compute_unit: str | None = None      # "cube" | "vector" | "idma" | "dma"
    npu_op: str | None = None            # NPU 算子名
    is_mapped: bool = False              # 映射/裂解完成
    format_annotation: dict | None = None # {inputs: [{dtype,format}], outputs: [...]}
    schedule_order: int | None = None    # 执行顺序
    task_id: int = 0                     # TidInfo 全局任务 ID (1-indexed)
    dependencies: list[str]              # 前驱节点 ID
    absorbed_inputs: dict                # param_name → tensor_id
    module_path: str | None = None       # 源模块路径
```

### 3.3 Graph

```python
@dataclass
class Graph:
    nodes: dict[str, Node]       # node_id → Node
    tensors: dict[str, Tensor]   # tensor_id → Tensor
    execution_order: list[str]   # 拓扑排序后的 node_id 列表
```

关键方法：`topo_sort()`, `validate()`, `to_dict()`/`from_dict()`, `summary()`

### 3.4 字段所有权表

| 字段 | 写入 Pass | 读取 Pass |
|------|----------|----------|
| `Node.op_type` | ① graph_capture | ②③ mapping/decomp |
| `Node.npu_op` | ②③ mapping/decomp | ⑤⑥⑨ format/validator/codegen |
| `Node.compute_unit` | ②③ mapping/decomp | ⑧⑨ scheduler/codegen |
| `Node.is_mapped` | ②③ mapping/decomp | ③⑥ decomp/validator |
| `Node.format_annotation` | ⑤ format_annotator | ⑦⑨ memory/codegen |
| `Node.schedule_order` | ⑧ scheduler | ⑨ codegen |
| `Node.dependencies` | ⑧ scheduler | ⑨ codegen |
| `Node.absorbed_inputs` | ④ op_absorption | ⑦⑨ memory/codegen |
| `Tensor.format` | ⑤ format_annotator | ⑦⑨ memory/codegen |
| `Tensor.dtype` | ①⑤ capture/format | ⑦⑨ memory/codegen |
| `Tensor.hbm_offset/size` | ⑦ memory_planner | ⑨ codegen |
| `Tensor.l1_offset` | ⑦ memory_planner | ⑨ codegen |
| `Tensor.storage` | ⑤b storage_assigner | ⑦⑨ memory/codegen |

## 4. 模块职责与接口

### 4.1 标准 Pass 接口

```python
# 大多数 Pass 遵循此模式：
def run(graph: Graph, config: dict) -> Graph

# 可选的后置校验：
def post_validate(graph: Graph) -> list[str]  # 返回错误消息列表
```

例外：
- `graph_capture.capture(model, input, mask)` — 不接收 Graph
- `memory_planner.run(graph, config)` → `(Graph, list[DmaPlan])`
- `scheduler.run(graph)` — 不需要 config

### 4.2 各模块摘要

| # | 模块 | 输入 config key | 核心逻辑 |
|---|------|----------------|---------|
| ① | graph_capture | — | torch.export → Graph IR，标注传播 |
| ② | op_mapping | `mapping` | 查表 direct_mappings.yaml 设置 npu_op |
| ③ | op_decomposition | `decomposition` | 1:N 裂解 + broadcast 插入 |
| ④ | op_absorption | `absorption` | 模式匹配删除被吸收节点 |
| ⑤ | format_annotator | `format` | 按优先级标注 dtype/format/compute_dtype |
| ⑤a | reformat_inserter | `reformat` | format 不匹配时插入 dma_reformat |
| ⑤b | storage_assigner | `storage` | 按 allowed_pairs 分配 local/pipe |
| ⑥ | validator | `supported_ops` (动态构建) | 检查 npu_op ∈ supported_ops |
| ⑦ | memory_planner | `hardware` | HBM best-fit + L1 布局 + DMA 计划 |
| ⑧ | scheduler | — | 拓扑排序 + 数据/单元依赖 |
| ⑨ | codegen | `signatures`+`hardware`+`debug` | 生成 C 工程全部文件 |

### 4.3 模块依赖关系

```
common (所有模块依赖)
  │
  ├── graph_capture        ─┐
  ├── op_mapping            │
  ├── op_decomposition      │ 互不依赖
  ├── op_absorption         │ 可并行开发
  ├── format_annotator      │
  ├── reformat_inserter     │
  ├── storage_assigner      │
  ├── validator             │
  ├── memory_planner        │
  ├── scheduler             │
  └── codegen              ─┘
       │
       ▼
    integration (串联编排)
```

## 5. 配置系统

### 5.1 配置文件清单 (`integration/config/`)

| 文件 | 用途 | 消费者 |
|-----|------|--------|
| `direct_mappings.yaml` | ATen→NPU 1:1 映射 (19条) | op_mapping |
| `decompositions.yaml` | 1:N 裂解规则 (2条) | op_decomposition |
| `absorptions.yaml` | 参数吸收规则 | op_absorption |
| `hardware_config.yaml` | HBM/L1/L2 大小、对齐、cube_size | memory_planner |
| `c_api_signatures.yaml` | C API 函数签名 + 参数源映射 | validator, codegen |
| `capture_rules.yaml` | 参数重命名、输入重排 | graph_capture |
| `model_config.yaml` | 模型参数 (d_model, seq_len 等) | 参考用 |
| `codegen_config.yaml` | 代码生成选项 | codegen |
| `debug.yaml` | 调试/trace 开关 | 全局 |

### 5.2 配置优先级

```
compile() 参数 > @torch2c_config 装饰器 > 配置文件默认值
```

### 5.3 配置加载

```python
# common/config_loader.py
def load_config(path: str, required_keys: list[str] = None) -> dict
# 使用 yaml.safe_load()，验证 required_keys 存在
# 失败抛 ConfigError
```

## 6. 异常体系

```
CompilerError (基类)
├── ConfigError        (配置文件错误)
├── MappingError       (算子映射失败)
├── DecompositionError (算子裂解失败)
├── AbsorptionError    (参数吸收失败)
├── ValidationError    (合法性校验失败)
├── MemoryPlanError    (内存编排失败)
└── CodegenError       (代码生成失败)
```

`DiagnosticCollector` 在管线中收集各 Pass 的 warning/error，最终汇总输出。

## 7. NPU CPU Mock (C 层)

位于 `npu_cpu_mock/`，纯 C99 实现全部 NPU 算子，用于 CPU 上的精度验证。

**核心类型** (`npu_api.h`)：
- `npu_dtype_t` — FP16/FP32/BF16/INT8/INT32
- `npu_format_t` — ND/NZ/NC1HWC0
- `npu_tensor_t` — {addr, dtype, format}
- `TidInfo` — {task_id, dep_task_id, dep_unit}

**实现文件**：
- `npu_compute_matmul.c` — matmul, matmul_bias
- `npu_compute_elementwise.c` — add, mul, gelu
- `npu_compute_norm.c` — layernorm part1/part2
- `npu_compute_softmax.c` — softmax part1/part2
- `npu_compute_transpose.c` — transpose, reshape, broadcast
- `npu_dma.c` — dma_move, dma_reformat, idma_move

## 8. 关键设计决策

| 决策 | 理由 |
|------|------|
| 禁止 torch.export 自动分解 | 保留 layer_norm/softmax 高级算子 |
| Graph IR 全管线共享 | 减少中间转换，简化数据流 |
| DMA 随路 format 转换 | 无独立 convert 节点，匹配硬件行为 |
| tensor.format = HBM 存储格式 | 消费者通过 DMA load 转换到计算格式 |
| Part1/Part2 裂解 | 匹配 NPU 双缓冲乒乓调度 |
| TidInfo 按单元依赖 | 支持流水线并行调度 |
| 配置驱动算子扩展 | 新算子仅需改 YAML，不改代码 |
| compute_dtype 分层优先级 | per-op > per-unit > global，灵活混精度 |
