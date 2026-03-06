# NPU编译栈一阶段需求文档（最终版）

## 文档信息

| 项目 | 说明 |
|------|------|
| 项目名称 | PyTorch前端 → NPU C接口 离线编译栈 |
| 阶段 | 一阶段 MVP |
| 目标 | 2层Encoder Transformer小shape端到端跑通，精度比对通过 |
| 团队 | 1-2人 + 多AI Agent并行 |
| 周期 | AI辅助下半天（~4h）完成代码+UT，1-2天联调 |
| 交付物 | Python离线编译器 + 生成的完整C工程 + demo + UT/ST框架 |
| 代码原则 | 简化代码、高复用、高可靠、完整日志系统、模块自包含 |

---

## 1. 项目背景与总体目标

### 1.1 背景

自研NPU芯片，架构类似华为昇腾达芬奇体系（多计算单元 + 片上SRAM + HBM），已有一套C语言算子接口。需要搭建从PyTorch模型到该C接口的完整编译链路。

### 1.2 总体目标

Python作为纯离线编译器/代码生成器，不与C侧有任何运行时交互。最终产物是一整套可编译的C源代码和数据文件。

### 1.3 一阶段MVP目标

给定一个固定shape的2层Encoder Transformer（小shape），Python端完成：图捕获 → 算子映射/裂解/吸收 → 内存编排 → 生成完整C工程。C工程能在NPU工具链上编译通过，执行后与PyTorch golden数据精度比对通过。

### 1.4 一阶段不做

| 特性 | 说明 |
|------|------|
| 动态shape | 所有shape固定 |
| 训练 | 仅推理 |
| 算子融合 | 所有算子独立执行 |
| 多核调度 | 单核 |
| L1 tiling | 假设L1够大 |
| Double buffer | DMA和计算不重叠 |
| Auto-tuning | 不搜索最优配置 |

---

## 2. 硬件架构约束

### 2.1 计算单元

| 计算单元 | 职责 | 典型算子 |
|----------|------|----------|
| Cube | 矩阵乘加 | matmul, linear |
| Vector | 向量运算、激活、归一化 | relu, gelu, layernorm子算子 |
| Scalar | 流程控制、地址计算 | reshape（地址重算） |
| DMA | 数据搬运 | load, store, format转换 |

不同流水线可异步并行，通过依赖关系和barrier同步。

### 2.2 存储层次

```
HBM / DDR（片外，大容量，高延迟）
    ↓ DMA
L2 Buffer（片上共享）
    ↓ DMA
L1 Buffer（核内，一阶段假设足够大）
    ↓
L0A / L0B / L0C（计算单元缓冲）
Unified Buffer（Vector工作空间）
```

### 2.3 数据格式与类型

- 分形格式（NZ/NC1HWC0等）匹配Cube硬件特性
- DMA搬运时可同步完成ND→分形格式转换
- 计算单元支持随路format/dtype转换
- **因此format/dtype转换不作为独立算子，只标注参数**

### 2.4 算子裂解特性

NPU的裂解是**固定成组的算子组合**（如layernorm → part1 + part2），不是原子化数学分解。中间tensor的shape/dtype由C接口设计决定。

---

## 3. 系统架构

### 3.1 整体流程

```
PyTorch 模型
    │
    ▼
① graph_capture    : torch.export → Graph IR
    ▼
② op_mapping       : ATen op → NPU op（1对1直接映射）
    ▼
③ op_decomposition : ATen op → NPU ops（1对N裂解）
    ▼
④ op_absorption    : 独立算子吸收为相邻算子的参数
    ▼
⑤ format_annotator : 标注每个tensor的format/dtype
    ▼
⑥ validator        : 校验所有算子在C接口中有对应
    ▼
⑦ memory_planner   : HBM全局规划 + L1局部排列 + DMA计划
    ▼
⑧ scheduler        : 计算单元分配 + 依赖关系生成
    ▼
⑨ codegen          : 生成完整C工程
    ▼
输出：完整C工程目录
```

### 3.2 模块间依赖关系

```
common（必须最先完成，是所有模块的唯一依赖）
  │
  ├── graph_capture     ─┐
  ├── op_mapping        ─┤
  ├── op_decomposition  ─┤
  ├── op_absorption     ─┼── 全部只依赖common，互相无依赖，可完全并行开发
  ├── format_annotator  ─┤
  ├── validator         ─┤
  ├── memory_planner    ─┤
  ├── scheduler         ─┤
  └── codegen           ─┘
          │
          ▼
      integration（串联所有模块，最后开发）
```

---

## 4. 代码工程原则

### 4.1 简化代码

- 每个模块核心代码不超过300行
- 每个函数不超过50行
- 数据结构用dataclass，不用复杂继承
- 每个Pass是一个函数：`def run(graph: Graph, config: dict) -> Graph`
- C代码生成用f-string模板，不引入Jinja2

### 4.2 高复用

- 所有Pass共用同一套Graph IR数据结构
- 配置加载统一在config_loader.py
- 图遍历、拓扑排序等通用操作封装在Graph IR中
- C代码的三段式模板封装为一个函数

### 4.3 高可靠 + 完整日志

#### 日志规范

统一使用Python logging，不用print。所有模块通过`common/logger.py`获取logger。

```
格式：[时间] [级别] [模块名] 消息

级别规范：
  DEBUG   : 逐算子处理详情
  INFO    : Pass入口/出口统计
  WARNING : 可疑但不致命（如跳过无操作算子）
  ERROR   : 致命错误
```

#### 每个Pass的日志要求

| 位置 | 必须记录的内容 |
|------|---------------|
| 入口 | `[INFO] Pass开始，输入图: N个节点, M条边` |
| 过程 | `[DEBUG] 处理: aten.mm → npu_matmul (cube)` |
| 出口 | `[INFO] Pass完成，输出图: N个节点。统计: 映射X, 跳过Y, 裂解Z` |
| 异常 | `[ERROR] 算子 aten.xxx (node_id=5) 未找到映射规则` |

#### 错误处理规范

- 配置缺失：启动时立即报错，不延迟到运行中
- 算子映射缺失：合法性校验Pass统一报错，列出**所有**缺失算子
- 内存溢出：明确报告哪个tensor导致溢出
- 生成的C代码带源信息注释

---

## 5. Graph IR 数据结构契约

**这是所有模块共享的核心数据结构，必须最先定义并严格遵守。**

### 5.1 Python数据结构定义

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Tensor:
    """图中的边——表示一个tensor"""
    id: str                                    # 全局唯一ID，如 "tensor_0", "q_proj_weight"
    shape: list[int]                           # 如 [1, 32, 64]
    dtype: str                                 # "fp16" | "fp32" | ...
    format: str = "nd"                         # "nd" | "nz" | "nc1hwc0"

    # 以下字段由后续Pass填充
    hbm_offset: Optional[int] = None           # Pass⑦ memory_planner 填充
    hbm_size: Optional[int] = None             # Pass⑦ 填充（padding后的字节数）
    l1_offset: Optional[int] = None            # Pass⑦ 填充
    is_weight: bool = False                    # 是否为权重tensor
    is_model_input: bool = False               # 是否为模型的外部输入
    is_model_output: bool = False              # 是否为模型的最终输出
    producer_node_id: Optional[str] = None     # 产生此tensor的节点ID
    consumer_node_ids: list[str] = field(default_factory=list)  # 消费此tensor的节点ID列表

@dataclass
class Node:
    """图中的节点——表示一个算子"""
    id: str                                    # 全局唯一ID，如 "node_0", "node_1"
    op_type: str                               # 算子类型，如 "aten.mm" 或 "npu_matmul"
    inputs: list[str] = field(default_factory=list)    # 输入tensor的ID列表
    outputs: list[str] = field(default_factory=list)   # 输出tensor的ID列表
    params: dict = field(default_factory=dict)         # 算子参数，如 {"dim": -1, "epsilon": 1e-5}

    # 以下字段由后续Pass填充
    compute_unit: Optional[str] = None         # Pass② "cube" | "vector" | "scalar"
    npu_op: Optional[str] = None               # Pass② 映射后的NPU算子名
    is_mapped: bool = False                    # Pass② 是否已完成映射
    format_annotation: Optional[dict] = None   # Pass⑤ 输入输出的format/dtype标注
    schedule_order: Optional[int] = None       # Pass⑧ 执行顺序
    dependencies: list[str] = field(default_factory=list)  # Pass⑧ 依赖的节点ID列表

    # 吸收相关
    absorbed_inputs: dict = field(default_factory=dict)  # Pass④ 被吸收的额外输入
        # 例如 {"mask": "tensor_mask_0"} 表示该算子吸收了一个mask参数

@dataclass
class Graph:
    """计算图——由节点和tensor组成"""
    nodes: dict[str, Node] = field(default_factory=dict)    # node_id → Node
    tensors: dict[str, Tensor] = field(default_factory=dict) # tensor_id → Tensor
    execution_order: list[str] = field(default_factory=list) # 拓扑排序后的node_id列表

    # 图操作方法
    def add_node(self, node: Node) -> None: ...
    def remove_node(self, node_id: str) -> None: ...
    def add_tensor(self, tensor: Tensor) -> None: ...
    def remove_tensor(self, tensor_id: str) -> None: ...
    def get_node(self, node_id: str) -> Node: ...
    def get_tensor(self, tensor_id: str) -> Tensor: ...
    def topo_sort(self) -> list[str]: ...                    # 返回拓扑排序后的node_id列表
    def validate(self) -> list[str]: ...                     # 返回校验错误列表（空=通过）
    def to_dict(self) -> dict: ...                           # 序列化为dict（可JSON dump）
    @classmethod
    def from_dict(cls, data: dict) -> 'Graph': ...           # 从dict反序列化
    def summary(self) -> str: ...                            # 返回图的摘要信息
```

### 5.2 JSON序列化格式

**所有模块的demo_input_graph.json和expected_output.json使用此格式：**

```json
{
  "nodes": {
    "node_0": {
      "id": "node_0",
      "op_type": "aten.mm",
      "inputs": ["tensor_input", "tensor_weight"],
      "outputs": ["tensor_mm_out"],
      "params": {},
      "compute_unit": null,
      "npu_op": null,
      "is_mapped": false,
      "format_annotation": null,
      "schedule_order": null,
      "dependencies": [],
      "absorbed_inputs": {}
    }
  },
  "tensors": {
    "tensor_input": {
      "id": "tensor_input",
      "shape": [1, 32, 64],
      "dtype": "fp16",
      "format": "nd",
      "hbm_offset": null,
      "hbm_size": null,
      "l1_offset": null,
      "is_weight": false,
      "is_model_input": true,
      "is_model_output": false,
      "producer_node_id": null,
      "consumer_node_ids": ["node_0"]
    },
    "tensor_weight": {
      "id": "tensor_weight",
      "shape": [64, 64],
      "dtype": "fp16",
      "format": "nd",
      "is_weight": true,
      "is_model_input": false,
      "is_model_output": false,
      "producer_node_id": null,
      "consumer_node_ids": ["node_0"]
    },
    "tensor_mm_out": {
      "id": "tensor_mm_out",
      "shape": [1, 32, 64],
      "dtype": "fp16",
      "format": "nd",
      "is_weight": false,
      "is_model_input": false,
      "is_model_output": true,
      "producer_node_id": "node_0",
      "consumer_node_ids": []
    }
  },
  "execution_order": ["node_0"]
}
```

**字段规则：**

- 所有null字段由后续Pass填充，JSON中保留null表示"尚未处理"
- 每个模块的demo只需要关注自己填充的字段，其他字段保持null
- `from_dict()`对null字段容错，未填充的字段保持默认值

---

## 6. 模块文件夹结构

**每个模块一个独立文件夹，包含代码、配置、UT和局部demo，可被独立的AI Agent开发和测试。**

```
torch2c/
│
├── common/                           ← Agent 0（最先开发）
│   ├── README.md
│   ├── graph_ir.py
│   ├── logger.py
│   ├── config_loader.py
│   ├── errors.py
│   └── tests/
│       ├── test_graph_ir.py
│       └── test_config_loader.py
│
├── graph_capture/                    ← Agent 1
│   ├── README.md
│   ├── graph_capture.py
│   ├── config/
│   │   └── .gitkeep
│   ├── demo/
│   │   ├── demo_model.py
│   │   ├── run_demo.py
│   │   └── expected_output.json
│   └── tests/
│       └── test_graph_capture.py
│
├── op_mapping/                       ← Agent 1
│   ├── README.md
│   ├── op_mapping.py
│   ├── config/
│   │   └── direct_mappings.yaml
│   ├── demo/
│   │   ├── demo_input_graph.json
│   │   ├── run_demo.py
│   │   └── expected_output.json
│   └── tests/
│       └── test_op_mapping.py
│
├── op_decomposition/                 ← Agent 1
│   ├── README.md
│   ├── op_decomposition.py
│   ├── config/
│   │   └── decompositions.yaml
│   ├── demo/
│   │   ├── demo_input_graph.json
│   │   ├── run_demo.py
│   │   └── expected_output.json
│   └── tests/
│       └── test_op_decomposition.py
│
├── op_absorption/                    ← Agent 2
│   ├── README.md
│   ├── op_absorption.py
│   ├── config/
│   │   └── absorptions.yaml
│   ├── demo/
│   │   ├── demo_input_graph.json
│   │   ├── run_demo.py
│   │   └── expected_output.json
│   └── tests/
│       └── test_op_absorption.py
│
├── format_annotator/                 ← Agent 2
│   ├── README.md
│   ├── format_annotator.py
│   ├── config/
│   │   └── type_format_config.yaml
│   ├── demo/
│   │   ├── demo_input_graph.json
│   │   ├── run_demo.py
│   │   └── expected_output.json
│   └── tests/
│       └── test_format_annotator.py
│
├── validator/                        ← Agent 2
│   ├── README.md
│   ├── validator.py
│   ├── config/
│   │   └── supported_ops.yaml
│   ├── demo/
│   │   ├── demo_valid_graph.json
│   │   ├── demo_invalid_graph.json
│   │   ├── run_demo.py
│   │   └── expected_output.txt
│   └── tests/
│       └── test_validator.py
│
├── memory_planner/                   ← Agent 3
│   ├── README.md
│   ├── memory_planner.py
│   ├── config/
│   │   └── hardware_config.yaml
│   ├── demo/
│   │   ├── demo_input_graph.json
│   │   ├── run_demo.py
│   │   └── expected_output.json
│   └── tests/
│       └── test_memory_planner.py
│
├── scheduler/                        ← Agent 3
│   ├── README.md
│   ├── scheduler.py
│   ├── config/
│   │   └── .gitkeep
│   ├── demo/
│   │   ├── demo_input_graph.json
│   │   ├── run_demo.py
│   │   └── expected_output.json
│   └── tests/
│       └── test_scheduler.py
│
├── codegen/                          ← Agent 4
│   ├── README.md
│   ├── c_emitter.py
│   ├── weight_exporter.py
│   ├── golden_exporter.py
│   ├── utils_emitter.py
│   ├── mock_emitter.py
│   ├── cmake_emitter.py
│   ├── templates/
│   │   ├── op_block.c.tmpl
│   │   ├── main.c.tmpl
│   │   ├── data_loader.c.tmpl
│   │   ├── comparator.c.tmpl
│   │   └── test_memory.c.tmpl
│   ├── config/
│   │   ├── c_api_signatures.yaml
│   │   └── codegen_config.yaml
│   ├── demo/
│   │   ├── demo_input_plan.json
│   │   ├── run_demo.py
│   │   └── expected_files.txt
│   └── tests/
│       ├── test_c_emitter.py
│       └── test_utils_emitter.py
│
├── integration/                      ← Agent 0（最后开发）
│   ├── README.md
│   ├── pipeline.py
│   ├── config/
│   │   ├── direct_mappings.yaml
│   │   ├── decompositions.yaml
│   │   ├── absorptions.yaml
│   │   ├── c_api_signatures.yaml
│   │   ├── type_format_config.yaml
│   │   ├── hardware_config.yaml
│   │   ├── model_config.yaml
│   │   └── codegen_config.yaml
│   ├── demo/
│   │   ├── encoder_model.py
│   │   ├── run_full_demo.py
│   │   └── validate_output.py
│   └── tests/
│       └── test_pipeline.py
│
└── main.py
```

---

## 7. 各模块README详细规格

### 7.1 common/README.md

```
# common — 基础设施模块

## 职责
提供所有其他模块共享的基础设施：图IR数据结构、日志系统、配置加载、异常定义。

## 本模块必须最先开发
所有其他模块仅依赖common，互相无依赖。common完成后，其他模块可完全并行开发。

## 文件说明

### graph_ir.py
定义 Graph / Node / Tensor 三个核心dataclass。
提供图操作方法：add_node, remove_node, topo_sort, to_dict, from_dict, validate, summary。
详见本文档"第5节 Graph IR数据结构契约"。

### logger.py
统一日志系统。提供 get_logger(module_name) 函数。
日志格式：[时间] [级别] [模块名] 消息
支持通过环境变量 NPU_LOG_LEVEL 设置级别（默认INFO）。

接口：
  get_logger(name: str) -> logging.Logger
  setup_logging(level: str = "INFO", log_file: str = None) -> None

### config_loader.py
YAML配置加载与schema校验。
接口：
  load_config(path: str, required_keys: list[str] = None) -> dict
      加载YAML文件，校验必填字段存在，返回dict。
      缺失必填字段时抛出 ConfigError。

### errors.py
统一异常定义：
  CompilerError          — 基类
  ConfigError            — 配置文件错误
  MappingError           — 算子映射失败
  DecompositionError     — 算子裂解失败
  AbsorptionError        — 参数吸收失败
  ValidationError        — 合法性校验失败
  MemoryPlanError        — 内存编排失败
  CodegenError           — 代码生成失败

## UT
  test_graph_ir.py:
    - test_add_remove_node: 增删节点后节点数正确
    - test_topo_sort: 线性链的拓扑排序结果正确
    - test_to_dict_from_dict: 序列化→反序列化后图一致
    - test_validate: 悬空引用能被检测到
  test_config_loader.py:
    - test_load_valid: 正常YAML加载成功
    - test_missing_key: 缺少必填字段时抛出ConfigError
    - test_file_not_found: 文件不存在时抛出FileNotFoundError
```

---

### 7.2 graph_capture/README.md

```
# graph_capture — Pass①：图捕获

## 职责
使用 torch.export 将 PyTorch nn.Module 导出为 ATen IR 图，转换为 Graph IR。

## 输入
- model: torch.nn.Module
- dummy_input: torch.Tensor（固定shape的样例输入）

## 输出
- Graph IR（所有节点的op_type为ATen算子名，如"aten.mm"、"aten.add.Tensor"）
- 权重tensor标记 is_weight=True
- 模型输入/输出tensor标记 is_model_input/is_model_output=True

## 接口
  def capture(model: nn.Module, dummy_input: torch.Tensor) -> Graph

## 日志
  INFO: "图捕获完成，节点数: N, tensor数: M, 权重tensor数: W"
  DEBUG: 每个节点的算子类型和输入输出tensor

## demo/
  demo_model.py:
    定义2层Encoder Transformer的PyTorch模型（小shape版）。
    hidden_size=64, num_heads=4, ffn_dim=256, seq_len=32, batch=1。

  run_demo.py:
    from demo_model import TransformerEncoder
    model = TransformerEncoder()
    dummy = torch.randn(1, 32, 64)
    graph = capture(model, dummy)
    print(graph.summary())
    # 保存图为JSON供后续模块demo使用
    json.dump(graph.to_dict(), open("captured_graph.json", "w"))

  expected_output.json:
    预期的算子类型清单（不需要完全精确，重点验证关键算子存在）：
    {
      "expected_op_types": [
        "aten.mm", "aten.add.Tensor", "aten.layer_norm",
        "aten._softmax", "aten.gelu", "aten.mul.Scalar",
        "aten.transpose.int", "aten.reshape", "aten.t"
      ],
      "expected_node_count_range": [40, 80],
      "expected_weight_tensor_count_range": [20, 40]
    }

## UT
  test_graph_capture.py:
    - test_capture_linear: 单个nn.Linear导出，验证有mm节点
    - test_capture_encoder: 完整encoder导出，验证关键算子类型存在
    - test_weight_marking: 验证权重tensor的is_weight=True
    - test_io_marking: 验证输入输出tensor的标记正确
```

---

### 7.3 op_mapping/README.md

```
# op_mapping — Pass②：算子直接映射

## 职责
将ATen算子名替换为NPU算子名（1对1直接映射），标注compute_unit。

## 输入
- Graph IR（节点的op_type为ATen名）
- config/direct_mappings.yaml

## 输出
- Graph IR（已映射的节点：npu_op被填充，is_mapped=True，compute_unit被填充）
- 未映射的节点保持原样（留给op_decomposition或validator处理）

## 接口
  def run(graph: Graph, config: dict) -> Graph

## 处理逻辑
  for node in graph.nodes:
      if node.op_type in config.mappings:
          node.npu_op = config.mappings[node.op_type].npu_op
          node.compute_unit = config.mappings[node.op_type].compute_unit
          node.is_mapped = True
      else:
          # 跳过，留给后续Pass处理
          log.debug(f"未映射: {node.op_type}，留给裂解或校验")

## 日志
  INFO: "映射完成。已映射: X, 未映射: Y"
  DEBUG: 每个算子的映射结果

## config/direct_mappings.yaml
（本模块局部配置，demo版）

  mappings:
    - aten_op: "aten.mm"
      npu_op: "npu_matmul"
      compute_unit: "cube"
    - aten_op: "aten.add.Tensor"
      npu_op: "npu_add"
      compute_unit: "vector"
    - aten_op: "aten.mul.Tensor"
      npu_op: "npu_mul"
      compute_unit: "vector"
    - aten_op: "aten.mul.Scalar"
      npu_op: "npu_mul_scalar"
      compute_unit: "vector"
    - aten_op: "aten.gelu"
      npu_op: "npu_gelu"
      compute_unit: "vector"
    - aten_op: "aten.transpose.int"
      npu_op: "npu_transpose"
      compute_unit: "vector"
    - aten_op: "aten.t"
      npu_op: "npu_transpose_2d"
      compute_unit: "vector"
    - aten_op: "aten.reshape"
      npu_op: "npu_reshape"
      compute_unit: "scalar"

## demo/demo_input_graph.json
  含5个节点的小图：mm → add → mul → gelu → reshape
  （使用第5节定义的JSON格式）

## demo/expected_output.json
  映射后5个节点的npu_op分别为：
  npu_matmul, npu_add, npu_mul, npu_gelu, npu_reshape

## UT
  test_op_mapping.py:
    - test_basic_mapping: 5个算子全部映射成功
    - test_unmapped_preserved: 含一个不在配置中的算子，验证它保持未映射
    - test_compute_unit: 验证matmul→cube, add→vector
    - test_empty_graph: 空图不报错
```

---

### 7.4 op_decomposition/README.md

```
# op_decomposition — Pass③：算子裂解

## 职责
将未映射的ATen算子按裂解规则替换为多个NPU算子节点（固定成组）。

## 输入
- Graph IR（经过op_mapping，部分节点已映射，部分未映射）
- config/decompositions.yaml

## 输出
- Graph IR（裂解后的节点替换原节点，中间tensor被创建）

## 接口
  def run(graph: Graph, config: dict) -> Graph

## 处理逻辑
  for node in graph.nodes (未映射的):
      if node.op_type in config.decompositions:
          rule = config.decompositions[node.op_type]
          # 1. 创建中间tensor
          # 2. 创建多个新节点，按order排列
          # 3. 连接输入输出（from: "source.input_N" 引用原节点输入）
          # 4. 删除原节点
          # 5. 标记新节点 is_mapped=True

## 日志
  INFO: "裂解完成。裂解了X个算子，新增Y个节点，新增Z个中间tensor"

## config/decompositions.yaml
（本模块局部配置，demo版）

  decompositions:
    - name: "layernorm_decompose"
      source_op: "aten.layer_norm"
      target_ops:
        - order: 1
          npu_op: "npu_layernorm_part1"
          compute_unit: "vector"
          inputs:
            - from: "source.input_0"
            - from: "source.input_1"
            - from: "source.input_2"
          outputs:
            - id: "layernorm_intermediate"
              shape_same_as: "source.input_0"
              dtype_same_as: "source.input_0"
          params_from_source:
            - { name: "epsilon", source_param: "eps" }
        - order: 2
          npu_op: "npu_layernorm_part2"
          compute_unit: "vector"
          inputs:
            - from: "layernorm_intermediate"
            - from: "source.input_0"
          outputs:
            - id: "source.output_0"

    - name: "softmax_decompose"
      source_op: "aten._softmax"
      target_ops:
        - order: 1
          npu_op: "npu_softmax_part1"
          compute_unit: "vector"
          inputs:
            - from: "source.input_0"
          outputs:
            - id: "softmax_intermediate"
              shape_same_as: "source.input_0"
              dtype_same_as: "source.input_0"
          params_from_source:
            - { name: "dim", source_param: "dim" }
        - order: 2
          npu_op: "npu_softmax_part2"
          compute_unit: "vector"
          inputs:
            - from: "softmax_intermediate"
          outputs:
            - id: "source.output_0"

## demo/demo_input_graph.json
  含3个节点：layer_norm → mm → softmax
  其中layer_norm和softmax未映射，mm已映射

## demo/expected_output.json
  裂解后：layernorm_part1 → layernorm_part2 → mm → softmax_part1 → softmax_part2
  节点数从3变为5，新增2个中间tensor

## UT
  test_op_decomposition.py:
    - test_layernorm_decompose: layer_norm裂解为2个part
    - test_softmax_decompose: softmax裂解为2个part
    - test_intermediate_tensor_created: 中间tensor存在且shape正确
    - test_already_mapped_skipped: 已映射的节点不被裂解
    - test_no_rule_preserved: 没有裂解规则的未映射节点保持原样
```

---

### 7.5 op_absorption/README.md

```
# op_absorption — Pass④：参数吸收

## 职责
将独立的算子（如mask的add）吸收为相邻算子的可选参数。

## 输入
- Graph IR（经过mapping和decomposition，所有节点都已映射）
- config/absorptions.yaml

## 输出
- Graph IR（被吸收的节点被删除，目标节点的absorbed_inputs被填充）

## 接口
  def run(graph: Graph, config: dict) -> Graph

## 处理逻辑
  for rule in config.absorptions:
      # 在图中寻找匹配pattern：
      #   1. 找到absorbed_op类型的节点
      #   2. 检查它的输出是否只有一个消费者
      #   3. 检查消费者是否是target_op类型
      #   4. 匹配成功：
      #      - 将被吸收节点的指定输入添加到目标节点的absorbed_inputs
      #      - 重连：目标节点的对应输入改为被吸收节点的非mask输入
      #      - 删除被吸收节点及其输出tensor

## 日志
  INFO: "吸收完成。吸收了X个算子，消除了Y个中间tensor"

## config/absorptions.yaml
（本模块局部配置，demo版）

  absorptions:
    - name: "softmax_absorb_mask"
      absorbed_op: "npu_add"
      target_op: "npu_softmax_part1"
      conditions:
        position: "immediately_before_target"
        absorbed_output_is_target_input: true
      param_mapping:
        absorbed_input_index: 1
        target_param_name: "mask"
      eliminates_intermediate: true

## demo/demo_input_graph.json
  含3个节点：add(scores, mask) → softmax_part1 → softmax_part2
  add的输入0是scores，输入1是mask

## demo/expected_output.json
  吸收后：节点数从3变为2（add被删除）
  softmax_part1的absorbed_inputs = {"mask": "tensor_mask"}
  softmax_part1的输入从add的输出变为add原来的输入0（scores）

## UT
  test_op_absorption.py:
    - test_mask_absorption: add被吸收进softmax_part1
    - test_absorbed_input_recorded: softmax_part1的absorbed_inputs包含mask
    - test_intermediate_removed: add的输出tensor被删除
    - test_no_match_preserved: 不匹配规则的add保持不变（如残差add）
    - test_empty_rules: absorptions为空列表时图不变
```

---

### 7.6 format_annotator/README.md

```
# format_annotator — Pass⑤：Format/Dtype标注

## 职责
根据每个NPU算子的format/dtype要求，标注每个tensor的preferred format和dtype。

## 输入
- Graph IR（所有节点已映射为NPU算子）
- config/type_format_config.yaml

## 输出
- Graph IR（每个节点的format_annotation被填充）

## 接口
  def run(graph: Graph, config: dict) -> Graph

## 处理逻辑
  for node in graph.nodes:
      req = config.op_format_requirements[node.npu_op]
      node.format_annotation = {
          "inputs": [{"format": r.format, "dtype": r.dtype} for r in req.inputs],
          "outputs": [{"format": r.format, "dtype": r.dtype} for r in req.outputs],
          "supports_format_convert": req.supports_format_convert,
          "supports_dtype_cast": req.supports_dtype_cast
      }
      # 同时更新tensor本身的format/dtype为该算子期望的值
      for i, tensor_id in enumerate(node.inputs):
          if i < len(req.inputs):
              graph.tensors[tensor_id].format = req.inputs[i].format
              graph.tensors[tensor_id].dtype = req.inputs[i].dtype

## 日志
  INFO: "Format标注完成。标注了N个节点，M个tensor"

## config/type_format_config.yaml
（本模块局部配置，demo版，含枚举定义和算子要求）
  见第5.5节的完整定义

## demo/demo_input_graph.json
  含2个节点：matmul(cube) → add(vector)
  所有tensor初始format=nd, dtype=fp16

## demo/expected_output.json
  matmul的输入tensor标注为format=nz, dtype=fp16
  add的输入tensor标注为format=nd, dtype=fp16

## UT
  test_format_annotator.py:
    - test_matmul_format: matmul输入标注为nz
    - test_vector_format: add输入标注为nd
    - test_annotation_structure: format_annotation字段结构正确
```

---

### 7.7 validator/README.md

```
# validator — Pass⑥：合法性校验

## 职责
验证所有节点都已映射到支持的NPU算子。

## 输入
- Graph IR（经过mapping、decomposition、absorption）
- config/supported_ops.yaml

## 输出
- 通过：返回原图不变
- 失败：抛出ValidationError，包含所有未支持算子的列表

## 接口
  def run(graph: Graph, config: dict) -> Graph

## 处理逻辑
  unsupported = []
  for node in graph.nodes:
      if node.npu_op not in config.supported_ops:
          unsupported.append(f"{node.id}: {node.op_type} (npu_op={node.npu_op})")
  if unsupported:
      raise ValidationError(f"以下算子未映射: {unsupported}")

## config/supported_ops.yaml

  supported_ops:
    - "npu_matmul"
    - "npu_add"
    - "npu_mul"
    - "npu_mul_scalar"
    - "npu_gelu"
    - "npu_layernorm_part1"
    - "npu_layernorm_part2"
    - "npu_softmax_part1"
    - "npu_softmax_part2"
    - "npu_transpose"
    - "npu_transpose_2d"
    - "npu_reshape"

## demo/
  demo_valid_graph.json: 全部节点都在支持列表中 → 通过
  demo_invalid_graph.json: 含一个"npu_unknown"节点 → 报错

## UT
  test_validator.py:
    - test_all_supported: 全部通过
    - test_one_unsupported: 报错且错误信息包含具体算子名
    - test_multiple_unsupported: 报错且列出所有未支持算子（不只第一个）
```

---

### 7.8 memory_planner/README.md

```
# memory_planner — Pass⑦：内存编排

## 职责
为所有tensor分配HBM偏移地址，为每个算子规划L1布局，生成DMA搬运计划。

## 输入
- Graph IR（校验通过，所有节点已映射）
- config/hardware_config.yaml

## 输出
- Graph IR（每个tensor的hbm_offset, hbm_size, l1_offset被填充）
- 附加数据：DMA计划列表（每个算子的load/store指令序列）

## 接口
  def run(graph: Graph, config: dict) -> tuple[Graph, list[DmaPlan]]

  @dataclass
  class DmaInstruction:
      op: str                    # "load" | "store"
      tensor_id: str
      hbm_offset: int
      l1_offset: int
      size_bytes: int
      src_format: str
      dst_format: str

  @dataclass
  class DmaPlan:
      node_id: str
      loads: list[DmaInstruction]
      stores: list[DmaInstruction]

## 处理逻辑

### HBM层（全局）
  1. 计算每个tensor的padded_size（分形格式对齐）
  2. 分析生命周期：first_use = min(消费者的执行顺序), last_use = max(消费者的执行顺序)
  3. 按first_use排序，best-fit分配HBM偏移
  4. last_use之后的tensor空间标记为可复用
  5. 地址按hardware_config.memory.hbm.alignment_bytes对齐

### L1层（每个算子独立）
  1. 按顺序排列：输入tensor → 权重tensor → 输出tensor
  2. 每个tensor起始地址按hardware_config.memory.l1.alignment_bytes对齐
  3. 检查总用量不超过L1容量

### DMA计划
  每个算子生成固定三段：
  1. load: 所有输入和权重从HBM→L1
  2. （算子执行，不在DMA计划中）
  3. store: 所有输出从L1→HBM

### 关键函数
  calc_padded_size(shape, dtype, format, cube_size) -> int
      计算padding后的字节数。分形格式需将shape的最后两维对齐到cube_size的整数倍。

  align_up(offset, alignment) -> int
      向上对齐。

## config/hardware_config.yaml
  见第5.6节完整定义

## demo/demo_input_graph.json
  5个算子的线性链：A→B→C→D→E
  每个算子有1个输入和1个输出
  tensor shape统一为[1,32,64], dtype=fp16

## demo/expected_output.json
  {
    "hbm_total_used": N,
    "tensor_offsets": {
      "tensor_0": {"hbm_offset": 0, "hbm_size": 4096},
      "tensor_1": {"hbm_offset": 4096, "hbm_size": 4096},
      ...
    },
    "reuse_count": M,
    "assertions": [
      "所有offset >= 0",
      "相邻活跃tensor的区间不重叠",
      "所有offset满足512B对齐"
    ]
  }

## UT
  test_memory_planner.py:
    - test_no_overlap: 同时活跃的tensor地址不重叠
    - test_alignment: 所有HBM offset是512的倍数，L1 offset是32的倍数
    - test_reuse: 线性链中dead tensor的空间被后续tensor复用
    - test_padded_size: calc_padded_size对非对齐shape正确padding
    - test_dma_plan: 每个算子有正确数量的load和store指令
    - test_l1_capacity_check: L1溢出时抛出MemoryPlanError
```

---

### 7.9 scheduler/README.md

```
# scheduler — Pass⑧：计算单元调度与依赖生成

## 职责
确定算子执行顺序，生成算子间的依赖关系。

## 输入
- Graph IR（已编排完成，每个节点有compute_unit）

## 输出
- Graph IR（每个节点的schedule_order和dependencies被填充）

## 接口
  def run(graph: Graph, config: dict = None) -> Graph

## 处理逻辑（保守策略）
  1. 拓扑排序确定基本执行顺序
  2. 遍历相邻算子对：
     - 如果有数据依赖（后者的输入是前者的输出）→ 插入依赖
     - 如果无数据依赖且使用不同compute_unit → 可并行（不插入依赖）
     - 如果无数据依赖但使用相同compute_unit → 串行（插入依赖）
  3. DMA操作后始终插入barrier

## 日志
  INFO: "调度完成。依赖关系: N条，可并行算子对: M"

## demo/demo_input_graph.json
  4个算子：matmul(cube) → add(vector) → matmul(cube) → gelu(vector)
  matmul→add有数据依赖，add→matmul有数据依赖

## demo/expected_output.json
  所有4个算子串行（因为有数据依赖链），依赖关系: [(0,1), (1,2), (2,3)]

## UT
  test_scheduler.py:
    - test_linear_chain: 线性依赖链的依赖关系正确
    - test_parallel_opportunity: 两个无依赖的不同compute_unit算子不产生依赖
    - test_same_unit_serialized: 两个无依赖的相同compute_unit算子串行化
    - test_schedule_order: 每个节点的schedule_order按拓扑排序递增
```

---

### 7.10 codegen/README.md

```
# codegen — Pass⑨：C代码生成

## 职责
根据编排结果和配置，生成完整的可编译C工程。

## 输入
- Graph IR（完整编排完成）
- DMA计划列表
- config/c_api_signatures.yaml
- config/codegen_config.yaml
- 权重数据（numpy数组）
- golden数据（PyTorch跑出的输入输出）

## 输出
- 完整C工程目录（见第6节目录结构）

## 文件说明

### c_emitter.py
生成model_graph.c/h — 主执行逻辑。
每个算子生成三段式代码块：DMA搬入 → 算子调用 → DMA搬出。
算子调用参数根据c_api_signatures.yaml的param_source自动提取。

### weight_exporter.py
将PyTorch权重导出为C静态数组（model_weights.h）。
接口: def export_weights(state_dict: dict, output_path: str, dtype: str)

### golden_exporter.py
将PyTorch的输入输出导出为二进制文件 + 描述文件。
接口: def export_golden(inputs, outputs, output_dir: str)

### utils_emitter.py
生成辅助工具C代码：data_loader, data_dumper, comparator。

### mock_emitter.py
根据c_api_signatures.yaml和type_format_config.yaml生成npu_mock.h。

### cmake_emitter.py
生成CMakeLists.txt，支持mock模式和真实SDK模式。

## 模板使用
templates/目录下的.tmpl文件是C代码片段模板，使用Python f-string插值。
模板中的占位符：
  {op_id}         — 算子编号
  {npu_op}        — NPU算子名
  {compute_unit}  — 计算单元名
  {params}        — 展开的参数列表
  {l1_offset}     — L1偏移
  {hbm_offset}    — HBM偏移
  {size}          — 搬运大小
  {src_fmt}       — 源format枚举
  {dst_fmt}       — 目标format枚举

## demo/demo_input_plan.json
  3个算子的完整编排结果（含HBM/L1偏移、DMA计划、依赖关系）

## demo/run_demo.py
  加载demo_input_plan.json，生成C工程到demo_output/目录
  然后执行 gcc -fsyntax-only -include npu_mock.h src/model_graph.c 验证语法

## UT
  test_c_emitter.py:
    - test_op_block_generation: 单个算子生成正确的三段式代码
    - test_param_filling: 参数根据signature正确填入
    - test_syntax_check: 生成的C代码通过gcc语法检查
  test_utils_emitter.py:
    - test_comparator_generation: comparator.c语法正确
    - test_data_loader_generation: data_loader.c语法正确
```

---

## 8. 配置文件完整定义

### 8.1 direct_mappings.yaml

```yaml
# 算子直接映射：1个ATen → 1个NPU
mappings:
  - aten_op: "aten.mm"
    npu_op: "npu_matmul"
    compute_unit: "cube"

  - aten_op: "aten.add.Tensor"
    npu_op: "npu_add"
    compute_unit: "vector"

  - aten_op: "aten.mul.Tensor"
    npu_op: "npu_mul"
    compute_unit: "vector"

  - aten_op: "aten.mul.Scalar"
    npu_op: "npu_mul_scalar"
    compute_unit: "vector"

  - aten_op: "aten.gelu"
    npu_op: "npu_gelu"
    compute_unit: "vector"

  - aten_op: "aten.transpose.int"
    npu_op: "npu_transpose"
    compute_unit: "vector"

  - aten_op: "aten.t"
    npu_op: "npu_transpose_2d"
    compute_unit: "vector"

  - aten_op: "aten.reshape"
    npu_op: "npu_reshape"
    compute_unit: "scalar"
```

### 8.2 decompositions.yaml

```yaml
# 算子裂解：1个ATen → N个NPU（固定成组）
decompositions:
  - name: "layernorm_decompose"
    source_op: "aten.layer_norm"
    target_ops:
      - order: 1
        npu_op: "npu_layernorm_part1"
        compute_unit: "vector"
        inputs:
          - from: "source.input_0"
          - from: "source.input_1"
          - from: "source.input_2"
        outputs:
          - id: "layernorm_intermediate"
            shape_same_as: "source.input_0"
            dtype_same_as: "source.input_0"
        params_from_source:
          - { name: "epsilon", source_param: "eps" }
      - order: 2
        npu_op: "npu_layernorm_part2"
        compute_unit: "vector"
        inputs:
          - from: "layernorm_intermediate"
          - from: "source.input_0"
        outputs:
          - id: "source.output_0"

  - name: "softmax_decompose"
    source_op: "aten._softmax"
    target_ops:
      - order: 1
        npu_op: "npu_softmax_part1"
        compute_unit: "vector"
        inputs:
          - from: "source.input_0"
        outputs:
          - id: "softmax_intermediate"
            shape_same_as: "source.input_0"
            dtype_same_as: "source.input_0"
        params_from_source:
          - { name: "dim", source_param: "dim" }
      - order: 2
        npu_op: "npu_softmax_part2"
        compute_unit: "vector"
        inputs:
          - from: "softmax_intermediate"
        outputs:
          - id: "source.output_0"
```

### 8.3 absorptions.yaml

```yaml
# 参数吸收：独立算子 → 相邻算子的可选参数
absorptions:
  - name: "softmax_absorb_mask"
    absorbed_op: "npu_add"
    target_op: "npu_softmax_part1"
    conditions:
      position: "immediately_before_target"
      absorbed_output_is_target_input: true
    param_mapping:
      absorbed_input_index: 1
      target_param_name: "mask"
    eliminates_intermediate: true
```

### 8.4 c_api_signatures.yaml

```yaml
# NPU C接口函数签名
compute_ops:

  npu_matmul:
    params:
      - { name: "a",      type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "b",      type: "addr",  source: "tensor.input_1.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "M",      type: "int",   source: "tensor.input_0.shape.0" }
      - { name: "N",      type: "int",   source: "tensor.input_1.shape.1" }
      - { name: "K",      type: "int",   source: "tensor.input_0.shape.1" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }
      - { name: "fmt",    type: "enum",  source: "tensor.input_0.format" }
    optional_params:
      - { name: "mask",   type: "addr",  source: "tensor.mask.l1_offset", default: "NULL" }

  npu_add:
    params:
      - { name: "a",      type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "b",      type: "addr",  source: "tensor.input_1.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }

  npu_mul:
    params:
      - { name: "a",      type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "b",      type: "addr",  source: "tensor.input_1.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }

  npu_mul_scalar:
    params:
      - { name: "input",  type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "scalar", type: "float", source: "param.scalar_value" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }

  npu_gelu:
    params:
      - { name: "input",  type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }

  npu_layernorm_part1:
    params:
      - { name: "input",    type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "gamma",    type: "addr",  source: "tensor.input_1.l1_offset" }
      - { name: "beta",     type: "addr",  source: "tensor.input_2.l1_offset" }
      - { name: "out",      type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "hidden",   type: "int",   source: "tensor.input_0.shape.-1" }
      - { name: "seq",      type: "int",   source: "tensor.input_0.shape.-2" }
      - { name: "eps",      type: "float", source: "param.epsilon" }
      - { name: "dtype",    type: "enum",  source: "tensor.input_0.dtype" }

  npu_layernorm_part2:
    params:
      - { name: "inter",    type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "orig",     type: "addr",  source: "tensor.input_1.l1_offset" }
      - { name: "out",      type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "hidden",   type: "int",   source: "tensor.input_0.shape.-1" }
      - { name: "dtype",    type: "enum",  source: "tensor.input_0.dtype" }

  npu_softmax_part1:
    params:
      - { name: "input",  type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "dim",    type: "int",   source: "param.dim" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }
    optional_params:
      - { name: "mask",   type: "addr",  source: "tensor.mask.l1_offset", default: "NULL" }

  npu_softmax_part2:
    params:
      - { name: "inter",  type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }

  npu_transpose:
    params:
      - { name: "input",  type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "dim0",   type: "int",   source: "param.dim0" }
      - { name: "dim1",   type: "int",   source: "param.dim1" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }

  npu_transpose_2d:
    params:
      - { name: "input",  type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "rows",   type: "int",   source: "tensor.input_0.shape.0" }
      - { name: "cols",   type: "int",   source: "tensor.input_0.shape.1" }
      - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }

  npu_reshape:
    params:
      - { name: "input",  type: "addr",  source: "tensor.input_0.l1_offset" }
      - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
      - { name: "count",  type: "int",   source: "tensor.input_0.elem_count" }

dma_ops:
  npu_dma_load:
    params:
      - { name: "l1_dst",    type: "addr" }
      - { name: "hbm_src",   type: "addr" }
      - { name: "size",      type: "int" }
      - { name: "src_fmt",   type: "enum", default: "NPU_FORMAT_ND" }
      - { name: "dst_fmt",   type: "enum", default: "NPU_FORMAT_ND" }
  npu_dma_store:
    params:
      - { name: "hbm_dst",   type: "addr" }
      - { name: "l1_src",    type: "addr" }
      - { name: "size",      type: "int" }
      - { name: "src_fmt",   type: "enum", default: "NPU_FORMAT_ND" }
      - { name: "dst_fmt",   type: "enum", default: "NPU_FORMAT_ND" }
  npu_dma_barrier:
    params: []

sync_ops:
  npu_set_dependency:
    params:
      - { name: "src_id",    type: "int" }
      - { name: "dst_id",    type: "int" }
  npu_barrier:
    params: []
```

### 8.5 type_format_config.yaml

```yaml
dtype_enum:
  fp16: "NPU_DTYPE_FP16"
  fp32: "NPU_DTYPE_FP32"
  bf16: "NPU_DTYPE_BF16"
  int8: "NPU_DTYPE_INT8"
  int32: "NPU_DTYPE_INT32"

dtype_bytes:
  fp16: 2
  fp32: 4
  bf16: 2
  int8: 1
  int32: 4

format_enum:
  nd: "NPU_FORMAT_ND"
  nz: "NPU_FORMAT_NZ"
  nc1hwc0: "NPU_FORMAT_NC1HWC0"

op_format_requirements:
  npu_matmul:
    inputs:  [{ format: "nz", dtype: "fp16" }, { format: "nz", dtype: "fp16" }]
    outputs: [{ format: "nz", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_add:
    inputs:  [{ format: "nd", dtype: "fp16" }, { format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_mul:
    inputs:  [{ format: "nd", dtype: "fp16" }, { format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_mul_scalar:
    inputs:  [{ format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_gelu:
    inputs:  [{ format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_layernorm_part1:
    inputs:  [{ format: "nd", dtype: "fp32" }, { format: "nd", dtype: "fp32" }, { format: "nd", dtype: "fp32" }]
    outputs: [{ format: "nd", dtype: "fp32" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_layernorm_part2:
    inputs:  [{ format: "nd", dtype: "fp32" }, { format: "nd", dtype: "fp32" }]
    outputs: [{ format: "nd", dtype: "fp32" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_softmax_part1:
    inputs:  [{ format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_softmax_part2:
    inputs:  [{ format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_transpose:
    inputs:  [{ format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_transpose_2d:
    inputs:  [{ format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
  npu_reshape:
    inputs:  [{ format: "nd", dtype: "fp16" }]
    outputs: [{ format: "nd", dtype: "fp16" }]
    supports_format_convert: true
    supports_dtype_cast: true
```

### 8.6 hardware_config.yaml

```yaml
memory:
  hbm:
    total_size_bytes: 4294967296
    alignment_bytes: 512
  l2:
    total_size_bytes: 33554432
    alignment_bytes: 256
  l1:
    total_size_bytes: 16777216
    alignment_bytes: 32

fractal:
  cube_size: 16
  c0_by_dtype:
    fp16: 16
    int8: 32

dma:
  supports_format_convert: true
  supports_dtype_cast: false
  max_transfer_size_bytes: 1048576
```

### 8.7 model_config.yaml

```yaml
model:
  name: "transformer_encoder_2layer_demo"
  type: "transformer_encoder"
  num_layers: 2
  hidden_size: 64
  num_attention_heads: 4
  intermediate_size: 256
  layer_norm_epsilon: 1.0e-5
  activation: "gelu"
  input_shape:
    batch_size: 1
    seq_len: 32
  compute_dtype: "fp16"
  weight_dtype: "fp16"

weights:
  export_format: "static_array"
  header_file: "model_weights.h"

golden:
  export_input: true
  export_output: true
  export_intermediate: false
  output_dir: "golden/"
  data_format: "raw_binary"
```

### 8.8 codegen_config.yaml

```yaml
codegen:
  output_dir: "output/"
  debug:
    insert_dump_points: false
    dump_guard_macro: "NPU_DEBUG_DUMP"
    print_op_names: true
  scheduling:
    strategy: "conservative"
  style:
    indent: "    "
    include_comments: true
```

---

## 9. 生成的C工程规格

### 9.1 目录结构

```
output/
├── CMakeLists.txt
├── npu_mock.h
├── src/
│   ├── model_graph.c
│   ├── model_graph.h
│   ├── model_memory.h
│   ├── model_params.h
│   └── model_weights.h
├── utils/
│   ├── data_loader.c / .h
│   ├── data_dumper.c / .h
│   ├── comparator.c / .h
│   └── tensor_utils.h
├── golden/
│   ├── input_0.bin / .desc
│   └── output_0.bin / .desc
├── tests/
│   ├── test_data_loader.c
│   ├── test_comparator.c
│   ├── test_memory_layout.c
│   └── CMakeLists.txt
├── main.c
└── README.md
```

### 9.2 CMakeLists.txt规格

```cmake
cmake_minimum_required(VERSION 3.10)
project(npu_model C)

# 用户配置：NPU SDK路径
set(NPU_SDK_PATH "" CACHE PATH "Path to NPU SDK")
if(NOT NPU_SDK_PATH)
    set(NPU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    message(WARNING "Using mock headers (compile-only)")
else()
    set(NPU_INCLUDE_DIR ${NPU_SDK_PATH}/include)
    link_directories(${NPU_SDK_PATH}/lib)
endif()

set(CMAKE_C_STANDARD 99)
add_compile_options(-Wall -Wextra)

option(NPU_DEBUG_DUMP "Enable debug dump" OFF)
if(NPU_DEBUG_DUMP)
    add_definitions(-DNPU_DEBUG_DUMP)
endif()

# 主程序
add_executable(npu_model_run
    main.c src/model_graph.c
    utils/data_loader.c utils/data_dumper.c utils/comparator.c)
target_include_directories(npu_model_run PRIVATE
    ${NPU_INCLUDE_DIR} src utils)
if(NPU_SDK_PATH)
    target_link_libraries(npu_model_run npu_runtime)
endif()

# 单元测试
enable_testing()
add_executable(test_data_loader tests/test_data_loader.c utils/data_loader.c)
target_include_directories(test_data_loader PRIVATE utils)
add_test(NAME test_data_loader COMMAND test_data_loader)

add_executable(test_comparator tests/test_comparator.c utils/comparator.c)
target_include_directories(test_comparator PRIVATE utils)
add_test(NAME test_comparator COMMAND test_comparator)

add_executable(test_memory_layout tests/test_memory_layout.c)
target_include_directories(test_memory_layout PRIVATE src)
add_test(NAME test_memory_layout COMMAND test_memory_layout)
```

### 9.3 npu_mock.h规格

```c
// 自动生成 — 仅用于编译验证，不含实现
#ifndef NPU_MOCK_H
#define NPU_MOCK_H
#include <stddef.h>
#include <stdint.h>

typedef enum { NPU_DTYPE_FP16=0, NPU_DTYPE_FP32, NPU_DTYPE_BF16, NPU_DTYPE_INT8, NPU_DTYPE_INT32 } npu_dtype_t;
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
void npu_transpose(void* input, void* out, int dim0, int dim1, int count, npu_dtype_t dtype);
void npu_transpose_2d(void* input, void* out, int rows, int cols, npu_dtype_t dtype);
void npu_reshape(void* input, void* out, int count);

void npu_dma_load(void* l1_dst, void* hbm_src, int size, npu_format_t src, npu_format_t dst);
void npu_dma_store(void* hbm_dst, void* l1_src, int size, npu_format_t src, npu_format_t dst);
void npu_dma_barrier(void);
void npu_set_dependency(int src_id, int dst_id);
void npu_barrier(void);

#ifdef NPU_DEBUG_DUMP
#include <stdio.h>
#define NPU_LOG(fmt, ...) printf("[NPU] " fmt "\n", ##__VA_ARGS__)
#else
#define NPU_LOG(fmt, ...) ((void)0)
#endif

#endif
```

### 9.4 C侧UT框架

```c
// 统一的测试框架宏（所有test_*.c文件使用）
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

static int _test_count = 0;
static int _pass_count = 0;

#define RUN_TEST(fn) do { \
    _test_count++; \
    printf("  [RUN] %s ... ", #fn); \
    fn(); \
    _pass_count++; \
    printf("PASS\n"); \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAIL\n    %s:%d: %d != %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        exit(1); \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL\n    %s:%d: condition false\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define TEST_SUMMARY() printf("\n=== %d/%d tests passed ===\n", _pass_count, _test_count)
```

### 9.5 data_loader规格

```c
// 接口
typedef struct {
    int shape[8];        // 最多8维
    int ndim;
    char dtype[16];      // "fp16", "fp32", ...
    char format[16];     // "nd", "nz", ...
    size_t total_bytes;
} tensor_desc_t;

int parse_desc(const char* desc_path, tensor_desc_t* desc);
int load_tensor(void* hbm_base, size_t offset,
                const char* bin_path, const char* desc_path);
// 返回值：0=成功, -1=文件不存在, -2=大小不匹配, -3=读取失败
```

### 9.6 comparator规格

```c
typedef struct {
    float max_abs_diff;
    float max_rel_diff;
    float cosine_similarity;
    float mse;
    int mismatch_count;
    int total_elements;
    int first_mismatch_index;
} compare_result_t;

int compare_tensors(const char* actual_path, const char* golden_path,
                    const char* desc_path, float abs_tol, float cos_tol,
                    compare_result_t* result);
// 返回值：0=通过, 1=失败
// result中填充详细的比对统计
```

### 9.7 desc文件格式

```
shape: 1,32,64
dtype: fp16
format: nd
byte_order: little_endian
total_bytes: 4096
```

---

## 10. Demo规格：2层Encoder Transformer

### 10.1 模型结构

```
Input [1, 32, 64]
│
├── Layer 1:
│   ├── LayerNorm  → layernorm_part1 + part2 (Vector)
│   ├── Q = x × W_q   (matmul, Cube)   [1,32,64]×[64,64]→[1,32,64]
│   ├── K = x × W_k   (matmul, Cube)
│   ├── V = x × W_v   (matmul, Cube)
│   ├── Q reshape [1,32,64]→[1,32,4,16] → transpose [1,4,32,16]
│   ├── K reshape + transpose 同上
│   ├── V reshape + transpose 同上
│   ├── K^T = K.transpose(-2,-1)  (transpose_2d)  [1,4,16,32]
│   ├── scores = Q × K^T  (matmul, Cube)  [1,4,32,32]
│   ├── scores = scores × 0.25  (mul_scalar, Vector)  (1/sqrt(16))
│   ├── attn = softmax(scores)  → softmax_part1 + part2 (Vector)
│   ├── context = attn × V  (matmul, Cube)  [1,4,32,16]
│   ├── context transpose [1,32,4,16] → reshape [1,32,64]
│   ├── output = context × W_o  (matmul, Cube)
│   ├── output = output + input  (add, Vector, residual)
│   ├── LayerNorm → layernorm_part1 + part2
│   ├── ffn1 = x × W_ff1  (matmul, Cube)  [1,32,64]×[64,256]→[1,32,256]
│   ├── ffn1 = gelu(ffn1)  (Vector)
│   ├── ffn2 = ffn1 × W_ff2  (matmul, Cube)  [1,32,256]×[256,64]→[1,32,64]
│   └── output = ffn2 + residual  (add, Vector)
│
├── Layer 2: 结构同Layer 1，不同权重
│
└── Output [1, 32, 64]
```

### 10.2 数据规模

| 参数 | 值 |
|------|-----|
| batch_size | 1 |
| seq_len | 32 |
| hidden_size | 64 |
| num_heads | 4 (head_dim=16, 对齐cube_size) |
| ffn_intermediate | 256 |
| 每层权重量 | ~100KB |
| 总tensor量 | ~500KB |

### 10.3 每层算子统计

| 算子 | 数量 | 计算单元 |
|------|------|----------|
| npu_layernorm_part1 | 2 | Vector |
| npu_layernorm_part2 | 2 | Vector |
| npu_matmul | 8 | Cube |
| npu_reshape | 8 | Scalar |
| npu_transpose | 4 | Vector |
| npu_transpose_2d | 1 | Vector |
| npu_mul_scalar | 1 | Vector |
| npu_softmax_part1 | 1 | Vector |
| npu_softmax_part2 | 1 | Vector |
| npu_gelu | 1 | Vector |
| npu_add | 2 | Vector |
| **每层** | **31** | |
| **2层** | **62** | |

---

## 11. 多Agent并行开发方案

### 11.1 Agent分配

| Agent | 负责模块 | 耗时 | 前置 |
|-------|---------|------|------|
| Agent 0 | common | 30min | 无 |
| Agent 1 | graph_capture + op_mapping + op_decomposition | 1.5h | common |
| Agent 2 | op_absorption + format_annotator + validator | 1.5h | common |
| Agent 3 | memory_planner + scheduler | 1.5h | common |
| Agent 4 | codegen | 2h | common |
| Agent 0 | integration（串联+端到端测试） | 1h | 全部模块 |

### 11.2 时间线

```
0:00 ─── Agent 0: 开发common ───────────────────── 0:30
         │
         ├─ 发布common给所有Agent ─┐
         │                         │
0:30 ─── Agent 1: capture+map+dec ─┼─ Agent 2: absorb+fmt+valid ─┼─ Agent 3: mem+sched ─┼─ Agent 4: codegen ─── 2:30
         │                         │                              │                      │
         └─────────────────────────┴──────────────────────────────┴──────────────────────┘
                                                                                          │
2:30 ─── Agent 0: integration 集成测试 ──────────────────────────────────────────────────── 3:30

3:30 ─── 缓冲：修bug、补测试 ───────────────────────────────────────────────────────────── 4:00

关键路径：0:30(common) + 2:00(最长并行) + 1:00(集成) + 0:30(buffer) = 4:00
```

### 11.3 Agent间契约

每个Agent拿到的信息：

1. common模块的完整代码（graph_ir.py / logger.py / config_loader.py / errors.py）
2. 自己负责的模块的README.md（本文档第7节）
3. Graph IR的JSON格式定义（本文档第5.2节）
4. 自己模块文件夹中config/目录的配置文件内容

每个Agent交付的产物：

1. 模块代码文件（如op_mapping.py）
2. 模块UT文件（如test_op_mapping.py）
3. 局部demo文件（demo_input_graph.json + run_demo.py + expected_output.json）
4. UT运行结果截图/日志（证明UT全部通过）

### 11.4 集成验收标准

Agent 0在集成阶段检查：

| 检查项 | 方法 |
|--------|------|
| 所有模块UT通过 | `cd torch2c && pytest --tb=short` |
| 端到端管线跑通 | `python integration/demo/run_full_demo.py` 无报错 |
| 生成的C代码语法正确 | `gcc -fsyntax-only -include npu_mock.h output/src/model_graph.c` |
| C侧UT通过 | `cd output && cmake . && make && ctest` |
| 算子数量正确 | 检查model_graph.c中算子调用数 = 62 |
| 日志完整 | 每个Pass有入口/出口INFO日志 |

---

## 12. 精度验证标准

| 指标 | 阈值 |
|------|------|
| 最大绝对误差 | < 1e-3（FP16） |
| 余弦相似度 | > 0.999 |
| 不匹配元素比例 | < 0.1% |

---

## 13. 日志规格汇总

### Python侧

```
格式：[2024-01-15 10:30:01] [INFO] [op_mapping] Pass完成，映射了15个算子
```

| Pass | 统计内容 |
|------|---------|
| graph_capture | 节点数、tensor数、权重tensor数 |
| op_mapping | 已映射数、未映射数 |
| op_decomposition | 裂解数、新增节点数、新增中间tensor数 |
| op_absorption | 吸收数、消除的中间tensor数 |
| format_annotator | 标注的tensor数 |
| validator | 通过/失败（列出未支持算子） |
| memory_planner | HBM总用量、峰值、复用节省量、L1最大用量 |
| scheduler | 依赖关系数、可并行对数 |
| codegen | 文件数、代码行数、算子调用数 |

### C侧

```c
#ifdef NPU_DEBUG_DUMP
  [NPU] === Op 0: npu_layernorm_part1 (vector) ===
  [NPU]   DMA load: hbm[0x0000] → l1[0x0000], 4096 bytes
  [NPU]   Execute: npu_layernorm_part1(l1+0, l1+4096, l1+4224, ...)
  [NPU]   DMA store: l1[0x1080] → hbm[0x2000], 4096 bytes
#endif
```

---

## 14. 配置文件填写检查清单

| 序号 | 文件 | 做什么 | 来源 | 耗时 |
|------|------|--------|------|------|
| 1 | direct_mappings.yaml | ATen→NPU 1对1映射 | C头文件 + torch.export | 1h |
| 2 | decompositions.yaml | 裂解规则 | C接口文档 | 30min |
| 3 | c_api_signatures.yaml | 函数签名 | C头文件 | 1-2h |
| 4 | type_format_config.yaml | 枚举+算子format | C头文件enum | 1h |
| 5 | hardware_config.yaml | 存储参数 | 芯片规格书 | 15min |
| 6 | absorptions.yaml | 吸收规则 | C接口文档 | 30min |
| 7 | model_config.yaml | 模型参数 | 模型定义 | 10min |
| 8 | codegen_config.yaml | 输出选项 | 用默认值 | 5min |

**总计约半天。** 1-4为核心配置。

---

## 15. 交付验收标准

### Python编译器

| 检查项 | 标准 |
|--------|------|
| 8个配置文件正确加载 | 无报错 |
| 每个模块UT独立通过 | pytest单模块全绿 |
| 全量UT通过 | pytest全绿 |
| 端到端demo通过 | 生成完整C工程 |
| 日志完整 | 每个Pass有入口/出口统计 |
| 生成的C代码语法正确 | gcc -fsyntax-only通过 |

### C工程

| 检查项 | 标准 |
|--------|------|
| CMake构建通过 | mock模式编译无错误 |
| C侧UT全通过 | ctest全通过 |
| 算子调用数正确 | model_graph.c中62个算子块 |
| 注释和日志完整 | 每个算子块有来源注释 |

### 真实环境ST

| 检查项 | 标准 |
|--------|------|
| 链接真实SDK编译通过 | 无错误 |
| 执行不崩溃 | model_run()正常返回 |
| 精度通过 | abs<1e-3, cos>0.999 |

---

## 16. 补充决策记录

以下为需求评审中确认的补充决策，各模块开发时必须遵循。

### 16.1 环境约束

| 项目 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.4+（Python 3.10兼容的最新稳定版） |
| 包管理 | pyproject.toml + pip install -e . |

### 16.2 图捕获（graph_capture）

- **禁止torch.export自动分解**：传入自定义decomposition table，保留 `aten.layer_norm`、`aten._softmax` 等高级算子不被拆解为原子op
- **ATen算子命名**：config中使用全称（如 `aten.mm.default`），需先跑一次torch.export确认实际算子名
- **多输出算子**：全部保留。如 `aten.layer_norm` 返回 `(output, mean, rstd)` 三个tensor，全部创建对应Tensor对象。无消费者的输出tensor由memory_planner回收
- **Scalar值**：作为常量tensor处理，`is_weight=True`，`shape=[1]`。如attention中的 `scores * 0.25`，0.25创建为一个scalar tensor
- **Mask输入**：Demo模型加入attention mask，作为模型外部输入，`is_model_input=True`，`shape=[1, 1, 32, 32]`

### 16.3 格式与精度（format_annotator）

- **tensor.format语义**：表示该tensor在HBM中的存储格式。DMA load时按消费者需求转换到L1
- **DMA随路转换**：同一tensor被不同format需求的算子消费时，不插入显式format_convert节点，由DMA搬运时自动完成格式转换
- **format_annotation扩展**：输入format/dtype、计算dtype、输出format/dtype 均可不同

```python
node.format_annotation = {
    "inputs":  [{"format": "nz", "dtype": "fp16"}],   # 每个输入可有独立的format和dtype
    "outputs": [{"format": "nd", "dtype": "fp32"}],    # 输出format/dtype可与输入不同
    "compute_dtype": "fp32",                            # 计算精度，可与输入输出均不同
    "supports_format_convert": True,
    "supports_dtype_cast": True
}
```

### 16.4 算子接口更新

- **npu_transpose 4D接口**：增加ndim和dims（shape）参数，支持高维tensor转置

```yaml
npu_transpose:
  params:
    - { name: "input",  type: "addr",  source: "tensor.input_0.l1_offset" }
    - { name: "out",    type: "addr",  source: "tensor.output_0.l1_offset" }
    - { name: "ndim",   type: "int",   source: "tensor.input_0.ndim" }
    - { name: "dims",   type: "int_array", source: "tensor.input_0.shape" }
    - { name: "dim0",   type: "int",   source: "param.dim0" }
    - { name: "dim1",   type: "int",   source: "param.dim1" }
    - { name: "dtype",  type: "enum",  source: "tensor.input_0.dtype" }
```

- **Reshape**：正常DMA搬运，与其他算子一样走完整的load/store流程，不做零拷贝优化

### 16.5 裂解规则

- **中间tensor shape**：与源算子输入同shape。这是NPU C接口设计决定的（固定成组裂解），不是数学意义上的中间量

### 16.6 配置管理

- **integration/config/ 与模块局部config/ 完全相同**，是同一份配置的副本。集成时统一从integration/config/加载
- **Demo数据准备**：各Agent先按README规格自行构造demo JSON，后续用graph_capture的真实输出修正

### 16.7 Absorption

- Demo模型加入attention mask后，`add(scores, mask) → softmax_part1` 的pattern可被匹配，absorption模块能端到端验证