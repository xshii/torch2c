# torch2c 架构优化 TODO

> 基于 Clean Code / Code Complete / GoF 设计模式 / Python Cookbook 分析。
> 目标：降低后续 AI agent 开发新 pass 的认知负担和出错概率。

---

## Sprint 1 — 零破坏性，立竿见影

### T1. Descriptor 替代 `node.params` 字符串查找
**问题**：40+ 处 `node.params["_roofline"]` / `node.params.get("_fusion_group")` 散布在 8 个模块中，typo 不报错，无 IDE 补全。

**方案**：在 `Node` 上用 Python Descriptor Protocol 提供 typed access，底层仍存 `params` dict，序列化兼容。

```python
class _PassSlot:
    def __init__(self, key, default=None):
        self.key = key; self.default = default
    def __get__(self, obj, _=None):
        if obj is None: return self
        return obj.params.get(self.key, self.default)
    def __set__(self, obj, value):
        obj.params[self.key] = value

class Node:
    roofline = _PassSlot("_roofline")
    tile_config = _PassSlot("_tile_config")
    fusion_group = _PassSlot("_fusion_group")
    fusion_role = _PassSlot("_fusion_role")
    mha_analysis = _PassSlot("_mha_analysis")
    weight_slices = _PassSlot("_weight_slices")
    tile_info = _PassSlot("_tile_info")
    npu_hint = _PassSlot("_npu")
    compute_dtype_val = _PassSlot("compute_dtype", default="fp16")
```

**收益**：旧代码 `node.params["_roofline"]` 仍然工作；新代码 `node.roofline` 有补全、typo 报 AttributeError。渐进采用。

**文件**：`torch2c/common/graph_ir.py`
**工作量**：~30 行

---

### T2. Enum 替代 magic string
**问题**：`compute_unit`/`format`/`storage`/`dtype`/`fusion_role` 全用裸字符串，`"cube"` 可能写成 `"Cube"` 或 `"CUBE"`。

**方案**：`str + Enum` 双继承，序列化兼容（json.dumps 直接输出 `"cube"`）。

```python
class ComputeUnit(str, Enum):
    CUBE = "cube"
    VECTOR = "vector"
    IDMA = "idma"
    DMA = "dma"

class TensorFormat(str, Enum):
    ND = "nd"
    NZ = "nz"
    ZZ = "zz"
    NN = "nn"

class Storage(str, Enum):
    HBM = "hbm"
    LOCAL = "local"
    PIPE = "pipe"
```

**收益**：`ComputeUnit.CUEB` → 立即 `AttributeError`；IDE 列出所有合法值；`match` 语句穷举检查。

**文件**：`torch2c/common/graph_ir.py`（定义）+ 各 pass 渐进替换
**工作量**：~20 行定义，迁移渐进

---

### T3. GraphBuilder 测试工具
**问题**：10+ 个测试文件各自 50+ 行手动创建 Graph/Node/Tensor，需手动维护 producer/consumer 一致性。

**方案**：Fluent Builder API。

```python
class GraphBuilder:
    def weight(self, shape, name=None, dtype="fp16") -> str: ...
    def input(self, shape, name=None, dtype="fp16") -> str: ...
    def op(self, npu_op, inputs, output_shape, compute_unit="cube") -> str: ...
    def build(self) -> Graph: ...

# 3 行代替 50 行
g = (GraphBuilder()
    .weight([64, 64], "W")
    .input([1, 32, 64], "X")
    .op("cube_matmul", ["X", "W"], [1, 32, 64])
    .build())
```

**收益**：新 pass 测试代码量降 60%，producer/consumer 自动维护不会出错。

**文件**：`torch2c/common/test_utils.py`（新建）
**工作量**：~60 行

---

## Sprint 2 — Graph 能力升级

### T4. Graph 改写原子方法
**问题**：4 个 pass 各自手动实现相同的图改写模式（替换节点输入 + 更新 consumer_node_ids + splice execution_order），15 行接线代码重复 4 次且容易出错。

**方案**：在 `Graph` 上增加原子化 API。

```python
class Graph:
    def replace_node(self, old_id: str, new_nodes: list[Node],
                     new_tensors: list[Tensor] | None = None) -> None:
        """删旧节点，插入新节点序列到原位置，更新所有 tensor 引用。"""

    def insert_before(self, target_id: str, new_node: Node,
                      new_tensor: Tensor | None = None) -> None:
        """在 target 前插入节点，自动接线。"""

    def rewire_input(self, node_id: str, port: int, new_tid: str) -> None:
        """替换节点第 port 个输入为 new_tid，自动更新 consumer 列表。"""

    def splice_execution_order(self, old_id: str, new_ids: list[str]) -> None:
        """execution_order 中用 new_ids 替换 old_id（O(n) 封装）。"""
```

**收益**：新 pass 用 `graph.insert_before()` 一行代替 15 行手动接线。

**受影响 pass**：op_decomposition (`_rewire_graph`)、op_absorption (`_absorb_one`)、reformat_inserter (`run`)、mha_merge (`_apply_split`)
**文件**：`torch2c/common/graph_ir.py`
**工作量**：~80 行

---

### T5. Graph 查询 API
**问题**：`_find_single_consumer()` 在 op_absorption 里是私有函数，但 mha_merge 和 fusion_planner 也需要相同逻辑。`intermediates` 过滤在 fusion_planner 和 storage_assigner 中重复。

**方案**：

```python
class Graph:
    def single_consumer(self, tensor_id: str) -> Node | None:
        """tensor 的唯一消费者，多消费者返回 None。"""

    def intermediates(self) -> Iterator[Tensor]:
        """非 weight、非 model_input/output、有 producer 的中间 tensor。"""

    def nodes_by_unit(self, unit: str) -> Iterator[Node]:
        """按 compute_unit 过滤。"""

    def consumers_of(self, node_id: str) -> list[Node]:
        """节点的直接下游节点。"""

    def producer_of(self, tensor_id: str) -> Node | None:
        """tensor 的生产者节点。"""
```

**文件**：`torch2c/common/graph_ir.py`
**工作量**：~40 行

---

### T6. FormatAnnotation 结构化
**问题**：`node.format_annotation` 是 `dict | None`，内部结构 `{"inputs": [{"format":..., "dtype":...}]}` 在 6 个模块中手动构建和解析。

**方案**：

```python
@dataclass(frozen=True)
class FormatSpec:
    format: str
    dtype: str

@dataclass
class FormatAnnotation:
    inputs: list[FormatSpec]
    outputs: list[FormatSpec]

    @classmethod
    def uniform(cls, n_in, n_out, fmt="nd", dtype="fp16") -> FormatAnnotation: ...
```

**文件**：`torch2c/common/graph_ir.py`（定义）+ format_annotator、reformat_inserter 等渐进迁移
**工作量**：~30 行定义，迁移渐进

---

## Sprint 3 — Pass 架构升级

### T7. Analysis Pass vs Transform Pass 分离
**问题**：所有 pass 共用 `run(graph, config) -> Graph`，但 roofline/global_tiler/fusion_planner 是只读分析，op_decomposition/absorption/reformat 是结构变换。无法缓存分析结果、无法声明依赖、无法自动校验顺序。

**方案**：LLVM PassManager 模式。

```python
class AnalysisPass(Protocol[T]):
    def analyze(self, graph: Graph, config: ...) -> T: ...

class TransformPass(Protocol):
    requires: list[type]           # 依赖的 analysis 类型
    invalidates: list[type]        # 会使哪些 analysis 失效
    def run(self, graph: Graph, config: ...) -> Graph: ...
```

Pipeline 自动缓存 analysis 结果，transform 声明 `invalidates` 触发重算。

**受影响模块**：pipeline.py、所有 pass
**工作量**：高（需迁移所有 pass 接口）

---

### T8. Pass 自注册（`__init_subclass__`）
**问题**：新增 pass 需要手动编辑 pipeline.py 的 3 个 list + pass_config.py 的 Enum。

**方案**：

```python
class CompilerPass:
    _registry: ClassVar[dict] = {}
    def __init_subclass__(cls, *, phase, number, toggle=None, **kw):
        super().__init_subclass__(**kw)
        cls._registry[cls.__name__] = cls
        cls._phase = phase
        cls._number = number
        cls._toggle = toggle
```

Pass 文件 import 即注册。

**文件**：`torch2c/common/pass_base.py`（新建）+ pipeline.py 重构
**工作量**：中

---

### T9. Config Schema（per-pass dataclass）
**问题**：每个 pass 的 config 结构仅在 docstring 或 `.get()` 调用中隐含，pipeline.py `_load_configs()` 返回 14 key dict 无文档。

**方案**：per-pass config dataclass + 自动从 YAML 水合。

```python
@dataclass
class MhaMergeConfig:
    prefer_merged_threshold: float = 0.9
    max_batch_for_split: int = 1
    last_dim_align: int = 16
    l1_size_bytes: int = 16 * 1024 * 1024
```

**文件**：各 pass 模块 + pipeline.py config 加载
**工作量**：中

---

## Sprint 4 — 质量与安全

### T10. Graph 事务（Context Manager）
**问题**：pass 中途异常导致 Graph 半修改状态，无法回滚。

**方案**：

```python
@contextmanager
def graph_transaction(graph: Graph):
    snapshot = graph.to_dict()
    try:
        yield graph
    except Exception:
        restored = Graph.from_dict(snapshot)
        graph.nodes, graph.tensors = restored.nodes, restored.tensors
        graph.execution_order = restored.execution_order
        raise
```

**文件**：`torch2c/common/graph_ir.py` + pipeline.py
**工作量**：~20 行

---

### T11. Pipeline.py 拆分
**问题**：725 行、6 个职责（pass 声明 + 拓扑导出 + config 加载 + 执行循环 + debug hooks + 编译入口）。

**方案**：

```
integration/pipeline/
    __init__.py          # compile(), compile_graph_only(), inspect()
    pass_registry.py     # _PassDesc, 3 pass lists, get_pass_topology()
    pass_runner.py       # _run_pass_list, _run_post_validation
    config_resolver.py   # _load_configs, _resolve_compile_configs
    pass_descriptions.py # _PASS_DESC dict (300 行纯描述文本)
```

**工作量**：低（纯拆分，无逻辑变更）

---

### T12. `validate_stage()` 接入 Pipeline
**问题**：`Graph.STAGE_CONTRACTS` 只覆盖 5 个 pass，且 `validate_stage()` 在 pipeline 中从未被调用。

**方案**：补齐所有 pass 的契约 + 在 `_run_pass_list` 中 pass 完成后自动调用。

**文件**：`torch2c/common/graph_ir.py`（补契约）+ pipeline.py（调用）
**工作量**：低

---

### T13. 代码小修
| 项 | 文件 | 说明 |
|----|------|------|
| `run()` 放文件顶部 | op_absorption.py | 入口函数在 L171，读者需跳过 helper |
| import 移到文件顶部 | op_absorption.py:94 | `from constants import` 在函数间 |
| `_PASS_DESC` 300 行文本抽离 | pipeline.py:185-292 | 纯描述文本与逻辑混杂 |
| graph_capture/codegen 纳入 pass list | pipeline.py | 目前硬编码在 compile() 中 |

---

## 实施优先级总览

| Sprint | 内容 | 向后兼容 | AI 降本 |
|--------|------|---------|---------|
| **1** | T1 Descriptor + T2 Enum + T3 GraphBuilder | 完全兼容 | ★★★★★ |
| **2** | T4 Graph 改写 + T5 查询 API + T6 FormatAnnotation | 完全兼容 | ★★★★ |
| **3** | T7 Analysis/Transform + T8 自注册 + T9 Config Schema | 需迁移 | ★★★★ |
| **4** | T10 事务 + T11 拆分 + T12 契约 + T13 小修 | 完全兼容 | ★★ |
