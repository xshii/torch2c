# 规格驱动测试 / 用例驱动测试

写测试不是"代码写完了补几个"，而是"先定义系统该怎么表现，再验证它做到了"。
本 skill 提供从规格/用例出发设计测试的完整方法论。

## 核心理念

```
规格 (Spec)  →  可测试属性  →  测试用例  →  实现
     ↑                                        │
     └────────── 失败时回到规格确认 ───────────┘
```

**不要**从实现出发写测试（"这个函数我调了一下，assert 不报错"）。
**要**从规格出发写测试（"根据硬件手册，NZ 格式的 dim[-2] 必须是 c0 的倍数"）。

## 方法一：规格驱动（Spec-Driven）

### 步骤

1. **找到规格源**：`docs/ordr.md`（需求）、`CLAUDE.md`（架构约束）、`hardware_config.yaml`（硬件参数）
2. **提取可测试属性**：每条规格转化为一个 `assert`
3. **分类**：正向（满足规格）、反向（违反规格应报错）、边界（刚好在阈值上）
4. **命名**：`test_{属性}_{场景}`

### 属性提取模板

从一条规格出发，问自己 5 个问题：

| 问题 | 产出的测试 |
|------|------------|
| 正常情况下结果是什么？ | `test_{property}_normal` |
| 输入非法时应该怎样？ | `test_{property}_rejects_invalid` |
| 边界值是什么？ | `test_{property}_boundary` |
| 空/零/极端情况？ | `test_{property}_edge_case` |
| 和其他规格有交叉吗？ | `test_{property}_with_{other}` |

### 示例：从 block_pad 规格提取测试

**规格**：_"ND 格式 fp16 的 dim[-1] 对齐到 16，dim[-2] 不对齐"_

```python
class TestBlockPadNDSpec:
    """规格：ND fp16 → dim[-1] 对齐 16, dim[-2] 不变。"""

    def test_normal_padding(self):
        """dim[-1]=17 应该 pad 到 32。"""
        g = _make_graph(("t1", [1, 3, 17], "fp16", "nd"))
        run(g, CONFIG)
        assert g.tensors["t1"].shape == [1, 3, 32]

    def test_dim_neg2_unchanged(self):
        """ND 格式 dim[-2] 不应被修改。"""
        g = _make_graph(("t1", [1, 5, 16], "fp16", "nd"))
        run(g, CONFIG)
        assert g.tensors["t1"].shape[-2] == 5  # 不变

    def test_already_aligned_no_change(self):
        """已对齐的 shape 不应改变，original_shape 为 None。"""
        g = _make_graph(("t1", [1, 16, 16], "fp16", "nd"))
        run(g, CONFIG)
        assert g.tensors["t1"].original_shape is None

    def test_boundary_exactly_16(self):
        """dim[-1]=16 刚好对齐，不应改变。"""
        g = _make_graph(("t1", [1, 3, 16], "fp16", "nd"))
        run(g, CONFIG)
        assert g.tensors["t1"].shape[-1] == 16

    def test_rejects_zero_dim(self):
        """scalar tensor (ndim=0) 应被跳过。"""
        g = _make_graph(("t1", [], "fp16", "nd"))
        run(g, CONFIG)
        assert g.tensors["t1"].shape == []
```

**规格**：_"NZ int8 的 dim[-2] 对齐到 32（c0=32），dim[-1] 对齐到 16"_

```python
class TestBlockPadNZInt8Spec:
    """规格：NZ int8 → dim[-2] 对齐 32, dim[-1] 对齐 16。"""

    def test_asymmetric_alignment(self):
        """两个维度的对齐值不同。"""
        g = _make_graph(("t1", [1, 5, 17], "int8", "nz"))
        run(g, CONFIG)
        assert g.tensors["t1"].shape[-2] == 32  # 5 → 32
        assert g.tensors["t1"].shape[-1] == 32  # 17 → 32
```

### 常见规格源 → 可测试属性

| 规格源 | 属性示例 |
|--------|----------|
| `format_capabilities` | Cube src1 只接受 NZ → 非 NZ 输入应触发 reformat |
| `block_pad.alignment` | 每个 format×dtype 组合的对齐值 |
| `memory.l1.total_size_bytes` | 超出 L1 时应 fallback 到 spill/tiled 策略 |
| `decompositions.yaml` | layernorm 裂解后应产生 part1 + part2 两个节点 |
| `direct_mappings.yaml` | 每个 ATen op 映射后 npu_op 正确 |
| pass 后置条件 | validator 后所有 node.is_mapped == True |

## 方法二：用例驱动（Scenario-Driven）

### 步骤

1. **定义场景**：一个具体的用户故事（模型 + 配置 + 预期行为）
2. **划分层次**：单元 → 集成 → 端到端
3. **设计场景矩阵**：不同模型 × 不同硬件配置 × 不同 pass 组合

### 场景矩阵模板

```
场景 = 模型复杂度 × 内存压力 × 精度模式
```

| 编号 | 模型 | L1 | 精度 | 预期策略 | 验证重点 |
|------|------|----|------|----------|----------|
| ST1 | Linear (AX+B) | 充裕 | fp16 | bulk | 最简编译链 |
| ST2 | Linear (AX+B) | 充裕 | fp32 compute | bulk | 混合精度 |
| ST3 | Linear (AX+B) | 紧张 | fp16 | tiled | tiling 正确性 |
| ST4 | 2-layer MLP | 充裕 | fp16 | bulk | 多节点复用 |
| ST5 | 2-layer MLP | 中等 | fp16 | perop | L1 liveness 复用 |
| ST6 | MHA (4 头) | 紧张 | fp16 | tiled | 注意力 tiling |
| ST7 | MHA | 紧张 | fp16 | tiled+double buffer | DMA 隐藏 |
| ST8 | MHA (多 batch) | 紧张 | fp16 | tiled | batch>1 tiling |
| ST9 | Full MHA+FFN | 充裕 | fp16 | perop | 完整注意力链 |
| ST10 | - | - | - | - | 策略对比分析 |

### 场景测试模板

```python
class TestST_XXX:
    """场景描述：模型 + 硬件配置 + 预期行为。"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    @pytest.fixture
    def config_dir(self):
        """创建自定义 L1 大小的配置目录。"""
        return _make_config_dir(l1_size=TODO_L1_SIZE)

    def test_compiles_successfully(self, output_dir, config_dir):
        """编译链正常完成，无异常。"""
        torch.manual_seed(42)
        model = TODO_Model()
        model.eval()
        compile(model=model, dummy_input=torch.randn(TODO_SHAPE),
                config_dir=config_dir, output_dir=output_dir)

    def test_uses_expected_strategy(self, output_dir, config_dir):
        """验证使用了预期的内存策略。"""
        torch.manual_seed(42)
        model = TODO_Model()
        model.eval()
        graph = compile_graph_only(model=model, dummy_input=torch.randn(TODO_SHAPE),
                                   config_dir=config_dir)
        # 检查策略
        strategy = graph.configs.get("_memory_strategy", "")
        assert strategy == "TODO_EXPECTED_STRATEGY"

    def test_golden_comparison(self, output_dir, config_dir):
        """C golden 比对通过。"""
        torch.manual_seed(42)
        result = _compile_and_validate(
            TODO_Model().eval(), torch.randn(TODO_SHAPE),
            output_dir, config_dir, atol=TODO_ATOL, cosine_tol=TODO_COSINE)
        assert result["passed"], f"Golden FAIL: {result['stdout']}"
```

## 方法三：不变量测试（Invariant-Based）

检查系统在任何输入下都必须满足的属性。

### 编译器不变量清单

```python
def assert_graph_invariants(graph: Graph) -> None:
    """编译器全局不变量 — 任何 pass 后都必须满足。"""

    # 1. 每个 tensor 的 producer 和 consumer 引用有效
    for t in graph.tensors.values():
        if t.producer_node_id:
            assert t.producer_node_id in graph.nodes, \
                f"tensor {t.id} 的 producer {t.producer_node_id} 不存在"
        for cid in t.consumer_node_ids:
            assert cid in graph.nodes, \
                f"tensor {t.id} 的 consumer {cid} 不存在"

    # 2. execution_order 中的节点都存在
    for nid in graph.execution_order:
        assert nid in graph.nodes, \
            f"execution_order 包含不存在的节点 {nid}"

    # 3. 每个节点的输入输出 tensor 都存在
    for node in graph.nodes.values():
        for tid in node.inputs + node.outputs:
            assert tid in graph.tensors, \
                f"节点 {node.id} 引用了不存在的 tensor {tid}"

    # 4. 无孤立 tensor（至少有 producer 或 consumer）
    for t in graph.tensors.values():
        has_ref = (t.producer_node_id is not None
                   or len(t.consumer_node_ids) > 0
                   or t.is_model_input or t.is_weight)
        assert has_ref, f"tensor {t.id} 是孤立的"
```

### Pass 级不变量

```python
# op_mapping 后
def assert_after_mapping(graph):
    for node in graph.nodes.values():
        if node.op_type.startswith("aten."):
            assert node.npu_op is not None, \
                f"mapping 后 {node.id} 仍无 npu_op"

# op_decomposition 后
def assert_after_decomposition(graph):
    for node in graph.nodes.values():
        assert node.is_mapped, \
            f"decomposition 后 {node.id} 仍未 is_mapped"

# format_annotator 后
def assert_after_format(graph):
    for node in graph.nodes.values():
        if node.is_mapped and node.compute_unit != "dma":
            assert node.format_annotation is not None, \
                f"format 后 {node.id} 仍无 format_annotation"

# block_pad 后
def assert_after_block_pad(graph, config):
    from torch2c.optpass.c_block_pad.block_pad import parse_alignment_table, get_align_rule
    table, fallback = parse_alignment_table(config)
    for t in graph.tensors.values():
        if len(t.shape) < 2:
            continue
        rule = get_align_rule(table, fallback, t.format, t.dtype)
        if rule.dim_neg1 > 1:
            assert t.shape[-1] % rule.dim_neg1 == 0, \
                f"tensor {t.id} dim[-1]={t.shape[-1]} 未对齐到 {rule.dim_neg1}"
        if rule.dim_neg2 > 1:
            assert t.shape[-2] % rule.dim_neg2 == 0, \
                f"tensor {t.id} dim[-2]={t.shape[-2]} 未对齐到 {rule.dim_neg2}"

# memory_planner 后
def assert_after_memory_plan(graph):
    for t in graph.tensors.values():
        if t.storage == "hbm":
            assert t.hbm_offset is not None, \
                f"memory_plan 后 tensor {t.id} 缺 hbm_offset"
            assert t.hbm_size is not None and t.hbm_size > 0, \
                f"memory_plan 后 tensor {t.id} 缺 hbm_size"
```

## 方法四：决策表测试（Decision Table）

用表格穷举输入组合和预期输出，适合配置驱动的逻辑。

### 示例：format_annotator 决策表

```python
# 规格：compute_unit × op_type → 输入 format × 输出 format
ANNOTATOR_DECISIONS = [
    # (compute_unit, npu_op,            input_fmt, output_fmt)
    ("cube",   "cube_matmul",           ["nd", "nz"], ["nd"]),
    ("cube",   "cube_matmul_bias",      ["nd", "nz", "nd"], ["nd"]),
    ("vector", "vector_add",            ["nd", "nd"], ["nd"]),
    ("vector", "vector_gelu",           ["nd"],       ["nd"]),
    ("vector", "vector_softmax",        ["nd"],       ["nd"]),
    ("idma",   "idma_reshape",          ["nd"],       ["nd"]),
    ("idma",   "idma_transpose",        ["nd"],       ["nd"]),
]

@pytest.mark.parametrize("unit,op,in_fmts,out_fmts", ANNOTATOR_DECISIONS,
                         ids=[d[1] for d in ANNOTATOR_DECISIONS])
def test_format_annotation_decisions(unit, op, in_fmts, out_fmts):
    """format_annotator 应按 format_capabilities 为每种算子设置正确 format。"""
    graph = _build_graph_for_op(unit, op, len(in_fmts), len(out_fmts))
    run(graph, config)
    node = list(graph.nodes.values())[0]
    ann = node.format_annotation
    assert [a["format"] for a in ann["inputs"]] == in_fmts
    assert [a["format"] for a in ann["outputs"]] == out_fmts
```

### 示例：get_dim_align 决策表

```python
ALIGN_DECISIONS = [
    # (format, dtype, expected_dim_neg2, expected_dim_neg1)
    ("nd",  "fp16", 1,  16),
    ("nd",  "int8", 1,  32),
    ("nz",  "fp16", 16, 16),
    ("nz",  "int8", 32, 16),
    ("zz",  "fp16", 16, 16),
    ("zz",  "int8", 16, 32),
    ("nn",  "fp16", 16, 16),
    ("nn",  "int8", 32, 16),
]

@pytest.mark.parametrize("fmt,dtype,exp_neg2,exp_neg1", ALIGN_DECISIONS,
                         ids=[f"{d[0]}_{d[1]}" for d in ALIGN_DECISIONS])
def test_alignment_table(fmt, dtype, exp_neg2, exp_neg1):
    """get_dim_align 应返回 format×dtype 对应的对齐值。"""
    neg2, neg1 = get_dim_align(fmt, dtype)
    assert neg2 == exp_neg2
    assert neg1 == exp_neg1
```

## 方法五：回归测试（Regression）

每个 bug fix 必须附带复现测试。

```python
class TestRegression:
    def test_fix_embedding_addr_to_ptr(self):
        """回归：npu_idma_embedding.c 曾用 .addr 导致编译失败。
        修复：commit ebe0b71，改为 .ptr。
        """
        # 编译包含 embedding 的模型不应报错
        model = EmbeddingModel()
        compile(model, dummy_input, ...)

    def test_fix_transpose_missing_compute_dtype(self):
        """回归：idma_transpose 签名缺 compute_dtype 参数。
        修复：commit ebe0b71，c_api_signatures.yaml 补充参数。
        """
        # codegen 生成的 transpose 调用应有 8 个参数
        ...
```

## 测试文件组织

```
torch2c/{module}/tests/
├── test_{module}.py           # 单元测试（函数级）
├── test_{module}_spec.py      # 规格驱动测试（从 spec 提取）
└── test_{module}_regression.py # 回归测试（从 bug 提取）

torch2c/integration/tests/
├── demo_st/
│   └── test_st_scenarios.py   # 端到端场景测试（ST1-ST10）
└── test_pipeline.py           # 管线集成测试
```

## 用例设计检查清单

每次写测试前过一遍：

- [ ] 从哪条规格/需求出发的？（写在 docstring 里）
- [ ] 覆盖了正向（happy path）吗？
- [ ] 覆盖了反向（error path）吗？
- [ ] 覆盖了边界值吗？
- [ ] 是否参数化了？（多个输入用 `@pytest.mark.parametrize`）
- [ ] 幂等性测试了吗？（pass 跑两次结果相同）
- [ ] 断言够具体吗？（不是 `assert result`，而是 `assert result == expected`）
- [ ] 随机种子固定了吗？（`torch.manual_seed(42)`）
- [ ] 测试名说明了测什么？（`test_nd_fp16_dim_neg2_unchanged`，不是 `test_1`）
