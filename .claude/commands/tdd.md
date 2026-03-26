# TDD 工作流

本项目采用测试驱动开发。任何功能改动都遵循 Red → Green → Refactor 循环。

## 核心原则

1. **先写测试，再写实现** — 测试定义了"完成"的标准
2. **最小实现** — 只写让测试通过的最少代码
3. **每个 commit 都是绿色** — 不提交 broken 的代码
4. **测试覆盖变更** — 每个 bug fix 附带复现测试

## 操作流程

### Step 1: Red — 写失败的测试

```python
# torch2c/{module}/tests/test_{feature}.py

def test_new_feature():
    """描述期望行为。"""
    g = _make_graph(...)  # 构建最小测试图
    result = run(g, config)
    assert result == expected  # 明确的断言
```

运行确认测试失败：

```bash
.venv/bin/pytest torch2c/{module}/tests/test_{feature}.py -v
# 预期：FAILED
```

### Step 2: Green — 最小实现

只写让测试通过的代码，不多不少：

```bash
.venv/bin/pytest torch2c/{module}/tests/test_{feature}.py -v
# 预期：PASSED
```

### Step 3: Refactor — 清理代码

在测试保护下重构：

```bash
# 确认没破坏任何东西
.venv/bin/pytest --tb=short -q
```

### Step 4: 全量回归

```bash
.venv/bin/pytest --tb=short -q
# 预期：448 passed（或更多）
```

## 测试模式模板

### 模式 A: Pass 测试

```python
from torch2c.common import Graph, Node, Tensor

def _make_graph(...) -> Graph:
    """构建包含特定模式的最小图。"""
    g = Graph()
    t_in = Tensor(id="t_in", shape=[1, 32, 64], dtype="fp16", format="nd")
    t_out = Tensor(id="t_out", shape=[1, 32, 64], dtype="fp16", format="nd")
    g.add_tensor(t_in)
    g.add_tensor(t_out)
    node = Node(
        id="n0", op_type="aten.xxx",
        inputs=["t_in"], outputs=["t_out"],
        compute_unit="vector", npu_op="vector_xxx", is_mapped=True,
    )
    g.add_node(node)
    g.execution_order = ["n0"]
    return g

class TestRun:
    def test_basic(self):
        """最基本的正常路径。"""
        g = _make_graph()
        run(g, {})
        # 断言变换结果

    def test_no_op(self):
        """不满足条件时不变换。"""
        g = _make_graph_without_pattern()
        run(g, {})
        # 断言图未改变

    def test_idempotent(self):
        """跑两次结果相同。"""

    def test_edge_case(self):
        """边界条件：空图、单节点、特殊 shape。"""
```

### 模式 B: Config 一致性测试

```python
def test_new_op_in_all_tables():
    """新算子出现在所有必要的配置表中。"""
    # 已有 test_config_consistency.py 自动检查
    # 只需运行：pytest torch2c/integration/tests/test_config_consistency.py
```

### 模式 C: 端到端测试

```python
def test_e2e_compile():
    """完整编译链不报错。"""
    from torch2c.integration.pipeline import compile
    model = SimpleModel()
    compile(model, torch.randn(1, 32, 64),
            config_dir=str(INTEGRATION_CONFIG_DIR),
            output_dir=str(tmp_path))
```

## 测试命名规范

```
test_{功能}_{场景}
test_pad_shape_nd_fp16          # 功能 + 具体场景
test_skip_when_already_aligned  # 边界条件
test_error_on_invalid_format    # 错误处理
```

## 测试质量检查

每个测试必须满足：
- [ ] 有 docstring 说明测试什么
- [ ] 断言明确（不是 `assert result`，而是 `assert result == expected`）
- [ ] 测试独立（不依赖其他测试的执行顺序）
- [ ] 测试可重复（固定随机种子或不用随机数）
