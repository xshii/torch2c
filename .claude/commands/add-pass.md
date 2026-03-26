# 新增优化 Pass

当需要添加一个新的编译器 pass 时，按以下步骤完成。

## 前置决策

1. **必须 pass 还是可选 pass？**
   - 必须 pass → 放在 `torch2c/a_capture/`、`b_lowering/`、`c_backend/`、`d_emission/` 下
   - 可选 pass → 放在 `torch2c/optpass/` 下

2. **在哪两个 pass 之间？** 决定了前缀：
   - `bc_` = B→C 之间（lowering 完成后，backend 之前）
   - `c_` = C 阶段内（format_annotator 之后，validator 之前）
   - `cd_` = C→D 之间（validator 之后，scheduler 之前）
   - `d_` = D 阶段内（scheduler 之后，codegen 之前）

3. **输入/输出是什么？** pass 的 `run(graph, config) -> Graph` 签名统一。

## 步骤清单

### 1. 创建模块目录

```
torch2c/optpass/{prefix}_{pass_name}/
├── __init__.py
├── {pass_name}.py          # 核心逻辑（< 300 行）
└── tests/
    ├── __init__.py
    └── test_{pass_name}.py
```

### 2. 编写 pass 入口 — `{pass_name}.py`

```python
"""pass_name — Pass说明。"""
from __future__ import annotations

from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)


def run(graph: Graph, config: dict) -> Graph:
    """Pass 入口。

    Args:
        graph: 上游 pass 产出的 Graph IR。
        config: 从 pipeline 传入的配置。

    Returns:
        修改后的 Graph（通常原地修改并返回同一对象）。
    """
    # 1. 从 config 读取参数
    # 2. 遍历 graph.nodes / graph.tensors
    # 3. 执行变换
    # 4. 对每个变换，用 log_opt 记录原因：
    #    log_opt(node, "pass_name", "动作", "原因说明")

    logger.info("pass_name 完成: %d 处变换", count)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """Pass 后置校验。返回错误消息列表（空=通过）。"""
    errors: list[str] = []
    # 校验变换后的图是否满足不变量
    return errors
```

### 3. 编写 `__init__.py`

```python
"""pass_name — 一句话说明。"""
from torch2c.optpass.{prefix}_{pass_name} import {pass_name}  # noqa: F401
```

### 4. 注册 pass toggle — `torch2c/common/pass_config.py`

```python
class OptionalPass(Enum):
    # ... 已有的 ...
    MY_PASS = auto()     # 新增
```

同时在 `PassConfig.__init__` 中添加默认值（通常为 True）。

### 5. 接入 pipeline — `torch2c/integration/pipeline.py`

在文件头部添加 import：

```python
from torch2c.optpass.{prefix}_{pass_name} import {pass_name}
```

在 `_OPTIMIZATION_PASSES` 列表中按执行顺序插入：

```python
_PassDesc(
    "pass_name", "编号", {pass_name}.run,
    "config_key", {pass_name}.post_validate,
    toggle=OptionalPass.MY_PASS,
),
```

### 6. 添加 pass 描述 — pipeline.py `_PASS_DESC`

```python
"pass_name": {
    "input": "输入描述",
    "output": "输出描述",
    "desc": "这个 pass 做什么，为什么需要",
},
```

### 7. 添加配置（如需要）

在 `torch2c/integration/config/` 下的相关 YAML 中添加配置段，
并在 `pipeline.py` 的 `_build_pass_configs()` 中传递给 pass。

### 8. 编写测试

```python
"""test_{pass_name} 单元测试。"""
from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.{prefix}_{pass_name}.{pass_name} import run, post_validate


def _make_graph(...) -> Graph:
    """构建测试用最小图。"""
    g = Graph()
    # 添加 node 和 tensor
    return g


class TestRun:
    def test_basic_transform(self):
        g = _make_graph(...)
        result = run(g, {})
        # 断言变换结果

    def test_no_op_when_nothing_to_do(self):
        g = _make_graph(...)
        result = run(g, {})
        # 断言图未变化

    def test_idempotent(self):
        """跑两次结果相同。"""
        g = _make_graph(...)
        run(g, {})
        snap1 = str(g)
        run(g, {})
        snap2 = str(g)
        assert snap1 == snap2


class TestPostValidate:
    def test_clean_after_run(self):
        g = _make_graph(...)
        run(g, {})
        assert post_validate(g) == []
```

### 9. 验证

```bash
# pass 自身测试
.venv/bin/pytest torch2c/optpass/{prefix}_{pass_name}/tests/ -v

# 全量回归
.venv/bin/pytest --tb=short -q
```

## pass 编写规范

1. **只改 graph 不改 config** — pass 是 graph → graph 的纯变换
2. **用 log_opt 记录每个决策** — 可视化和 debug 依赖它
3. **保持幂等** — run 两次结果不变
4. **post_validate 校验不变量** — 而非重复 run 的逻辑
5. **函数 < 50 行** — 超过就拆子函数
