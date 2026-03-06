"""CompilerPass Protocol — 编译器 Pass 的接口约定。

文档性质的协议定义。大多数 Pass 遵循 run(graph, config) -> Graph，
但 memory_planner 返回 tuple、scheduler config 可选，因此不做强制适配。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .graph_ir import Graph


@runtime_checkable
class CompilerPass(Protocol):
    """编译器 Pass 的通用接口。"""

    def run(self, graph: Graph, config: dict) -> Graph: ...
