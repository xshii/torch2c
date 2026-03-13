"""CompilerPass Protocol — 编译器 Pass 的接口约定。

所有 Pass 模块应导出 run(graph, config) -> Graph。
post_validate(graph) -> list[str] 可选（校验阶段产出）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .graph_ir import Graph


@runtime_checkable
class CompilerPass(Protocol):
    """编译器 Pass 的通用接口。"""

    def run(self, graph: Graph, config: dict) -> Graph: ...


@runtime_checkable
class ValidatablePass(Protocol):
    """带后校验的 Pass 接口。"""

    def run(self, graph: Graph, config: dict) -> Graph: ...

    def post_validate(self, graph: Graph) -> list[str]: ...
