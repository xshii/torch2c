"""CodegenPlan — codegen 层的类型化输入接口。

替代 graph.to_dict() 返回的裸 dict，让 codegen 模块通过属性访问
Graph/Node/Tensor 字段而非 dict.get()。
"""

from __future__ import annotations

from dataclasses import asdict

from torch2c.common import DmaPlan, Graph, Node, Tensor


class CodegenPlan:
    """codegen 的输入：Graph + DMA 计划。

    提供类型化属性访问：plan.graph, plan.nodes, plan.tensors, plan.dma_map。
    """

    def __init__(self, graph: Graph, dma_plans: list[DmaPlan]):
        self.graph = graph
        self.dma_plans = dma_plans
        # 按 node_id 索引的 DMA 计划 dict（codegen 常用）
        self.dma_map: dict[str, dict] = {
            dp.node_id: asdict(dp) for dp in dma_plans
        }

    @property
    def nodes(self) -> dict[str, Node]:
        return self.graph.nodes

    @property
    def tensors(self) -> dict[str, Tensor]:
        return self.graph.tensors

    @property
    def execution_order(self) -> list[str]:
        return self.graph.execution_order
