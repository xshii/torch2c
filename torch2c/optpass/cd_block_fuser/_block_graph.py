"""块级数据流图 — pass 内部临时结构。

将 Graph IR 转为以数据搬运为中心的视图：
  DataBlock = tensor 在 HBM↔L1 间的搬运成本模型
  ComputeBlock = 算子的计算 + DMA 成本
  BlockGraph = 二者的连接关系

不持久化到 Graph IR，只在 block_fuser pass 内部使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul

from torch2c.common import Graph, get_logger
from torch2c.common.sizing import calc_padded_size, get_dim_align
from torch2c.optpass.cd_roofline.roofline_analyzer import (
    CostModel, RooflineHwParams, parse_cost_model,
)

logger = get_logger(__name__)

# 可 tile 的算子集合（和 global_tiler 一致）
_TILEABLE_OPS = frozenset({
    "cube_matmul", "cube_matmul_bias",
    "dma_reformat",
    "vector_add", "vector_mul", "vector_sub", "vector_div",
    "vector_relu", "vector_gelu", "vector_mul_scalar",
    "vector_fill", "vector_dropout",
    "vector_layernorm_part1", "vector_layernorm_part2",
    "vector_softmax_part1", "vector_softmax_part2",
    "vector_rmsnorm_part1", "vector_rmsnorm_part2",
    "idma_reshape", "idma_broadcast",
})


def _prod(shape: list[int]) -> int:
    return reduce(mul, shape, 1) if shape else 0


def _tensor_padded_bytes(t) -> int:
    """计算 tensor 的 padded 字节数。"""
    return calc_padded_size(
        t.shape, t.dtype, t.format,
        get_dim_align(t.format, t.dtype),
    )


@dataclass
class DataBlock:
    """tensor 在内存层级间的搬运模型。"""

    tensor_id: str
    size_bytes: int                # padded size
    producer_id: str | None        # 生产者节点 id
    consumer_ids: list[str]        # 消费者节点 id 列表
    current_tier: str              # hbm / local / pipe
    is_external: bool              # model_input / model_output / weight

    @property
    def elimination_benefit(self) -> int:
        """融合收益 = 消除这条 HBM 搬运省多少 bytes（load + store）。"""
        if self.is_external or self.current_tier != "hbm":
            return 0
        return 2 * self.size_bytes

    @property
    def l1_pressure(self) -> int:
        """L1 压力 = 在 L1 中占用的 bytes。"""
        return self.size_bytes


@dataclass
class ComputeBlock:
    """算子在块级图中的表示。"""

    node_id: str
    compute_unit: str
    npu_op: str
    compute_cycles: int            # from roofline / cost_model
    launch_cycles: int
    input_block_ids: list[str]     # DataBlock.tensor_id
    output_block_ids: list[str]
    # tiling
    tileable: bool
    tile_dim: int                  # 哪个维度可 tile（通常 -2）
    tile_dim_size: int             # 该维度原始大小


@dataclass
class BlockGraph:
    """块级数据流图。"""

    compute_blocks: dict[str, ComputeBlock] = field(default_factory=dict)
    data_blocks: dict[str, DataBlock] = field(default_factory=dict)
    topo_order: list[str] = field(default_factory=list)

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        hw: RooflineHwParams,
        cost_model: CostModel,
    ) -> BlockGraph:
        """从 Graph IR 构建 BlockGraph。"""
        bg = cls()
        bg.topo_order = list(graph.execution_order)

        # 构建 DataBlock
        for tid, t in graph.tensors.items():
            is_ext = t.is_model_input or t.is_model_output or t.is_weight
            bg.data_blocks[tid] = DataBlock(
                tensor_id=tid,
                size_bytes=_tensor_padded_bytes(t),
                producer_id=t.producer_node_id,
                consumer_ids=list(t.consumer_node_ids),
                current_tier=t.storage or "hbm",
                is_external=is_ext,
            )

        # 构建 ComputeBlock
        for nid in bg.topo_order:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            op = node.npu_op or node.op_type
            cu = (node.compute_unit or "vector").lower()

            # 从 cost_model 获取 cycles
            cp = cost_model.get(op, cu)
            if cu == "cube":
                # matmul flops
                if len(node.inputs) >= 2:
                    a = graph.tensors.get(node.inputs[0])
                    b = graph.tensors.get(node.inputs[1])
                    if a and b:
                        m = a.shape[-2] if len(a.shape) >= 2 else 1
                        k = a.shape[-1] if len(a.shape) >= 1 else 1
                        n = b.shape[-1] if len(b.shape) >= 1 else 1
                        batch = _prod(a.shape[:-2]) if len(a.shape) > 2 else 1
                        compute_cy = (2 * batch * m * n * k) // hw.cube_ops_per_cycle
                    else:
                        compute_cy = 0
                else:
                    compute_cy = 0
            elif cu in ("dma", "idma"):
                compute_cy = 0
            else:
                # vector elementwise
                elem = 0
                for out_tid in node.outputs:
                    ot = graph.tensors.get(out_tid)
                    if ot:
                        elem = _prod(ot.shape)
                        break
                compute_cy = (elem * cp.flops_multiplier) // hw.vector_ops_per_cycle

            # tile 信息
            tileable = op in _TILEABLE_OPS
            tile_dim = -2
            tile_dim_size = 1
            if tileable and node.inputs:
                first_input = graph.tensors.get(node.inputs[0])
                if first_input and len(first_input.shape) >= 2:
                    tile_dim_size = first_input.shape[-2]

            bg.compute_blocks[nid] = ComputeBlock(
                node_id=nid,
                compute_unit=cu,
                npu_op=op,
                compute_cycles=compute_cy,
                launch_cycles=cp.launch_cycles,
                input_block_ids=list(node.inputs),
                output_block_ids=list(node.outputs),
                tileable=tileable,
                tile_dim=tile_dim,
                tile_dim_size=tile_dim_size,
            )

        return bg

    def get_fusible_edges(self) -> list[DataBlock]:
        """获取所有可融合的边，按 elimination_benefit 降序排列。"""
        edges = []
        for db in self.data_blocks.values():
            if db.elimination_benefit > 0 and db.producer_id and db.consumer_ids:
                edges.append(db)
        edges.sort(key=lambda db: db.elimination_benefit, reverse=True)
        return edges
