"""reformat_inserter — Pass⑤a：自动插入 dma_reformat 节点。

当中间 tensor 的格式与下游节点所需格式不匹配时，自动插入
dma_reformat 节点（compute_unit=idma）完成格式转换。

外部输入（is_model_input）和权重（is_weight）的格式转换由
DMA load 随路完成，不需要插入额外节点。

规则：
  - cube 计算单元要求输入/输出为 nz 格式
  - vector 计算单元要求输入/输出为 nd 格式
  - 仅对中间 tensor（非 input / weight）做检查
"""

from __future__ import annotations

from npu_compiler.common import Graph, Node, Tensor, get_logger

logger = get_logger(__name__)

_DEFAULT_UNIT_FORMATS: dict[str, str] = {
    "cube": "nz",
    "vector": "nd",
}


def run(graph: Graph, config: dict) -> Graph:
    """扫描格式不匹配的边，在 consumer 之前插入 dma_reformat 节点。

    config 可选键：
        compute_unit_formats: dict[str, str] — 计算单元到所需格式的映射。
            默认 {"cube": "nz", "vector": "nd"}。
    """
    unit_formats = config.get("compute_unit_formats", _DEFAULT_UNIT_FORMATS)

    inserted = 0
    new_order: list[str] = []
    used_ids: set[str] = set(graph.tensors.keys()) | set(graph.nodes.keys())

    for node_id in list(graph.execution_order):
        node = graph.nodes[node_id]
        required_fmt = unit_formats.get(node.compute_unit or "")

        if required_fmt is None:
            new_order.append(node_id)
            continue

        # 检查每个输入 tensor 是否需要格式转换
        for i, input_tid in enumerate(list(node.inputs)):
            tensor = graph.tensors[input_tid]

            # 外部输入和权重由 DMA load 随路转换
            if tensor.is_model_input or tensor.is_weight:
                continue

            # 格式已匹配
            if tensor.format == required_fmt:
                continue

            # 生成唯一 ID
            rf_node_id = _unique_id(f"reformat_{input_tid}", used_ids)
            rf_tensor_id = _unique_id(f"{input_tid}_{required_fmt}", used_ids)

            # 创建转换后的中间 tensor
            graph.add_tensor(Tensor(
                id=rf_tensor_id,
                shape=list(tensor.shape),
                dtype=tensor.dtype,
                format=required_fmt,
                producer_node_id=rf_node_id,
                consumer_node_ids=[node_id],
            ))

            # 创建 reformat 节点
            graph.add_node(Node(
                id=rf_node_id,
                op_type="dma_reformat",
                inputs=[input_tid],
                outputs=[rf_tensor_id],
                compute_unit="idma",
                npu_op="dma_reformat",
                is_mapped=True,
                format_annotation={
                    "inputs": [{"format": tensor.format, "dtype": tensor.dtype}],
                    "outputs": [{"format": required_fmt, "dtype": tensor.dtype}],
                },
            ))

            # 更新原 tensor 的 consumer 列表
            if node_id in tensor.consumer_node_ids:
                tensor.consumer_node_ids.remove(node_id)
            tensor.consumer_node_ids.append(rf_node_id)

            # 更新消费者节点的 inputs
            node.inputs[i] = rf_tensor_id

            # reformat 节点插在消费者之前
            new_order.append(rf_node_id)

            inserted += 1
            logger.debug(
                "插入 reformat: %s(%s→%s) 在 %s 之前",
                rf_node_id, tensor.format, required_fmt, node_id,
            )

        new_order.append(node_id)

    graph.execution_order = new_order
    logger.info("reformat_inserter 完成，插入了 %d 个 reformat 节点", inserted)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """校验 dma_reformat 节点的格式一致性。"""
    errors: list[str] = []
    for node in graph.nodes.values():
        if node.op_type != "dma_reformat":
            continue
        if not node.format_annotation:
            errors.append(f"reformat 节点 {node.id} 缺少 format_annotation")
            continue
        in_fmt = node.format_annotation["inputs"][0]["format"]
        out_fmt = node.format_annotation["outputs"][0]["format"]
        if in_fmt == out_fmt:
            errors.append(
                f"reformat 节点 {node.id} 输入输出格式相同({in_fmt})，无需转换"
            )
    return errors


def _unique_id(base: str, used: set[str]) -> str:
    """生成不与已有 ID 冲突的唯一标识。"""
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base}_{n}" in used:
        n += 1
    uid = f"{base}_{n}"
    used.add(uid)
    return uid
