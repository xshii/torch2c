"""算子裂解：将未映射 ATen 算子按固定规则替换为多个 NPU 算子节点。

处理逻辑：
1. 遍历图中所有未映射节点
2. 如果节点 op_type 在裂解规则中，按 steps 创建新节点组
3. 中间 tensor shape = 源算子第一个输入的 shape（§16.5）
4. 删除原节点，新节点标记 is_mapped=True
"""

from __future__ import annotations

from ..common.graph_ir import Graph, Node, Tensor
from ..common.logger import get_logger

logger = get_logger(__name__)


def run(graph: Graph, config: dict) -> Graph:
    """对图执行算子裂解。

    Args:
        graph: 输入 Graph IR（经过 op_mapping，部分节点已映射）。
        config: 裂解配置字典，需包含 ``decompositions`` 键。

    Returns:
        同一 Graph 对象（原地修改），裂解后的节点替换原节点。
    """
    rules = config.get("decompositions", {})

    # 收集待裂解节点（不能在迭代时修改 dict）
    targets = [n for n in graph.nodes.values()
               if not n.is_mapped and n.op_type in rules]

    total_new_nodes = 0
    total_new_tensors = 0

    for node in targets:
        nn_count, nt_count = _decompose_node(graph, node, rules[node.op_type])
        total_new_nodes += nn_count
        total_new_tensors += nt_count

    logger.info("裂解完成。裂解了%d个算子，新增%d个节点，新增%d个中间tensor",
                len(targets), total_new_nodes, total_new_tensors)
    return graph


def _decompose_node(graph: Graph, node: Node, rule: dict) -> tuple[int, int]:
    """裂解单个节点，返回 (新节点数, 新中间tensor数)。"""
    steps = rule["steps"]

    # §16.5：中间 tensor shape = 源算子第一个输入的 shape
    first_input_tid = node.inputs[0] if node.inputs else None
    first_input = graph.get_tensor(first_input_tid) if first_input_tid else None
    inter_shape = list(first_input.shape) if first_input else [1]
    inter_dtype = first_input.dtype if first_input else "fp16"

    # 创建中间 tensor（steps 之间各一个）
    intermediates: list[str] = []
    for i in range(len(steps) - 1):
        tid = f"{node.id}_inter_{i}"
        graph.add_tensor(Tensor(id=tid, shape=list(inter_shape), dtype=inter_dtype))
        intermediates.append(tid)

    # 创建新节点
    new_nodes: list[Node] = []
    for i, step in enumerate(steps):
        nid = f"{node.id}_step_{i}"

        if i == 0:
            inputs = list(node.inputs)
        else:
            inputs = [intermediates[i - 1]]

        if i == len(steps) - 1:
            outputs = [node.outputs[0]] if node.outputs else []
        else:
            outputs = [intermediates[i]]

        new_nodes.append(Node(
            id=nid,
            op_type=step["npu_op"],
            inputs=inputs,
            outputs=outputs,
            params=dict(node.params),
            npu_op=step["npu_op"],
            compute_unit=step["compute_unit"],
            is_mapped=True,
        ))

    # 更新中间 tensor 的 producer/consumer 引用
    for i, inter_tid in enumerate(intermediates):
        t = graph.get_tensor(inter_tid)
        t.producer_node_id = new_nodes[i].id
        t.consumer_node_ids = [new_nodes[i + 1].id]

    # 更新原输入 tensor：移除旧节点，关联新 step_0
    for input_tid in node.inputs:
        t = graph.get_tensor(input_tid)
        if t and node.id in t.consumer_node_ids:
            t.consumer_node_ids.remove(node.id)
        if t and new_nodes[0].id not in t.consumer_node_ids:
            t.consumer_node_ids.append(new_nodes[0].id)

    # 更新第一个输出 tensor 的 producer
    if node.outputs:
        first_out = graph.get_tensor(node.outputs[0])
        if first_out:
            first_out.producer_node_id = new_nodes[-1].id

    # 其余输出 tensor 失去 producer（由 memory_planner 回收）
    for out_tid in node.outputs[1:]:
        t = graph.get_tensor(out_tid)
        if t:
            t.producer_node_id = None

    # 更新 execution_order：在原位置插入新节点
    if node.id in graph.execution_order:
        idx = graph.execution_order.index(node.id)
        graph.execution_order.remove(node.id)
        for i, new_n in enumerate(new_nodes):
            graph.execution_order.insert(idx + i, new_n.id)

    # 替换节点
    graph.nodes.pop(node.id, None)
    for new_n in new_nodes:
        graph.nodes[new_n.id] = new_n

    return len(new_nodes), len(intermediates)
