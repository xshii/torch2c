"""算子裂解：将未映射 ATen 算子按固定规则替换为多个 NPU 算子节点。

处理逻辑：
1. 遍历图中所有未映射节点
2. 如果节点 op_type 在裂解规则中，按 steps 创建新节点组
3. 中间 tensor shape = 源算子第一个输入的 shape（§16.5）
4. 删除原节点，新节点标记 is_mapped=True
5. 对已映射的逐元素算子，若两个输入 shape 不同则插入 idma_broadcast
"""

from __future__ import annotations

import math

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
    targets = [n for n in graph.nodes.values() if not n.is_mapped and n.op_type in rules]

    total_new_nodes = 0
    total_new_tensors = 0

    for node in targets:
        nn_count, nt_count = _decompose_node(graph, node, rules[node.op_type])
        total_new_nodes += nn_count
        total_new_tensors += nt_count

    logger.info(
        "裂解完成。裂解了%d个算子，新增%d个节点，新增%d个中间tensor",
        len(targets),
        total_new_nodes,
        total_new_tensors,
    )

    # 广播展开：逐元素算子的输入 shape 不匹配时插入 idma_broadcast
    _insert_broadcast_expands(graph)

    return graph


def _create_intermediates(
    graph: Graph, node: Node, steps: list[dict],
) -> tuple[list[str], list[int], str]:
    """创建中间 tensor（steps 之间各一个），返回 (intermediate_tids, inter_shape, inter_dtype)。"""
    first_input_tid = node.inputs[0] if node.inputs else None
    first_input = graph.get_tensor(first_input_tid) if first_input_tid else None
    inter_shape = list(first_input.shape) if first_input else [1]
    inter_dtype = first_input.dtype if first_input else "fp16"

    intermediates: list[str] = []
    for i in range(len(steps) - 1):
        tid = f"{node.id}_inter_{i}"
        graph.add_tensor(Tensor(id=tid, shape=list(inter_shape), dtype=inter_dtype))
        intermediates.append(tid)
    return intermediates, inter_shape, inter_dtype


def _build_step_nodes(
    node: Node, steps: list[dict], intermediates: list[str],
) -> list[Node]:
    """按 steps 构建新节点列表（含 extra_inputs 解析）。"""
    new_nodes: list[Node] = []
    for i, step in enumerate(steps):
        nid = f"{node.id}_step_{i}"

        if i == 0:
            inputs = list(node.inputs)
        else:
            inputs = [intermediates[i - 1]]
            for ref in step.get("extra_inputs", []):
                parts = ref.split(".")
                if parts[0] == "original" and node.inputs:
                    idx = int(parts[1])
                    if idx < len(node.inputs):
                        inputs.append(node.inputs[idx])

        if i == len(steps) - 1:
            outputs = [node.outputs[0]] if node.outputs else []
        else:
            outputs = [intermediates[i]]

        new_nodes.append(
            Node(
                id=nid,
                op_type=step["npu_op"],
                inputs=inputs,
                outputs=outputs,
                params=dict(node.params),
                npu_op=step["npu_op"],
                compute_unit=step["compute_unit"],
                is_mapped=True,
            )
        )
    return new_nodes


def _rewire_graph(graph: Graph, node: Node, new_nodes: list[Node], intermediates: list[str]) -> None:
    """更新 tensor 引用、execution_order，替换节点。"""
    # 中间 tensor 的 producer/consumer
    for i, inter_tid in enumerate(intermediates):
        t = graph.get_tensor(inter_tid)
        assert t is not None
        t.producer_node_id = new_nodes[i].id
        t.consumer_node_ids = [new_nodes[i + 1].id]

    # 原输入 tensor：移除旧节点，关联新 step_0
    for input_tid in node.inputs:
        t = graph.get_tensor(input_tid)
        if t and node.id in t.consumer_node_ids:
            t.consumer_node_ids.remove(node.id)
        if t and new_nodes[0].id not in t.consumer_node_ids:
            t.consumer_node_ids.append(new_nodes[0].id)

    # 后续 step 引用的原始输入更新 consumer
    orig_input_set = set(node.inputs)
    for new_n in new_nodes[1:]:
        for tid in new_n.inputs:
            if tid in orig_input_set:
                t = graph.get_tensor(tid)
                if t and new_n.id not in t.consumer_node_ids:
                    t.consumer_node_ids.append(new_n.id)

    # 更新第一个输出 tensor 的 producer
    if node.outputs:
        first_out = graph.get_tensor(node.outputs[0])
        if first_out:
            first_out.producer_node_id = new_nodes[-1].id

    # 其余输出 tensor 失去 producer
    for out_tid in node.outputs[1:]:
        t = graph.get_tensor(out_tid)
        if t:
            t.producer_node_id = None

    # 更新 execution_order
    if node.id in graph.execution_order:
        idx = graph.execution_order.index(node.id)
        graph.execution_order.remove(node.id)
        for i, new_n in enumerate(new_nodes):
            graph.execution_order.insert(idx + i, new_n.id)

    # 替换节点
    graph.nodes.pop(node.id, None)
    for new_n in new_nodes:
        graph.nodes[new_n.id] = new_n


def _decompose_node(graph: Graph, node: Node, rule: dict) -> tuple[int, int]:
    """裂解单个节点，返回 (新节点数, 新中间tensor数)。"""
    steps = rule["steps"]
    intermediates, _, _ = _create_intermediates(graph, node, steps)
    new_nodes = _build_step_nodes(node, steps, intermediates)
    _rewire_graph(graph, node, new_nodes, intermediates)
    return len(new_nodes), len(intermediates)


# ---- 广播展开 ----

_ELEMENTWISE_OPS = {"vector_add", "vector_sub", "vector_mul", "vector_div"}


def _needs_broadcast(graph: Graph, node: Node) -> int | None:
    """检查逐元素算子的两个输入是否需要广播。

    Returns:
        需要广播的输入索引 (0 或 1)，不需要则返回 None。
    """
    if node.npu_op not in _ELEMENTWISE_OPS or len(node.inputs) < 2:
        return None
    t0 = graph.get_tensor(node.inputs[0])
    t1 = graph.get_tensor(node.inputs[1])
    if t0 is None or t1 is None:
        return None
    n0 = math.prod(t0.shape)
    n1 = math.prod(t1.shape)
    if n0 == n1:
        return None
    # 返回较小那个的索引
    return 1 if n1 < n0 else 0


def _broadcast_shape(small_shape: list[int], big_shape: list[int]) -> list[int]:
    """计算广播后的 shape（结果 = big_shape）。"""
    return list(big_shape)


def _insert_broadcast(graph: Graph, node: Node, input_idx: int) -> None:
    """在 node 的第 input_idx 个输入前插入 idma_broadcast 节点。"""
    small_tid = node.inputs[input_idx]
    big_tid = node.inputs[1 - input_idx]
    small_t = graph.get_tensor(small_tid)
    big_t = graph.get_tensor(big_tid)
    if small_t is None or big_t is None:
        return

    # 创建广播输出 tensor
    bc_out_tid = f"{node.id}_bc_{input_idx}"
    bc_shape = _broadcast_shape(small_t.shape, big_t.shape)
    bc_out = Tensor(
        id=bc_out_tid, shape=bc_shape, dtype=small_t.dtype,
        producer_node_id=f"{node.id}_broadcast",
    )
    graph.add_tensor(bc_out)

    # 创建 idma_broadcast 节点
    bc_nid = f"{node.id}_broadcast"
    bc_node = Node(
        id=bc_nid, op_type="aten.expand.default",
        inputs=[small_tid], outputs=[bc_out_tid],
        npu_op="idma_broadcast", compute_unit="idma", is_mapped=True,
    )
    graph.add_node(bc_node)

    # 更新 execution_order：broadcast 在原节点之前
    if node.id in graph.execution_order:
        idx = graph.execution_order.index(node.id)
        graph.execution_order.remove(bc_nid)  # add_node 会追加到末尾
        graph.execution_order.insert(idx, bc_nid)

    # 更新 tensor 引用
    small_t.consumer_node_ids.append(bc_nid)
    bc_out.consumer_node_ids.append(node.id)
    if node.id in small_t.consumer_node_ids:
        small_t.consumer_node_ids.remove(node.id)

    # 替换原节点的输入
    node.inputs[input_idx] = bc_out_tid

    logger.debug("插入广播: %s → %s (shape %s → %s)", bc_nid, node.id, small_t.shape, bc_shape)


def _insert_broadcast_expands(graph: Graph) -> int:
    """扫描所有逐元素算子，对 shape 不匹配的输入插入 idma_broadcast。"""
    count = 0
    for node in list(graph.nodes.values()):
        idx = _needs_broadcast(graph, node)
        if idx is not None:
            _insert_broadcast(graph, node, idx)
            count += 1
    if count:
        logger.info("插入了 %d 个广播展开节点", count)
    return count


def post_validate(graph: Graph) -> list[str]:
    """op_decomposition 后的校验：所有节点必须已映射并有 npu_op/compute_unit。"""
    errors: list[str] = []
    for n in graph.nodes.values():
        if not n.is_mapped:
            errors.append(f"节点 {n.id} 未映射 (is_mapped=False)")
        if not n.npu_op:
            errors.append(f"节点 {n.id} 缺少 npu_op")
        if not n.compute_unit:
            errors.append(f"节点 {n.id} 缺少 compute_unit")
    return errors
