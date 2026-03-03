"""op_absorption — Pass④：将独立算子吸收为相邻算子的可选参数。"""

from __future__ import annotations

from npu_compiler.common import Graph, get_logger, load_config

logger = get_logger(__name__)

_CONFIG_REQUIRED_KEYS = ["absorptions"]


def load_absorption_config(path: str) -> dict:
    """加载吸收规则配置。"""
    return load_config(path, required_keys=_CONFIG_REQUIRED_KEYS)


def _find_single_consumer(graph: Graph, tensor_id: str) -> str | None:
    """找到 tensor 的唯一消费者节点 ID，若有多个返回 None。"""
    tensor = graph.get_tensor(tensor_id)
    if tensor is None:
        return None
    consumers = tensor.consumer_node_ids
    if len(consumers) == 1:
        return consumers[0]
    return None


def _try_absorb(graph: Graph, rule: dict) -> tuple[int, int]:
    """尝试按一条规则执行吸收，返回 (吸收算子数, 消除tensor数)。"""
    absorbed_op = rule["absorbed_op"]
    target_op = rule["target_op"]
    param_name = rule["param_name"]
    absorbed_idx = rule["absorbed_input_index"]
    passthrough_idx = rule["passthrough_input_index"]

    nodes_to_remove: list[str] = []
    tensors_to_remove: list[str] = []

    for node in list(graph.nodes.values()):
        if node.npu_op != absorbed_op:
            continue
        if len(node.outputs) != 1:
            continue

        out_tid = node.outputs[0]
        consumer_id = _find_single_consumer(graph, out_tid)
        if consumer_id is None:
            continue

        consumer = graph.get_node(consumer_id)
        if consumer is None or consumer.npu_op != target_op:
            continue

        # 执行吸收
        param_tid = node.inputs[absorbed_idx]
        passthrough_tid = node.inputs[passthrough_idx]

        consumer.absorbed_inputs[param_name] = param_tid
        # 消费者的对应输入改为被吸收节点的直通输入
        for i, tid in enumerate(consumer.inputs):
            if tid == out_tid:
                consumer.inputs[i] = passthrough_tid
                break
        # 更新 passthrough tensor 的消费者列表
        pt = graph.get_tensor(passthrough_tid)
        if pt and consumer_id not in pt.consumer_node_ids:
            pt.consumer_node_ids.append(consumer_id)

        nodes_to_remove.append(node.id)
        tensors_to_remove.append(out_tid)

    for nid in nodes_to_remove:
        graph.remove_node(nid)
    for tid in tensors_to_remove:
        graph.remove_tensor(tid)

    return len(nodes_to_remove), len(tensors_to_remove)


def run(graph: Graph, config: dict) -> Graph:
    """执行参数吸收 pass。"""
    total_absorbed = 0
    total_tensors = 0

    for rule in config.get("absorptions", []):
        a, t = _try_absorb(graph, rule)
        total_absorbed += a
        total_tensors += t

    logger.info("吸收完成。吸收了%d个算子，消除了%d个中间tensor", total_absorbed, total_tensors)
    return graph
