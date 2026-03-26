"""op_absorption — Pass④：将独立算子吸收为相邻算子的可选参数。"""

from __future__ import annotations

from torch2c.common import Graph, get_logger, load_config
from torch2c.common.constants import MATMUL_OPS as _MATMUL_OPS
from torch2c.common.opt_log import log_opt

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


def _absorb_one(graph: Graph, node, consumer, consumer_id: str, rule: dict) -> str:
    """执行单次吸收：重连 consumer 输入，清理旧引用。返回待删除的中间 tensor id。"""
    param_tid = node.inputs[rule["absorbed_input_index"]]
    passthrough_tid = node.inputs[rule["passthrough_input_index"]]
    out_tid = node.outputs[0]

    consumer.absorbed_inputs[rule["param_name"]] = param_tid
    for i, tid in enumerate(consumer.inputs):
        if tid == out_tid:
            consumer.inputs[i] = passthrough_tid
            break

    for tid in node.inputs:
        t = graph.get_tensor(tid)
        if t and node.id in t.consumer_node_ids:
            t.consumer_node_ids.remove(node.id)

    pt = graph.get_tensor(passthrough_tid)
    if pt and consumer_id not in pt.consumer_node_ids:
        pt.consumer_node_ids.append(consumer_id)
    at = graph.get_tensor(param_tid)
    if at and consumer_id not in at.consumer_node_ids:
        at.consumer_node_ids.append(consumer_id)

    return out_tid


def _try_absorb(graph: Graph, rule: dict) -> tuple[int, int]:
    """尝试按一条规则执行吸收，返回 (吸收算子数, 消除tensor数)。"""
    absorbed_op = rule["absorbed_op"]
    target_op = rule["target_op"]
    nodes_to_remove: list[str] = []
    tensors_to_remove: list[str] = []

    for node in list(graph.nodes.values()):
        if node.npu_op != absorbed_op or len(node.outputs) != 1:
            continue
        consumer_id = _find_single_consumer(graph, node.outputs[0])
        if consumer_id is None:
            continue
        consumer = graph.get_node(consumer_id)
        if consumer is None or consumer.npu_op != target_op:
            continue

        param_tid = node.inputs[rule["absorbed_input_index"]]
        param_t = graph.get_tensor(param_tid)
        param_shape = param_t.shape if param_t else []
        out_tid = _absorb_one(graph, node, consumer, consumer_id, rule)
        log_opt(
            consumer, "op_absorption", "参数吸收",
            f"Cube 指令原生支持 bias 累加，将 {node.npu_op} 的 {param_tid}{param_shape} "
            f"作为 {rule['param_name']} 参数融入 {consumer.npu_op}，"
            f"省去 1 次独立 Vector 运算 + 1 次 HBM 读写（消除 {out_tid}）",
        )
        nodes_to_remove.append(node.id)
        tensors_to_remove.append(out_tid)

    for nid in nodes_to_remove:
        graph.remove_node(nid)
    for tid in tensors_to_remove:
        graph.remove_tensor(tid)

    return len(nodes_to_remove), len(tensors_to_remove)


def _absorb_weight_transposes(graph: Graph) -> int:
    """吸收权重 transpose 节点。

    当 idma_transpose 的输入是权重 tensor 时，删除该节点，
    消费者直接引用原始权重，并注入 transpose_b=1。
    DMA 随路搬运的 ND→NZ format 转换自动处理转置。
    """
    nodes_to_remove: list[str] = []
    tensors_to_remove: list[str] = []

    for node in list(graph.nodes.values()):
        if node.npu_op != "idma_transpose" or len(node.inputs) != 1:
            continue
        weight_tid = node.inputs[0]
        weight_t = graph.get_tensor(weight_tid)
        if weight_t is None or not weight_t.is_weight:
            continue
        if len(node.outputs) != 1:
            continue

        out_tid = node.outputs[0]
        out_t = graph.get_tensor(out_tid)
        if out_t is None:
            continue

        # 将消费者的输入从 transpose 输出改为原始权重
        for consumer_id in list(out_t.consumer_node_ids):
            consumer = graph.get_node(consumer_id)
            if consumer is None:
                continue
            for i, tid in enumerate(consumer.inputs):
                if tid == out_tid:
                    consumer.inputs[i] = weight_tid
            if consumer_id not in weight_t.consumer_node_ids:
                weight_t.consumer_node_ids.append(consumer_id)
            # 消费者是 matmul 类算子时注入 transpose_b=1
            if consumer.npu_op in _MATMUL_OPS:
                consumer.params["transpose_b"] = 1
                log_opt(
                    consumer, "op_absorption", "权重转置吸收",
                    f"DMA 搬运 ND→NZ 格式时硬件自动完成转置，"
                    f"无需显式 transpose 节点，省去 1 次 Vector 计算 + 中间 buffer",
                )

        # 清理 transpose 节点的引用
        if node.id in weight_t.consumer_node_ids:
            weight_t.consumer_node_ids.remove(node.id)

        nodes_to_remove.append(node.id)
        tensors_to_remove.append(out_tid)

    for nid in nodes_to_remove:
        graph.remove_node(nid)
    for tid in tensors_to_remove:
        graph.remove_tensor(tid)

    if nodes_to_remove:
        logger.info("吸收了 %d 个权重 transpose（DMA ND→NZ 随路转置）", len(nodes_to_remove))

    return len(nodes_to_remove)


def post_validate(graph: Graph) -> list[str]:
    """op_absorption 后的校验：被吸收节点已清理，absorbed_inputs 引用的 tensor 存在。"""
    errors: list[str] = []
    for node in graph.nodes.values():
        for param_name, tid in node.absorbed_inputs.items():
            if tid not in graph.tensors:
                errors.append(
                    f"节点 {node.id} 的 absorbed_inputs[{param_name}] 引用了不存在的 tensor {tid}"
                )
    return errors


def run(graph: Graph, config: dict) -> Graph:
    """执行参数吸收 pass。"""
    total_absorbed = 0
    total_tensors = 0

    # 权重 transpose 吸收（DMA ND→NZ 随路转置处理）
    wt_absorbed = _absorb_weight_transposes(graph)
    total_absorbed += wt_absorbed
    total_tensors += wt_absorbed  # 每个吸收的 transpose 消除一个中间 tensor

    for rule in config.get("absorptions", []):
        a, t = _try_absorb(graph, rule)
        total_absorbed += a
        total_tensors += t

    logger.info("吸收完成。吸收了%d个算子，消除了%d个中间tensor", total_absorbed, total_tensors)
    return graph
