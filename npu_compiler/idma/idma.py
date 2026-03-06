"""idma — Pass⑤b：内部 DMA 存储位置分配。

对于裂解产生的中间 tensor，如果满足以下条件，
其 DMA 输出可以不落 HBM，直接进入下游 op 的 local buffer：
  1. 不是外部输入（is_model_input=False）
  2. 不是权重（is_weight=False）
  3. 不是模型输出（is_model_output=False）
  4. 只有一个消费者节点（单 consumer）
  5. producer → consumer 的计算单元对在 allowed_pairs 中

计算单元类型包括：
  "cube"   — 矩阵计算单元
  "vector" — 向量计算单元
  "idma"   — DMA 搬运/格式转换单元

storage 取值：
  "hbm"   — 默认，落主存
  "l2"    — 片上 L2 buffer
  "local" — 直接进下游 op 的 local buffer，不占 HBM

配置示例（hardware_config.yaml）：
  local_bypass:
    allowed_pairs:
      - ["cube", "vector"]     # matmul 输出 → vector op
      - ["cube", "idma"]       # matmul 输出 → DMA reformat
      - ["idma", "vector"]     # DMA reformat 输出 → vector op
      - ["vector", "vector"]   # vector op → vector op
"""

from __future__ import annotations

from npu_compiler.common import Graph, get_logger

logger = get_logger(__name__)

# 没有配置时的默认 allowed_pairs
_DEFAULT_ALLOWED_PAIRS: set[tuple[str, str]] = {
    ("cube", "vector"),
    ("cube", "idma"),
    ("idma", "vector"),
    ("vector", "vector"),
}


def _parse_allowed_pairs(config: dict) -> set[tuple[str, str]]:
    """从配置中解析 allowed_pairs，返回 set of (producer_unit, consumer_unit)。"""
    raw = config.get("allowed_pairs")
    if raw is None:
        return _DEFAULT_ALLOWED_PAIRS
    return {(pair[0], pair[1]) for pair in raw}


def _is_eligible_for_local(
    graph: Graph,
    tensor_id: str,
    allowed_pairs: set[tuple[str, str]],
) -> bool:
    """判断中间 tensor 是否可以不落 HBM，直接进下游 local buffer。"""
    t = graph.get_tensor(tensor_id)
    if t is None:
        return False
    # 外部输入和权重必须从 HBM 搬运，不能融合
    if t.is_model_input or t.is_weight or t.is_model_output:
        return False
    # 只有单消费者才能直接导入
    if len(t.consumer_node_ids) != 1:
        return False
    # 必须有 producer（即是某个 op 的输出）
    if t.producer_node_id is None:
        return False
    # 检查计算单元对是否允许
    producer = graph.get_node(t.producer_node_id)
    consumer = graph.get_node(t.consumer_node_ids[0])
    if producer is None or consumer is None:
        return False
    pair = (producer.compute_unit or "", consumer.compute_unit or "")
    if pair not in allowed_pairs:
        logger.debug(
            "tensor %s: 计算单元对 %s 不在 allowed_pairs 中，保留 hbm",
            tensor_id, pair,
        )
        return False
    return True


def run(graph: Graph, config: dict) -> Graph:
    """为中间 tensor 分配存储位置。

    Args:
        graph: 经过 format_annotator 的 Graph IR。
        config: 配置字典，可选键：
            enable_local_storage: bool — 是否启用 local buffer 优化，默认 True。
            allowed_pairs: list[list[str]] — 允许 local 直通的计算单元对。
                计算单元: "cube", "vector", "idma"。
                默认: cube→vector, cube→idma, idma→vector, vector→vector。

    Returns:
        同一 Graph 对象（原地修改）。
    """
    enable = config.get("enable_local_storage", True)
    if not enable:
        logger.info("local storage 优化已禁用")
        return graph

    allowed_pairs = _parse_allowed_pairs(config)
    logger.info("allowed_pairs: %s", allowed_pairs)

    assigned_count = 0
    for tid in list(graph.tensors):
        if _is_eligible_for_local(graph, tid, allowed_pairs):
            graph.tensors[tid].storage = "local"
            assigned_count += 1
            logger.debug("tensor %s -> storage=local", tid)

    logger.info("存储分配完成。%d 个中间 tensor 标记为 local", assigned_count)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """idma 后的校验：local tensor 不能是外部输入/权重/模型输出。"""
    errors: list[str] = []
    for t in graph.tensors.values():
        if t.storage == "local":
            if t.is_model_input:
                errors.append(f"tensor {t.id} storage=local 但是模型输入")
            if t.is_weight:
                errors.append(f"tensor {t.id} storage=local 但是权重")
            if t.is_model_output:
                errors.append(f"tensor {t.id} storage=local 但是模型输出")
    return errors
