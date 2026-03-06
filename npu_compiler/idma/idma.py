"""idma — Pass⑤b：内部 DMA 存储位置分配。

对于裂解产生的中间 reformat tensor，如果满足以下条件，
其 DMA 输出可以不落 HBM，直接进入下游 op 的 local buffer：
  1. 不是外部输入（is_model_input=False）
  2. 不是权重（is_weight=False）
  3. 不是模型输出（is_model_output=False）
  4. 只有一个消费者节点（单 consumer）

storage 取值：
  "hbm"   — 默认，落主存
  "l2"    — 片上 L2 buffer
  "local" — 直接进下游 op 的 local buffer，不占 HBM
"""

from __future__ import annotations

from npu_compiler.common import Graph, get_logger

logger = get_logger(__name__)


def _is_eligible_for_local(graph: Graph, tensor_id: str) -> bool:
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
    return True


def run(graph: Graph, config: dict) -> Graph:
    """为中间 tensor 分配存储位置。

    Args:
        graph: 经过 format_annotator 的 Graph IR。
        config: 配置字典，可选键：
            enable_local_storage: bool — 是否启用 local buffer 优化，默认 True。

    Returns:
        同一 Graph 对象（原地修改）。
    """
    enable = config.get("enable_local_storage", True)
    if not enable:
        logger.info("local storage 优化已禁用")
        return graph

    assigned_count = 0
    for tid in list(graph.tensors):
        if _is_eligible_for_local(graph, tid):
            graph.tensors[tid].storage = "local"
            assigned_count += 1
            logger.debug("tensor %s -> storage=local", tid)

    logger.info("存储分配完成。%d 个中间 tensor 标记为 local", assigned_count)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """storage_assigner 后的校验：local tensor 不能是外部输入/权重/模型输出。"""
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
