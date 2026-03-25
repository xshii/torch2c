"""format_planner — Pass⑤a：基于硬件能力的图级最优 format 分配。

读取 format_capabilities 配置 + 用户 npu() hints，贪心前向传播为每个 tensor
选择最优 format，使 reformat 插入次数最小化。

算法：
  1. 拓扑序遍历所有节点
  2. 对每个输入 tensor，检查是否满足当前节点的 src 格式要求
  3. 对每个输出 tensor，用 consumer lookahead 选最优 dst 格式
  4. 填充 node.format_annotation（包含 format + dtype）
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt
from torch2c.optpass.c_format_planner._capabilities import (
    FormatCapabilities, UnitCapability, parse_capabilities,
)

logger = get_logger(__name__)


def _get_non_absorbed_inputs(node, graph: Graph) -> list[str]:
    """获取节点的非 absorbed 输入 tensor id 列表。"""
    return node.active_inputs()


def _collect_consumer_src_requirements(
    tensor_id: str,
    graph: Graph,
    capabilities: FormatCapabilities,
) -> list[frozenset[str]]:
    """收集 tensor 所有消费者对该 tensor 的 src 格式要求。"""
    tensor = graph.tensors[tensor_id]
    needs: list[frozenset[str]] = []

    for consumer_id in tensor.consumer_node_ids:
        consumer = graph.nodes.get(consumer_id)
        if consumer is None or consumer.compute_unit is None:
            continue
        cap = capabilities.get(consumer.compute_unit, consumer.npu_op)
        # 找到 tensor 在 consumer 的非 absorbed 输入中的位置
        non_absorbed = _get_non_absorbed_inputs(consumer, graph)
        try:
            pos = non_absorbed.index(tensor_id)
        except ValueError:
            continue
        needs.append(cap.get_src_formats(position=pos))

    return needs


def _pick_best_format(
    dst_options: frozenset[str],
    consumer_needs: list[frozenset[str]],
) -> str:
    """选择满足最多 consumer 的输出格式。"""
    if not consumer_needs:
        # 没有消费者（模型输出），选 nd 作为默认
        return "nd" if "nd" in dst_options else next(iter(dst_options))

    scores: dict[str, int] = {fmt: 0 for fmt in dst_options}
    for need in consumer_needs:
        for fmt in dst_options:
            if fmt in need:
                scores[fmt] += 1

    return max(scores, key=scores.get)


def run(graph: Graph, config: dict) -> Graph:
    """执行 format planner pass。

    config 键：
        format_capabilities: dict — 硬件格式能力配置（来自 hardware_config.yaml）。
    """
    raw_caps = config.get("format_capabilities", {})
    if not raw_caps:
        logger.info("format_planner: 无 format_capabilities 配置，跳过")
        return graph

    capabilities = parse_capabilities(raw_caps)
    planned_tensors = 0
    annotated_nodes = 0

    # 用拓扑序遍历
    topo_order = graph.topo_sort()

    for node_id in topo_order:
        node = graph.nodes[node_id]
        if node.compute_unit is None:
            continue

        cap = capabilities.get(node.compute_unit, node.npu_op)

        non_absorbed = _get_non_absorbed_inputs(node, graph)

        # ---- 1. 确定每个输入的所需格式 ----
        input_annotations = []
        for i, tid in enumerate(node.inputs):
            tensor = graph.tensors.get(tid)
            if tensor is None:
                continue

            # absorbed 输入保留现有标注
            if tid not in non_absorbed:
                existing_ann = _get_existing_input_annotation(node, i)
                if existing_ann:
                    input_annotations.append(existing_ann)
                else:
                    input_annotations.append({
                        "format": tensor.format,
                        "dtype": tensor.dtype,
                    })
                continue

            pos = non_absorbed.index(tid)
            required = cap.get_src_formats(position=pos)

            if tensor.format and tensor.format != "nd":
                # tensor 已有 format（非默认），检查是否兼容
                if tensor.format in required:
                    input_annotations.append({
                        "format": tensor.format,
                        "dtype": tensor.dtype,
                    })
                else:
                    # 不兼容 → format_annotation 记录所需格式，reformat_inserter 会处理
                    needed = next(iter(required))
                    input_annotations.append({
                        "format": needed,
                        "dtype": tensor.dtype,
                    })
            else:
                # tensor.format 未决定或为默认 nd
                if "nd" in required:
                    input_annotations.append({
                        "format": "nd",
                        "dtype": tensor.dtype,
                    })
                else:
                    input_annotations.append({
                        "format": next(iter(required)),
                        "dtype": tensor.dtype,
                    })

        # ---- 2. 确定输出 tensor 的格式 ----
        output_annotations = []
        for tid in node.outputs:
            tensor = graph.tensors.get(tid)
            if tensor is None:
                continue

            dst_options = cap.dst

            if len(dst_options) == 1:
                chosen = next(iter(dst_options))
            else:
                # lookahead: 看所有 consumer 需要什么
                consumer_needs = _collect_consumer_src_requirements(
                    tid, graph, capabilities,
                )
                chosen = _pick_best_format(dst_options, consumer_needs)

            # 检查用户 npu() override
            npu_hint = node.params.get("_npu", {}).get("output")
            if npu_hint and hasattr(npu_hint, "format") and npu_hint.format:
                if npu_hint.format != chosen:
                    logger.warning(
                        "节点 %s 输出格式: 最优=%s, 用户指定=%s, 按用户设置",
                        node.id, chosen, npu_hint.format,
                    )
                chosen = npu_hint.format

            old_format = tensor.format
            tensor.format = chosen
            planned_tensors += 1
            if old_format != chosen:
                # 收集消费者信息用于日志
                consumer_info = []
                for cid in tensor.consumer_node_ids:
                    cn = graph.nodes.get(cid)
                    if cn:
                        consumer_info.append(f"{cid} ({cn.npu_op or cn.op_type})")
                log_opt(
                    node, "format_planner", "格式规划",
                    f"{tid}: {old_format}→{chosen}。"
                    + (f"下游 {', '.join(consumer_info)} 的硬件端口要求 {chosen} 格式，"
                       f"在此处预转换可避免每次消费时的运行时格式转换开销"
                       if consumer_info
                       else f"模型输出 tensor，使用默认格式 {chosen}"),
                )

            output_annotations.append({
                "format": chosen,
                "dtype": tensor.dtype,
            })

        # ---- 3. 填充 format_annotation ----
        if node.format_annotation:
            # 合并：保留 dtype 信息，更新 format
            existing = node.format_annotation
            _merge_format_into_annotation(existing, input_annotations, output_annotations)
        else:
            node.format_annotation = {
                "inputs": input_annotations,
                "outputs": output_annotations,
            }

        annotated_nodes += 1

    logger.info(
        "format_planner 完成: 规划了 %d 个 tensor 的格式, 标注了 %d 个节点",
        planned_tensors, annotated_nodes,
    )
    return graph


def _get_existing_input_annotation(node, index: int) -> dict | None:
    """从现有 format_annotation 获取指定输入的标注。"""
    if not node.format_annotation:
        return None
    inputs = node.format_annotation.get("inputs", [])
    if index < len(inputs):
        return inputs[index]
    return None


def _merge_format_into_annotation(
    existing: dict,
    input_anns: list[dict],
    output_anns: list[dict],
) -> None:
    """将 planner 的 format 决策合并到现有 annotation 中。"""
    existing_inputs = existing.get("inputs", [])
    for i, ann in enumerate(input_anns):
        if i < len(existing_inputs):
            existing_inputs[i]["format"] = ann["format"]
        else:
            existing_inputs.append(ann)
    existing["inputs"] = existing_inputs if existing_inputs else input_anns

    existing_outputs = existing.get("outputs", [])
    for i, ann in enumerate(output_anns):
        if i < len(existing_outputs):
            existing_outputs[i]["format"] = ann["format"]
        else:
            existing_outputs.append(ann)
    existing["outputs"] = existing_outputs if existing_outputs else output_anns


def post_validate(graph: Graph) -> list[str]:
    """format_planner 后校验：有 producer 的中间 tensor 必须有 format。"""
    errors: list[str] = []
    for t in graph.tensors.values():
        if t.producer_node_id is not None and not t.format:
            errors.append(f"tensor {t.id} 经过 format_planner 后仍缺少 format")
    return errors
