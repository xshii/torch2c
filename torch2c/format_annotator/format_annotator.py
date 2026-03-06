"""format_annotator — Pass⑤：Format/Dtype 标注。

dtype 和 format 由模型自身决定（graph_capture 捕获），用户可通过
target_dtype / target_format 覆盖。不再依赖静态 YAML 配置。
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

logger = get_logger(__name__)


def run(graph: Graph, config: dict) -> Graph:
    """执行 Format/Dtype 标注 pass。

    config 可选键：
        target_dtype:  用户指定的全局 dtype（如 "fp16"），None 表示继承模型原始值。
        target_format: 用户指定的全局 format（如 "nz"），None 表示继承模型原始值。
    """
    target_dtype = config.get("target_dtype")
    target_format = config.get("target_format")

    annotated_nodes = 0
    updated_tensors = 0

    for node in graph.nodes.values():
        if node.npu_op is None:
            continue

        # 为每个输入构建标注，dtype/format 来自 tensor 自身或用户覆盖
        input_annotations = []
        for tid in node.inputs:
            t = graph.get_tensor(tid)
            if t is not None:
                input_annotations.append({
                    "format": target_format or t.format or "nd",
                    "dtype": target_dtype or t.dtype or "fp16",
                })

        # 为每个输出构建标注并同步更新 tensor
        output_annotations = []
        for tid in node.outputs:
            t = graph.get_tensor(tid)
            if t is not None:
                fmt = target_format or t.format or "nd"
                dt = target_dtype or t.dtype or "fp16"
                output_annotations.append({"format": fmt, "dtype": dt})
                t.format = fmt
                t.dtype = dt
                updated_tensors += 1

        node.format_annotation = {
            "inputs": input_annotations,
            "outputs": output_annotations,
        }
        annotated_nodes += 1

    logger.info("Format标注完成。标注了%d个节点，%d个tensor", annotated_nodes, updated_tensors)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """format_annotator 后的校验：有 producer 的 tensor 必须有 dtype。"""
    errors: list[str] = []
    for t in graph.tensors.values():
        if t.producer_node_id is not None and not t.dtype:
            errors.append(f"有 producer 的 tensor {t.id} 缺少 dtype")
    return errors
