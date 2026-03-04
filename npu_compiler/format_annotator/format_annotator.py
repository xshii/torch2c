"""format_annotator — Pass⑤：Format/Dtype 标注。"""

from __future__ import annotations

from npu_compiler.common import Graph, get_logger, load_config

logger = get_logger(__name__)

_CONFIG_REQUIRED_KEYS = ["op_format_requirements"]


def load_format_config(path: str) -> dict:
    """加载 format/dtype 标注配置。"""
    return load_config(path, required_keys=_CONFIG_REQUIRED_KEYS)


def _annotate_node(graph: Graph, node, req: dict) -> int:
    """为单个节点标注 format/dtype，返回更新的 tensor 数量。"""
    node.format_annotation = {
        "inputs": [{"format": r["format"], "dtype": r["dtype"]} for r in req["inputs"]],
        "outputs": [{"format": r["format"], "dtype": r["dtype"]} for r in req["outputs"]],
        "compute_dtype": req["compute_dtype"],
        "supports_format_convert": req.get("supports_format_convert", False),
        "supports_dtype_cast": req.get("supports_dtype_cast", False),
    }

    updated = 0
    for i, tid in enumerate(node.outputs):
        if i < len(req["outputs"]):
            tensor = graph.get_tensor(tid)
            if tensor is not None:
                tensor.format = req["outputs"][i]["format"]
                tensor.dtype = req["outputs"][i]["dtype"]
                updated += 1
    return updated


def run(graph: Graph, config: dict) -> Graph:
    """执行 Format/Dtype 标注 pass。"""
    requirements = config["op_format_requirements"]
    annotated_nodes = 0
    updated_tensors = 0

    for node in graph.nodes.values():
        op = node.npu_op
        if op is None:
            continue
        if op not in requirements:
            continue
        req = requirements[op]
        updated_tensors += _annotate_node(graph, node, req)
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
