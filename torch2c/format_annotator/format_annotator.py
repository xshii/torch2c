"""format_annotator — Pass⑤：Format/Dtype 标注。

dtype 和 format 由模型自身决定（graph_capture 捕获），用户可通过
target_dtype / target_format 覆盖。不再依赖静态 YAML 配置。

compute_dtype 支持分层规则，按优先级从高到低：
  1. node.params 已有值（op_decomposition 等 Pass 预设的）
  2. per-op 规则：compute_dtype_rules.by_op[npu_op]
  3. per-unit 规则：compute_dtype_rules.by_compute_unit[compute_unit]
  4. 全局默认：compute_dtype_rules.default 或 target_dtype
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

logger = get_logger(__name__)


def _resolve_compute_dtype(node, rules: dict, fallback: str) -> str:
    """按分层规则解析 node 的 compute_dtype。"""
    # 1. per-op
    by_op = rules.get("by_op", {})
    if node.npu_op and node.npu_op in by_op:
        return by_op[node.npu_op]
    # 2. per-unit
    by_unit = rules.get("by_compute_unit", {})
    if node.compute_unit and node.compute_unit in by_unit:
        return by_unit[node.compute_unit]
    # 3. global default
    return rules.get("default", fallback)


def run(graph: Graph, config: dict) -> Graph:
    """执行 Format/Dtype 标注 pass。

    config 可选键：
        target_dtype:   存储 dtype（如 "fp16"），None 则继承模型原始值。
        target_format:  存储 format（如 "nz"），None 则继承模型原始值。
        compute_dtype:  全局计算精度（如 "fp32"），None 则跟随 target_dtype。
            当 compute_dtype_rules 存在时被忽略。
        compute_dtype_rules:  分层计算精度规则，dict：
            default: str — 全局默认（如 "fp16"）
            by_compute_unit: dict[str, str] — 按计算单元（如 {"cube": "fp32"}）
            by_op: dict[str, str] — 按算子（如 {"vector_softmax_part1": "fp32"}）
    """
    target_dtype = config.get("target_dtype")
    target_format = config.get("target_format")

    # 构建 compute_dtype 分层规则
    rules = config.get("compute_dtype_rules")
    if rules is None:
        # 兼容旧的 compute_dtype 单值配置
        global_compute = config.get("compute_dtype") or target_dtype or "fp16"
        rules = {"default": global_compute}

    dtype_fallback = target_dtype or "fp16"

    annotated_nodes = 0
    updated_tensors = 0

    for node in graph.nodes.values():
        if node.npu_op is None:
            continue

        # per-node npu() 标注（优先级最高）：NpuSpec 对象
        npu_ann = node.params.get("_npu", {})
        npu_in = npu_ann.get("input")    # NpuSpec | None
        npu_out = npu_ann.get("output")  # NpuSpec | None

        # 为每个输入构建标注（仅 dtype，format 留给 format_planner）
        # 优先级：per-node NpuSpec > global target > tensor 原始值 > 默认
        input_annotations = []
        for tid in node.inputs:
            t = graph.get_tensor(tid)
            if t is not None:
                fmt = (npu_in.format if npu_in else None) or target_format or t.format or "nd"
                dt = (npu_in.dtype if npu_in else None) or target_dtype or t.dtype or "fp16"
                input_annotations.append({"format": fmt, "dtype": dt})

        # 为每个输出构建标注并更新 tensor dtype + format
        # format_planner（如果启用）会在后续覆盖 tensor.format
        output_annotations = []
        for tid in node.outputs:
            t = graph.get_tensor(tid)
            if t is not None:
                fmt = (npu_out.format if npu_out else None) or target_format or t.format or "nd"
                dt = (npu_out.dtype if npu_out else None) or target_dtype or t.dtype or "fp16"
                output_annotations.append({"format": fmt, "dtype": dt})
                t.format = fmt
                t.dtype = dt
                updated_tensors += 1

        node.format_annotation = {
            "inputs": input_annotations,
            "outputs": output_annotations,
        }

        # compute_dtype 写入 node.params（codegen 通过 param.compute_dtype 取值）
        if "compute_dtype" not in node.params:
            node.params["compute_dtype"] = _resolve_compute_dtype(
                node, rules, dtype_fallback,
            )

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
