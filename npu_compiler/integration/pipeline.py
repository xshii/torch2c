"""integration — 编译管线：串联全部 9 个 Pass，实现从 PyTorch 模型到 C 工程的完整编译。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from npu_compiler.codegen import c_emitter, golden_exporter, weight_exporter
from npu_compiler.codegen.c_project import (
    cmake_emitter,
    main_emitter,
    mock_emitter,
    utils_emitter,
)
from npu_compiler.common import DiagnosticCollector, Graph, get_logger, load_config
from npu_compiler.format_annotator import format_annotator
from npu_compiler.graph_capture import graph_capture
from npu_compiler.memory_planner import memory_planner
from npu_compiler.op_absorption import op_absorption
from npu_compiler.op_decomposition import op_decomposition
from npu_compiler.op_mapping import op_mapping
from npu_compiler.scheduler import scheduler
from npu_compiler.validator import validator

logger = get_logger(__name__)


# ---- 声明式 Pass 描述 ----


@dataclass(frozen=True)
class _PassDesc:
    """中间 Pass 的声明式描述。"""

    name: str
    number: str
    run_fn: Callable
    config_key: str | None
    validate_fn: Callable | None = None
    post_hook: Callable | None = None


def _propagate_input_dtypes(graph: Graph) -> None:
    """将输入 tensor 的 dtype 设为其首个消费节点的标注 dtype。"""
    for t in graph.tensors.values():
        if t.producer_node_id is not None:
            continue
        if not t.consumer_node_ids:
            continue
        consumer = graph.get_node(t.consumer_node_ids[0])
        if consumer is None:
            continue
        ann = getattr(consumer, "format_annotation", None)
        if ann is None:
            continue
        idx = None
        for i, tid in enumerate(consumer.inputs):
            if tid == t.id:
                idx = i
                break
        if idx is not None and idx < len(ann.get("inputs", [])):
            t.dtype = ann["inputs"][idx]["dtype"]


_MIDDLE_PASSES: list[_PassDesc] = [
    _PassDesc("op_mapping", "②", op_mapping.run, "mapping", op_mapping.post_validate),
    _PassDesc(
        "op_decomposition",
        "③",
        op_decomposition.run,
        "decomposition",
        op_decomposition.post_validate,
    ),
    _PassDesc("op_absorption", "④", op_absorption.run, "absorption", op_absorption.post_validate),
    _PassDesc(
        "format_annotator",
        "⑤",
        format_annotator.run,
        "format",
        format_annotator.post_validate,
        _propagate_input_dtypes,
    ),
]


# ---- 工具函数 ----


def _load_configs(config_dir: str) -> dict:
    """加载全部配置文件。"""
    return {
        "mapping": load_config(os.path.join(config_dir, "direct_mappings.yaml")),
        "decomposition": load_config(os.path.join(config_dir, "decompositions.yaml")),
        "absorption": load_config(os.path.join(config_dir, "absorptions.yaml")),
        "format": load_config(os.path.join(config_dir, "type_format_config.yaml")),
        "hardware": load_config(os.path.join(config_dir, "hardware_config.yaml")),
        "signatures": load_config(os.path.join(config_dir, "c_api_signatures.yaml")),
    }


def _build_validator_config(signatures_config: dict) -> dict:
    """从 c_api_signatures 的 compute_ops 键集构建 validator 配置。"""
    return {"supported_ops": list(signatures_config.get("compute_ops", {}).keys())}


def _build_codegen_plan(graph: Graph, dma_plans: list) -> dict:
    """将 Graph + DMA 计划序列化为 codegen plan dict。"""
    return {
        "nodes": {nid: asdict(n) for nid, n in graph.nodes.items()},
        "tensors": {tid: asdict(t) for tid, t in graph.tensors.items()},
        "dma_plans": [asdict(dp) for dp in dma_plans],
        "execution_order": list(graph.execution_order),
    }


def _run_golden(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """前向推理获取 golden 输入/输出。"""
    model.eval()
    with torch.no_grad():
        out = model(dummy_input, mask) if mask is not None else model(dummy_input)

    inputs = [dummy_input.cpu().float().numpy().astype(np.float16)]
    if mask is not None:
        inputs.append(mask.cpu().float().numpy().astype(np.float16))

    if isinstance(out, torch.Tensor):
        outputs = [out.cpu().float().numpy().astype(np.float16)]
    else:
        outputs = [o.cpu().float().numpy().astype(np.float16) for o in out]
    return inputs, outputs


def _run_post_validation(
    collector: DiagnosticCollector,
    phase: str,
    graph: Graph,
    validate_fn=None,
) -> None:
    """收集阶段校验诊断。优先使用 Pass 模块的 post_validate。"""
    errors = graph.validate()
    if validate_fn is not None:
        errors.extend(validate_fn(graph))
    for msg in errors:
        collector.warn(phase, msg)
        logger.warning("Pass %s 校验: %s", phase, msg)


# ---- 主入口 ----


def compile(
    model: nn.Module,
    dummy_input: torch.Tensor,
    config_dir: str,
    output_dir: str = "output",
    mask: torch.Tensor | None = None,
    *,
    atol: float = 1e-2,
    cosine_tol: float = 0.999,
) -> str:
    """完整编译流水线。

    Args:
        model: PyTorch 模型。
        dummy_input: 样例输入张量。
        config_dir: 配置文件目录路径。
        output_dir: C 工程输出目录路径。
        mask: 可选 attention mask 张量。

    Returns:
        生成的 C 工程输出目录路径。
    """
    logger.info("=== 编译管线开始 ===")
    configs = _load_configs(config_dir)
    collector = DiagnosticCollector()

    # Pass ① graph_capture（特殊：不是 run(graph, config) 模式）
    logger.info("Pass ① graph_capture 开始")
    graph = graph_capture.capture(model, dummy_input, mask=mask)
    _run_post_validation(collector, "graph_capture", graph, graph_capture.post_validate)
    logger.info("Pass ① 完成: %s", graph.summary())

    # Pass ②-⑤ 声明式循环
    for p in _MIDDLE_PASSES:
        logger.info("Pass %s %s 开始", p.number, p.name)
        graph = p.run_fn(graph, configs[p.config_key])
        if p.post_hook:
            p.post_hook(graph)
        if p.validate_fn:
            _run_post_validation(collector, p.name, graph, p.validate_fn)
        logger.info("Pass %s 完成", p.number)

    # Pass ⑥ validator（特殊：config 从 signatures 派生）
    logger.info("Pass ⑥ validator 开始")
    validator_cfg = _build_validator_config(configs["signatures"])
    try:
        graph = validator.run(graph, validator_cfg)
    except Exception as exc:
        collector.error("validator", str(exc))
        raise
    logger.info("Pass ⑥ 完成")

    # Pass ⑦ memory_planner（特殊：返回 tuple）
    logger.info("Pass ⑦ memory_planner 开始")
    graph, dma_plans = memory_planner.run(graph, configs["hardware"])
    _run_post_validation(collector, "memory_planner", graph, memory_planner.post_validate)
    logger.info("Pass ⑦ 完成")

    # Pass ⑧ scheduler（特殊：无 config）
    logger.info("Pass ⑧ scheduler 开始")
    graph = scheduler.run(graph)
    _run_post_validation(collector, "scheduler", graph, scheduler.post_validate)
    logger.info("Pass ⑧ 完成")

    logger.info(collector.summary())

    # Pass ⑨ codegen
    _run_codegen(
        model, dummy_input, mask, graph, dma_plans, configs, config_dir, output_dir, atol, cosine_tol
    )

    logger.info("=== 编译管线完成，输出目录: %s ===", output_dir)
    return output_dir


def _run_codegen(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None,
    graph: Graph,
    dma_plans: list,
    configs: dict,
    config_dir: str,
    output_dir: str,
    atol: float,
    cosine_tol: float,
) -> None:
    """Pass ⑨：C 代码生成 + 权重导出 + golden 数据。"""
    logger.info("Pass ⑨ codegen 开始")
    plan = _build_codegen_plan(graph, dma_plans)

    c_emitter.run(plan, output_dir, config_dir=config_dir)
    mock_emitter.run(output_dir, config_dir=config_dir)
    main_emitter.run(plan, configs["hardware"], output_dir, atol=atol, cosine_tol=cosine_tol)
    cmake_emitter.run(output_dir)
    utils_emitter.run(output_dir)

    weight_path = os.path.join(output_dir, "src", "model_weights.h")
    weight_offsets = {
        t["name"]: t.get("hbm_offset", 0) or 0
        for t in plan["tensors"].values()
        if t.get("is_weight") and t.get("name")
    }
    weight_exporter.export_weights(model.state_dict(), weight_path, offsets=weight_offsets)

    golden_dir = os.path.join(output_dir, "golden")
    inputs, outputs = _run_golden(model, dummy_input, mask)
    golden_exporter.export_golden(inputs, outputs, golden_dir)
    logger.info("Pass ⑨ 完成")
