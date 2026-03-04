"""integration — 编译管线：串联全部 9 个 Pass，实现从 PyTorch 模型到 C 工程的完整编译。"""

from __future__ import annotations

import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from npu_compiler.common import DiagnosticCollector, Graph, get_logger, load_config
from npu_compiler.codegen import c_emitter, golden_exporter, weight_exporter
from npu_compiler.codegen.c_project import (
    cmake_emitter, main_emitter, mock_emitter, utils_emitter,
)
from npu_compiler.format_annotator import format_annotator
from npu_compiler.graph_capture import graph_capture
from npu_compiler.memory_planner import memory_planner
from npu_compiler.op_absorption import op_absorption
from npu_compiler.op_decomposition import op_decomposition
from npu_compiler.op_mapping import op_mapping
from npu_compiler.scheduler import scheduler
from npu_compiler.validator import validator

logger = get_logger(__name__)

def _propagate_input_dtypes(graph: Graph) -> None:
    """将输入 tensor 的 dtype 设为其首个消费节点的标注 dtype。

    format_annotator 只标注输出 tensor，输入 tensor（模型输入、权重）
    保留了 graph_capture 的原始 dtype。此函数根据消费者的 format_annotation
    来推断输入 tensor 应使用的 dtype。
    """
    for t in graph.tensors.values():
        if t.producer_node_id is not None:
            continue  # 由 format_annotator 通过输出标注处理
        if not t.consumer_node_ids:
            continue
        consumer = graph.get_node(t.consumer_node_ids[0])
        if consumer is None:
            continue
        ann = getattr(consumer, "format_annotation", None)
        if ann is None:
            continue
        # 找到该 tensor 在 consumer.inputs 中的位置
        idx = None
        for i, tid in enumerate(consumer.inputs):
            if tid == t.id:
                idx = i
                break
        if idx is not None and idx < len(ann.get("inputs", [])):
            t.dtype = ann["inputs"][idx]["dtype"]


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
    model: nn.Module, dummy_input: torch.Tensor,
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
) -> None:
    """收集指定阶段的校验诊断。"""
    for msg in graph.validate_phase(phase):
        collector.warn(phase, msg)
        logger.warning("Pass %s 校验: %s", phase, msg)


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

    # Pass ① graph_capture
    logger.info("Pass ① graph_capture 开始")
    graph = graph_capture.capture(model, dummy_input, mask=mask)
    _run_post_validation(collector, "graph_capture", graph)
    logger.info("Pass ① 完成: %s", graph.summary())

    # Pass ② op_mapping
    logger.info("Pass ② op_mapping 开始")
    graph = op_mapping.run(graph, configs["mapping"])
    _run_post_validation(collector, "op_mapping", graph)
    logger.info("Pass ② 完成")

    # Pass ③ op_decomposition
    logger.info("Pass ③ op_decomposition 开始")
    graph = op_decomposition.run(graph, configs["decomposition"])
    _run_post_validation(collector, "op_decomposition", graph)
    logger.info("Pass ③ 完成: %s", graph.summary())

    # Pass ④ op_absorption
    logger.info("Pass ④ op_absorption 开始")
    graph = op_absorption.run(graph, configs["absorption"])
    logger.info("Pass ④ 完成: %s", graph.summary())

    # Pass ⑤ format_annotator
    logger.info("Pass ⑤ format_annotator 开始")
    graph = format_annotator.run(graph, configs["format"])
    _propagate_input_dtypes(graph)
    _run_post_validation(collector, "format_annotator", graph)
    logger.info("Pass ⑤ 完成")

    # Pass ⑥ validator
    logger.info("Pass ⑥ validator 开始")
    validator_cfg = _build_validator_config(configs["signatures"])
    graph = validator.run(graph, validator_cfg)
    logger.info("Pass ⑥ 完成")

    # Pass ⑦ memory_planner
    logger.info("Pass ⑦ memory_planner 开始")
    graph, dma_plans = memory_planner.run(graph, configs["hardware"])
    _run_post_validation(collector, "memory_planner", graph)
    logger.info("Pass ⑦ 完成")

    # Pass ⑧ scheduler
    logger.info("Pass ⑧ scheduler 开始")
    graph = scheduler.run(graph)
    logger.info("Pass ⑧ 完成")

    logger.info(collector.summary())

    # Pass ⑨ codegen
    logger.info("Pass ⑨ codegen 开始")
    plan = _build_codegen_plan(graph, dma_plans)

    c_emitter.run(plan, output_dir, config_dir=config_dir)
    mock_emitter.run(output_dir, config_dir=config_dir)
    main_emitter.run(plan, configs["hardware"], output_dir,
                     atol=atol, cosine_tol=cosine_tol)
    cmake_emitter.run(output_dir)
    utils_emitter.run(output_dir)

    weight_path = os.path.join(output_dir, "src", "model_weights.h")
    weight_offsets = {
        t["name"]: t.get("hbm_offset", 0) or 0
        for t in plan["tensors"].values()
        if t.get("is_weight") and t.get("name")
    }
    weight_exporter.export_weights(
        model.state_dict(), weight_path, offsets=weight_offsets,
    )

    golden_dir = os.path.join(output_dir, "golden")
    inputs, outputs = _run_golden(model, dummy_input, mask)
    golden_exporter.export_golden(inputs, outputs, golden_dir)

    logger.info("Pass ⑨ 完成")
    logger.info("=== 编译管线完成，输出目录: %s ===", output_dir)
    return output_dir
