"""_codegen_runner — Pass ⑨ codegen 执行辅助函数。"""

from __future__ import annotations

import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from torch2c.codegen import c_emitter, golden_exporter, weight_exporter
from torch2c.codegen.c_project import (
    cmake_emitter,
    main_emitter,
    mock_emitter,
    utils_emitter,
)
from torch2c.common import Graph, dtype_numpy, get_logger

logger = get_logger(__name__)


def _build_codegen_plan(graph: Graph, dma_plans: list) -> dict:
    """将 Graph + DMA 计划序列化为 codegen plan dict。"""
    plan = graph.to_dict()
    plan["dma_plans"] = [asdict(dp) for dp in dma_plans]
    return plan


def _infer_golden_dtype(graph: Graph) -> type:
    """从 graph 的 model_input tensor 推断 golden 数据精度。"""
    for t in graph.tensors.values():
        if t.is_model_input and t.dtype:
            return dtype_numpy(t.dtype)
    return np.float16


def _run_golden(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    numpy_dtype: type = np.float16,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """前向推理获取 golden 输入/输出。"""
    model.eval()
    with torch.no_grad():
        out = model(dummy_input, mask) if mask is not None else model(dummy_input)

    inputs = [dummy_input.cpu().float().numpy().astype(numpy_dtype)]
    if mask is not None:
        inputs.append(mask.cpu().float().numpy().astype(numpy_dtype))

    if isinstance(out, torch.Tensor):
        outputs = [out.cpu().float().numpy().astype(numpy_dtype)]
    else:
        outputs = [o.cpu().float().numpy().astype(numpy_dtype) for o in out]
    return inputs, outputs


def _emit_c_project(
    plan: dict,
    configs: dict,
    config_dir: str,
    output_dir: str,
    golden_dtype: type,
    atol: float,
    cosine_tol: float,
    static_golden: bool,
) -> None:
    """生成 C 源文件、CMake 和辅助文件。"""
    c_emitter.run(plan, output_dir, config_dir=config_dir)
    mock_emitter.run(output_dir, config_dir=config_dir)
    elem_size = 2 if golden_dtype == np.float16 else 4
    main_emitter.run(
        plan, configs["hardware"], output_dir,
        atol=atol, cosine_tol=cosine_tol,
        static_mode=static_golden, elem_size=elem_size,
    )
    cmake_emitter.run(output_dir)
    utils_emitter.run(output_dir)


def _compute_offsets(plan: dict, key: str) -> list[int]:
    """从 plan tensors 中按 hbm_offset 排序提取偏移列表。"""
    filtered = [t for t in plan["tensors"].values() if t.get(key)]
    sorted_tensors = sorted(filtered, key=lambda t: t.get("hbm_offset", 0) or 0)
    return [t.get("hbm_offset", 0) or 0 for t in sorted_tensors]


def _export_weights_and_golden(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None,
    plan: dict,
    graph: Graph,
    output_dir: str,
    static_golden: bool,
) -> None:
    """导出权重和 golden 数据（文件模式 + 可选静态模式）。"""
    golden_dtype = _infer_golden_dtype(graph)
    dtype_str = {np.float16: "fp16", np.float32: "fp32"}.get(golden_dtype, "fp16")

    # 权重导出
    weight_path = os.path.join(output_dir, "src", "model_weights.h")
    weight_offsets = {
        t["name"]: t.get("hbm_offset", 0) or 0
        for t in plan["tensors"].values()
        if t.get("is_weight") and t.get("name")
    }
    weight_exporter.export_weights(
        model.state_dict(), weight_path, dtype=dtype_str, offsets=weight_offsets
    )

    # golden 前向推理
    inputs, outputs = _run_golden(model, dummy_input, mask, numpy_dtype=golden_dtype)

    # golden 数据：文件模式
    golden_dir = os.path.join(output_dir, "golden")
    golden_exporter.export_golden(inputs, outputs, golden_dir, dtype=dtype_str)

    # 可选静态模式
    if static_golden:
        input_offsets = _compute_offsets(plan, "is_model_input")
        output_offsets = _compute_offsets(plan, "is_model_output")
        golden_exporter.export_golden_static(
            inputs, outputs, input_offsets, output_offsets,
            os.path.join(output_dir, "src", "model_golden.h"),
        )


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
    *,
    static_golden: bool = False,
) -> None:
    """Pass ⑨：C 代码生成 + 权重导出 + golden 数据。"""
    logger.info("Pass ⑨ codegen 开始")
    plan = _build_codegen_plan(graph, dma_plans)
    golden_dtype = _infer_golden_dtype(graph)

    _emit_c_project(
        plan, configs, config_dir, output_dir,
        golden_dtype, atol, cosine_tol, static_golden,
    )
    _export_weights_and_golden(
        model, dummy_input, mask, plan, graph, output_dir, static_golden,
    )
    logger.info("Pass ⑨ 完成")
