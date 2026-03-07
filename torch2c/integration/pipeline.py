"""integration — 编译管线：串联全部 9 个 Pass，实现从 PyTorch 模型到 C 工程的完整编译。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch2c.viz import emit_graph_ascii, emit_graph_dot, emit_lifetime_ascii
from torch2c.common import CompilerError, DiagnosticCollector, Graph, get_logger, get_model_config, load_config
from torch2c.integration._codegen_runner import _run_codegen
from torch2c.format_annotator import format_annotator
from torch2c.graph_capture import graph_capture
from torch2c.reformat_inserter import reformat_inserter
from torch2c.memory_planner import memory_planner
from torch2c.op_absorption import op_absorption
from torch2c.op_decomposition import op_decomposition
from torch2c.op_mapping import op_mapping
from torch2c.scheduler import scheduler
from torch2c.storage_assigner import storage_assigner
from torch2c.validator import validator

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
    # viz_hook(graph, output_dir, cube_size) — Pass 完成后生成可视化产物
    viz_hook: Callable | None = None


def _emit_graph_viz(graph: Graph, output_dir: str, cube_size: int) -> None:
    """storage_assigner 后生成算子依赖图（DOT + ASCII）。"""
    emit_graph_dot(graph, output_dir, cube_size)
    emit_graph_ascii(graph, output_dir, cube_size)


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
    ),
    _PassDesc(
        "reformat_inserter",
        "⑤a",
        reformat_inserter.run,
        "reformat",
        reformat_inserter.post_validate,
    ),
    _PassDesc(
        "storage_assigner",
        "⑤b",
        storage_assigner.run,
        "storage",
        storage_assigner.post_validate,
        _emit_graph_viz,
    ),
]


# ---- 工具函数 ----


def _load_configs(
    config_dir: str,
    target_dtype: str | None = None,
    target_format: str | None = None,
    compute_dtype: str | None = None,
) -> dict:
    """加载全部配置文件，函数参数可覆盖 model_config.yaml 默认值。"""
    hardware = load_config(os.path.join(config_dir, "hardware_config.yaml"))

    format_config: dict = {
        "target_dtype": target_dtype,
        "target_format": target_format,
    }
    if compute_dtype:
        format_config["compute_dtype"] = compute_dtype

    return {
        "mapping": load_config(os.path.join(config_dir, "direct_mappings.yaml")),
        "decomposition": load_config(os.path.join(config_dir, "decompositions.yaml")),
        "absorption": load_config(os.path.join(config_dir, "absorptions.yaml")),
        "format": format_config,
        "reformat": {},
        "storage": {
            "enable_local_storage": True,
            **hardware.get("local_bypass", {}),
        },
        "hardware": hardware,
        "signatures": load_config(os.path.join(config_dir, "c_api_signatures.yaml")),
    }


def _build_validator_config(signatures_config: dict) -> dict:
    """从 c_api_signatures 的 compute_ops 键集构建 validator 配置。"""
    return {"supported_ops": list(signatures_config.get("compute_ops", {}).keys())}


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
        collector.error(phase, msg)
        logger.warning("Pass %s 校验: %s", phase, msg)


def _run_middle_passes(
    graph: Graph,
    configs: dict,
    collector: DiagnosticCollector,
    output_dir: str,
    cube_size: int,
) -> Graph:
    """Pass ②-⑤：声明式中间 Pass 循环。"""
    for p in _MIDDLE_PASSES:
        logger.info("Pass %s %s 开始", p.number, p.name)
        if p.config_key is None:
            raise CompilerError(f"Pass {p.name} 缺少 config_key")
        graph = p.run_fn(graph, configs[p.config_key])
        if p.validate_fn:
            _run_post_validation(collector, p.name, graph, p.validate_fn)
        if p.viz_hook:
            p.viz_hook(graph, output_dir, cube_size)
        logger.info("Pass %s 完成", p.number)
    return graph


def _run_late_passes(
    graph: Graph,
    configs: dict,
    collector: DiagnosticCollector,
    output_dir: str,
    cube_size: int,
) -> tuple[Graph, list]:
    """Pass ⑥ validator → ⑦ memory_planner → ⑧ scheduler。"""
    # Pass ⑥ validator
    logger.info("Pass ⑥ validator 开始")
    validator_cfg = _build_validator_config(configs["signatures"])
    try:
        graph = validator.run(graph, validator_cfg)
    except CompilerError as exc:
        collector.error("validator", str(exc))
        raise
    logger.info("Pass ⑥ 完成")

    # Pass ⑦ memory_planner
    logger.info("Pass ⑦ memory_planner 开始")
    graph, dma_plans = memory_planner.run(graph, configs["hardware"])
    _run_post_validation(collector, "memory_planner", graph, memory_planner.post_validate)
    emit_lifetime_ascii(graph, output_dir, cube_size)
    logger.info("Pass ⑦ 完成")

    # Pass ⑧ scheduler
    logger.info("Pass ⑧ scheduler 开始")
    graph = scheduler.run(graph)
    _run_post_validation(collector, "scheduler", graph, scheduler.post_validate)
    logger.info("Pass ⑧ 完成")

    return graph, dma_plans


def _resolve_compile_configs(
    model: nn.Module,
    config_dir: str,
    target_dtype: str | None,
    target_format: str | None,
    compute_dtype: str | None,
) -> dict:
    """解析模型装饰器配置并加载编译配置。

    配置优先级：compile() 参数 > 模型 @torch2c_config > 默认值。
    """
    model_cfg = get_model_config(model)
    resolved_dtype = target_dtype or model_cfg.get("target_dtype")
    resolved_format = target_format or model_cfg.get("target_format")
    resolved_compute = compute_dtype or model_cfg.get("compute_dtype")
    compute_rules = model_cfg.get("compute_dtype_rules")

    configs = _load_configs(
        config_dir,
        target_dtype=resolved_dtype,
        target_format=resolved_format,
        compute_dtype=resolved_compute,
    )
    # 分层 compute_dtype_rules 从模型配置注入（如果有且未被参数覆盖）
    if compute_rules and not resolved_compute:
        configs["format"]["compute_dtype_rules"] = compute_rules
    return configs


# ---- 主入口 ----


def compile(
    model: nn.Module,
    dummy_input: torch.Tensor,
    config_dir: str,
    output_dir: str = "output",
    mask: torch.Tensor | None = None,
    *,
    target_dtype: str | None = None,
    target_format: str | None = None,
    compute_dtype: str | None = None,
    atol: float = 1e-2,
    cosine_tol: float = 0.999,
    static_golden: bool = False,
) -> str:
    """完整编译流水线：9 Pass 从 PyTorch 模型到 C 工程。返回输出目录路径。"""
    logger.info("=== 编译管线开始 ===")

    configs = _resolve_compile_configs(
        model, config_dir, target_dtype, target_format, compute_dtype,
    )
    collector = DiagnosticCollector()

    # Pass ① graph_capture（特殊：不是 run(graph, config) 模式）
    logger.info("Pass ① graph_capture 开始")
    graph = graph_capture.capture(model, dummy_input, mask=mask)
    _run_post_validation(collector, "graph_capture", graph, graph_capture.post_validate)
    logger.info("Pass ① 完成: %s", graph.summary())

    # Pass ②-⑤ 声明式循环
    cube_size = configs["hardware"]["fractal"]["cube_size"]
    graph = _run_middle_passes(graph, configs, collector, output_dir, cube_size)

    # Pass ⑥-⑧
    graph, dma_plans = _run_late_passes(graph, configs, collector, output_dir, cube_size)

    logger.info(collector.summary())
    if collector.has_errors():
        raise CompilerError(f"编译中止：{collector.summary()}")

    # Pass ⑨ codegen
    _run_codegen(
        model, dummy_input, mask, graph, dma_plans, configs, config_dir, output_dir,
        atol, cosine_tol, static_golden=static_golden,
    )

    logger.info("=== 编译管线完成，输出目录: %s ===", output_dir)
    return output_dir


# ---- 诊断入口 ----


def inspect(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Graph:
    """graph_capture + 标注诊断打印，返回 Graph。

    用于快速查看模型标注是否正确传播，不执行编译。
    """
    from torch2c.common import format_model_annotations

    inputs = [dummy_input] + ([mask] if mask is not None else [])
    print(format_model_annotations(model, inputs=inputs))

    graph = graph_capture.capture(model, dummy_input, mask=mask)
    print("\n" + graph.summary())
    print("\n" + graph.format_npu_annotations())
    return graph
