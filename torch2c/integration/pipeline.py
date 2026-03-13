"""integration — 编译管线：串联全部 9 个 Pass，实现从 PyTorch 模型到 C 工程的完整编译。"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch2c.viz import emit_graph_html, emit_lifetime_html
from torch2c.common import (
    CompilerError, DiagnosticCollector, Graph, graph_diff,
    get_logger, get_model_config, load_config, load_debug_config,
)
from torch2c.common.pass_config import OptionalPass, PassConfig
from torch2c.integration._codegen_runner import _dump_intermediates, _run_codegen
from torch2c.format_annotator import format_annotator
from torch2c.fusion import fusion_planner
from torch2c.global_tiler import global_tiler
from torch2c.graph_capture import graph_capture
from torch2c.reformat_inserter import reformat_inserter
from torch2c.memory_planner import memory_planner
from torch2c.op_absorption import op_absorption
from torch2c.op_decomposition import op_decomposition
from torch2c.op_mapping import op_mapping
from torch2c.roofline import roofline_analyzer
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
    # toggle: 若非 None，则可通过 PassConfig 关闭该 Pass
    toggle: OptionalPass | None = None


def _emit_schedule_viz(
    graph: Graph, output_dir: str, cube_size: int,
    hw_config: dict | None = None, dma_plans: list | None = None,
) -> None:
    """memory_planner 后生成 Pipeline Schedule 甘特图（HTML），含 DMA 搬运。"""
    emit_graph_html(graph, output_dir, cube_size, hw_config, dma_plans)


# Phase 2: Graph Optimization (②-④)
_OPTIMIZATION_PASSES: list[_PassDesc] = [
    _PassDesc("op_mapping", "②", op_mapping.run, "mapping", op_mapping.post_validate),
    _PassDesc(
        "op_decomposition",
        "③",
        op_decomposition.run,
        "decomposition",
        op_decomposition.post_validate,
    ),
    _PassDesc("op_absorption", "④", op_absorption.run, "absorption",
             op_absorption.post_validate, toggle=OptionalPass.ABSORPTION),
]

# Phase 3: Backend Annotation (⑤-⑤b)
_ANNOTATION_PASSES: list[_PassDesc] = [
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
        toggle=OptionalPass.STORAGE_ASSIGNER,
    ),
]


# ---- 工具函数 ----


def _load_pass_config(config_dir: str) -> PassConfig:
    """加载可选 Pass 开关，缺失时默认全部启用。"""
    opt_path = os.path.join(config_dir, "optimization_config.yaml")
    if not os.path.exists(opt_path):
        return PassConfig()
    opt = load_config(opt_path)
    return PassConfig.from_dict(opt.get("passes", {}))


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

    # 加载维测配置（debug.yaml 不存在时全部关闭）
    debug = load_debug_config(config_dir)

    return {
        "mapping": load_config(os.path.join(config_dir, "direct_mappings.yaml")),
        "decomposition": load_config(os.path.join(config_dir, "decompositions.yaml")),
        "absorption": load_config(os.path.join(config_dir, "absorptions.yaml")),
        "format": format_config,
        "reformat": {
            "compute_unit_formats": hardware.get("unit_formats", {}),
        },
        "storage": {
            "enable_local_storage": True,
            **hardware.get("local_bypass", {}),
        },
        "hardware": hardware,
        "signatures": load_config(os.path.join(config_dir, "c_api_signatures.yaml")),
        "debug": debug,
        "pass_config": _load_pass_config(config_dir),
    }


def _build_validator_config(signatures_config: dict) -> dict:
    """从 c_api_signatures 所有 section 的键集构建 validator 配置。"""
    ops: list[str] = []
    for section in ("compute_ops", "dma_ops", "idma_ops"):
        ops.extend(signatures_config.get(section, {}).keys())
    return {"supported_ops": ops}


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


def _dump_pass_snapshot(
    graph: Graph,
    before: dict,
    output_dir: str,
    number: str,
    name: str,
) -> None:
    """保存 Pass 快照和 diff 到 debug 目录。"""
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    after = graph.to_dict()
    prefix = f"pass_{number}_{name}"
    with open(os.path.join(debug_dir, f"{prefix}.json"), "w") as f:
        json.dump(after, f, indent=2, ensure_ascii=False)
    diff = graph_diff(before, after)
    with open(os.path.join(debug_dir, f"{prefix}_diff.json"), "w") as f:
        json.dump(diff, f, indent=2, ensure_ascii=False, default=str)
    logger.debug("debug_dump: %s written", prefix)


def _run_pass_list(
    graph: Graph,
    passes: list[_PassDesc],
    configs: dict,
    collector: DiagnosticCollector,
    output_dir: str,
    cube_size: int,
    debug_dump: bool = False,
) -> Graph:
    """声明式 Pass 循环，支持 OptionalPass 开关。"""
    pc: PassConfig = configs.get("pass_config", PassConfig())
    for p in passes:
        if p.toggle and not pc.is_enabled(p.toggle):
            logger.info("Pass %s %s 已禁用，跳过", p.number, p.name)
            continue
        logger.info("Pass %s %s 开始", p.number, p.name)
        if p.config_key is None:
            raise CompilerError(f"Pass {p.name} 缺少 config_key")
        before = graph.to_dict() if debug_dump else None
        graph = p.run_fn(graph, configs[p.config_key])
        if debug_dump:
            _dump_pass_snapshot(graph, before, output_dir, p.number, p.name)
        if p.validate_fn:
            _run_post_validation(collector, p.name, graph, p.validate_fn)
        if p.viz_hook:
            p.viz_hook(graph, output_dir, cube_size)
        logger.info("Pass %s 完成", p.number)
    return graph


def _run_phase_checkpoint(
    collector: DiagnosticCollector,
    phase_name: str,
    graph: Graph,
    debug_dump: bool,
) -> None:
    """Phase 边界校验检查点（仅 debug_dump 模式下执行）。"""
    if not debug_dump:
        return
    errors = graph.validate()
    if errors:
        for msg in errors:
            collector.warn(phase_name, msg)
            logger.warning("Phase %s 边界校验: %s", phase_name, msg)
    else:
        logger.debug("Phase %s 边界校验通过", phase_name)


def _run_late_passes(
    graph: Graph,
    configs: dict,
    collector: DiagnosticCollector,
    output_dir: str,
    cube_size: int,
    debug_dump: bool = False,
    model_name: str | None = None,
) -> Graph:
    """Pass ⑥-⑧: validator → roofline → fusion → scheduler → global_tiler → memory_planner。"""
    # Pass ⑥ validator
    logger.info("Pass ⑥ validator 开始")
    validator_cfg = _build_validator_config(configs["signatures"])
    before = graph.to_dict() if debug_dump else None
    try:
        graph = validator.run(graph, validator_cfg)
    except CompilerError as exc:
        collector.error("validator", str(exc))
        raise
    if debug_dump:
        _dump_pass_snapshot(graph, before, output_dir, "⑥", "validator")
    logger.info("Pass ⑥ 完成")

    pc: PassConfig = configs.get("pass_config", PassConfig())

    # Pass ⑥b roofline_analyzer（标注计算强度）
    if pc.is_enabled(OptionalPass.ROOFLINE_ANALYZER):
        logger.info("Pass ⑥b roofline_analyzer 开始")
        graph = roofline_analyzer.run(graph, configs)
        logger.info("Pass ⑥b 完成")
    else:
        logger.info("Pass ⑥b roofline_analyzer 已禁用，跳过")

    # Pass ⑥c fusion_planner（识别可融合算子组）
    if pc.is_enabled(OptionalPass.FUSION_PLANNER):
        logger.info("Pass ⑥c fusion_planner 开始")
        graph = fusion_planner.run(graph, configs)
        logger.info("Pass ⑥c 完成")
    else:
        logger.info("Pass ⑥c fusion_planner 已禁用，跳过")

    # Pass ⑦ scheduler（先确定执行顺序，再分配内存）
    logger.info("Pass ⑦ scheduler 开始")
    before = graph.to_dict() if debug_dump else None
    graph = scheduler.run(graph)
    if debug_dump:
        _dump_pass_snapshot(graph, before, output_dir, "⑦", "scheduler")
    _run_post_validation(collector, "scheduler", graph, scheduler.post_validate)
    logger.info("Pass ⑦ 完成")

    # Pass ⑦b global_tiler（全局最优 tiling 决策）
    if pc.is_enabled(OptionalPass.GLOBAL_TILER):
        logger.info("Pass ⑦b global_tiler 开始")
        graph = global_tiler.run(graph, configs)
        tile_override = {}
        for nid, node in graph.nodes.items():
            tc = node.params.get("_tile_config")
            if tc:
                tile_override[nid] = tc
        if tile_override:
            configs["hardware"].setdefault("tile_override", {}).update(tile_override)
        logger.info("Pass ⑦b 完成")
    else:
        logger.info("Pass ⑦b global_tiler 已禁用，跳过")

    # Pass ⑧ memory_planner（基于 scheduler 确定的执行顺序分配内存）
    logger.info("Pass ⑧ memory_planner 开始")
    before = graph.to_dict() if debug_dump else None
    graph = memory_planner.run(graph, configs["hardware"])
    if debug_dump:
        _dump_pass_snapshot(graph, before, output_dir, "⑧", "memory_planner")
    _run_post_validation(collector, "memory_planner", graph, memory_planner.post_validate)
    _emit_schedule_viz(graph, output_dir, cube_size, configs.get("hardware"), graph.dma_plans)
    emit_lifetime_html(graph, output_dir, cube_size, configs.get("hardware"),
                       dma_plans=graph.dma_plans, title=model_name)
    logger.info("Pass ⑧ 完成")

    return graph


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
    output_dir: str | None = None,
    mask: torch.Tensor | None = None,
    *,
    target_dtype: str | None = None,
    target_format: str | None = None,
    compute_dtype: str | None = None,
    atol: float = 1e-2,
    cosine_tol: float = 0.999,
    static_golden: bool = False,
    debug_dump: bool = False,
    tile_override: dict | None = None,
    pass_toggles: dict[str, bool] | None = None,
) -> str:
    """完整编译流水线：9 Pass 从 PyTorch 模型到 C 工程。返回输出目录路径。

    output_dir 默认为 output/<ModelClassName>。

    tile_override: 手动 tiling 参数，格式为 {node_id: {"tile_size": int, "num_buffers": int}}。
        省略的字段使用自动计算值。设为 None 或 {} 使用全自动 tiling。

    pass_toggles: 可选 Pass 开关覆盖，优先级高于 optimization_config.yaml。
        例如 {"absorption": False, "global_tiler": False}。
        也可传入 PassConfig 实例（会自动适配）。
    """
    if output_dir is None:
        model_name = type(model).__name__
        output_dir = os.path.join("output", model_name)
    logger.info("=== 编译管线开始 ===")

    configs = _resolve_compile_configs(
        model, config_dir, target_dtype, target_format, compute_dtype,
    )
    if tile_override:
        configs["hardware"]["tile_override"] = tile_override
    # pass_toggles 覆盖 config 文件中的开关
    if pass_toggles:
        base_pc: PassConfig = configs["pass_config"]
        from dataclasses import asdict
        merged = {**asdict(base_pc), **pass_toggles}
        configs["pass_config"] = PassConfig.from_dict(merged)
    collector = DiagnosticCollector()

    # Pass ① graph_capture（特殊：不是 run(graph, config) 模式）
    logger.info("Pass ① graph_capture 开始")
    graph = graph_capture.capture(model, dummy_input, mask=mask)
    _run_post_validation(collector, "graph_capture", graph, graph_capture.post_validate)
    logger.info("Pass ① 完成: %s", graph.summary())

    cube_size = configs["hardware"]["fractal"]["cube_size"]
    model_name = type(model).__name__

    # Phase 2: Graph Optimization (②-④)
    graph = _run_pass_list(graph, _OPTIMIZATION_PASSES, configs, collector,
                           output_dir, cube_size, debug_dump)
    _run_phase_checkpoint(collector, "optimization", graph, debug_dump)

    # Phase 3: Backend Annotation (⑤-⑤b)
    graph = _run_pass_list(graph, _ANNOTATION_PASSES, configs, collector,
                           output_dir, cube_size, debug_dump)
    _run_phase_checkpoint(collector, "annotation", graph, debug_dump)

    # Phase 4-5: Validation + Backend (⑥-⑧)
    graph = _run_late_passes(graph, configs, collector, output_dir, cube_size,
                             debug_dump, model_name=model_name)

    logger.info(collector.summary())
    if collector.has_errors():
        raise CompilerError(f"编译中止：{collector.summary()}")

    # Pass ⑨ codegen
    _run_codegen(
        model, dummy_input, mask, graph, graph.dma_plans, configs, config_dir, output_dir,
        atol, cosine_tol, static_golden=static_golden,
    )

    if debug_dump:
        _dump_intermediates(model, dummy_input, mask, graph, output_dir)

    logger.info("=== 编译管线完成，输出目录: %s ===", output_dir)
    return output_dir


def compile_graph_only(
    model: nn.Module,
    dummy_input: torch.Tensor,
    config_dir: str,
    output_dir: str,
    mask: torch.Tensor | None = None,
    *,
    target_dtype: str | None = None,
    target_format: str | None = None,
    pass_toggles: dict[str, bool] | None = None,
) -> Graph:
    """仅运行 Pass ①-⑧（不含 codegen），返回编排后的 Graph。

    用于 benchmark / 策略对比，跳过 C 代码生成和 golden 验证。
    """
    configs = _resolve_compile_configs(model, config_dir, target_dtype, target_format, None)
    if pass_toggles:
        from dataclasses import asdict
        base_pc: PassConfig = configs["pass_config"]
        configs["pass_config"] = PassConfig.from_dict({**asdict(base_pc), **pass_toggles})
    collector = DiagnosticCollector()

    graph = graph_capture.capture(model, dummy_input, mask=mask)
    cube_size = configs["hardware"]["fractal"]["cube_size"]

    graph = _run_pass_list(graph, _OPTIMIZATION_PASSES, configs, collector,
                           output_dir, cube_size)
    graph = _run_pass_list(graph, _ANNOTATION_PASSES, configs, collector,
                           output_dir, cube_size)
    graph = _run_late_passes(graph, configs, collector, output_dir, cube_size,
                             model_name=type(model).__name__)
    return graph


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
