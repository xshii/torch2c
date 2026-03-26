"""integration — 编译管线：串联全部 9 个 Pass，实现从 PyTorch 模型到 C 工程的完整编译。"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from torch2c.viz import emit_graph_html, emit_lifetime_html
from torch2c.common import (
    CompilerError, DiagnosticCollector, Graph, graph_diff,
    get_logger, get_model_config, load_config, load_debug_config,
)
from torch2c.common.pass_config import OptionalPass, PassConfig
from torch2c.integration._codegen_runner import _dump_intermediates, _run_codegen
from torch2c.optpass.c_block_pad import block_pad
from torch2c.c_backend.format_annotator import format_annotator
from torch2c.optpass.c_format_planner import format_planner
from torch2c.optpass.cd_fusion import fusion_planner
from torch2c.optpass.d_global_tiler import global_tiler
from torch2c.a_capture.graph_capture import graph_capture
from torch2c.c_backend.reformat_inserter import reformat_inserter
from torch2c.d_emission.memory_planner import memory_planner
from torch2c.optpass.bc_mha_merge import mha_merge
from torch2c.optpass.bc_op_absorption import op_absorption
from torch2c.b_lowering.op_decomposition import op_decomposition
from torch2c.b_lowering.op_mapping import op_mapping
from torch2c.optpass.cd_block_fuser import block_fuser
from torch2c.optpass.cd_roofline import roofline_analyzer
from torch2c.d_emission.scheduler import scheduler
from torch2c.optpass.c_storage_assigner import storage_assigner
from torch2c.c_backend.validator import validator

logger = get_logger(__name__)


# ---- 声明式 Pass 描述 ----


@dataclass(frozen=True)
class _PassDesc:
    """中间 Pass 的声明式描述。

    config_key: configs dict 中的键名。None 表示传入完整 configs dict。
    config_builder: 若非 None，用 config_builder(configs) 的返回值代替 config_key 查找。
    post_hook: 若非 None，在 run() 成功后调用 post_hook(graph, configs)。
    collect_errors: 若为 True，捕获 CompilerError 并收集到 collector 后重新抛出。
    viz_hook: 若非 None，在 Pass 完成后调用 viz_hook(graph, output_dir, cube_size, configs)。
    validate_with_config: 若为 True，validate_fn(graph, config) 传入 pass config。
    """

    name: str
    number: str
    run_fn: Callable
    config_key: str | None
    validate_fn: Callable | None = None
    viz_hook: Callable | None = None
    toggle: OptionalPass | None = None
    config_builder: Callable | None = None
    post_hook: Callable | None = None
    collect_errors: bool = False
    validate_with_config: bool = False
    # T7: Analysis/Transform 分类与依赖声明
    kind: str = "transform"  # "analysis" | "transform"
    provides: frozenset = frozenset()  # 此 pass 产出的阶段标记
    requires: frozenset = frozenset()  # 此 pass 依赖的阶段标记


# ── 依赖阶段标记 ──
MAPPED = "mapped"         # op_mapping + op_decomposition 完成
ANNOTATED = "annotated"   # format_annotator 完成
SCHEDULED = "scheduled"   # scheduler 完成


def _validate_pass_dependencies(all_passes: list[list[_PassDesc]]) -> list[str]:
    """校验所有 pass 的依赖是否被前序 pass 满足。启动时调用。"""
    available: set[str] = set()
    errors: list[str] = []
    for pass_list in all_passes:
        for p in pass_list:
            missing = p.requires - available
            if missing:
                errors.append(
                    f"Pass {p.number} {p.name} requires {missing} "
                    f"but not provided by prior passes"
                )
            available |= p.provides
    return errors


# ---- Pass hooks ----


def _global_tiler_post_hook(graph: Graph, configs: dict) -> None:
    """将 global_tiler 决策注入 hardware config 的 tile_override。"""
    tile_override = {}
    for nid, node in graph.nodes.items():
        tc = node.params.get("_tile_config")
        if tc:
            tile_override[nid] = tc
    if tile_override:
        configs["hardware"].setdefault("tile_override", {}).update(tile_override)


def _memory_planner_viz(
    graph: Graph, output_dir: str, cube_size: int, configs: dict,
) -> None:
    """memory_planner 后生成甘特图 + lifetime 图。"""
    hw = configs.get("hardware")
    model_name = configs.get("_model_name")
    emit_graph_html(graph, output_dir, cube_size, hw, graph.dma_plans)
    emit_lifetime_html(graph, output_dir, cube_size, hw,
                       dma_plans=graph.dma_plans, title=model_name)


# ---- Pass 列表 ----


# Phase 2: Graph Optimization (②-④)
_OPTIMIZATION_PASSES: list[_PassDesc] = [
    _PassDesc("op_mapping", "②", op_mapping.run, "mapping", op_mapping.post_validate,
              provides=frozenset({MAPPED})),
    _PassDesc("op_decomposition", "③", op_decomposition.run, "decomposition",
              op_decomposition.post_validate,
              requires=frozenset({MAPPED}), provides=frozenset({MAPPED})),
    _PassDesc("op_absorption", "④", op_absorption.run, "absorption",
              op_absorption.post_validate, toggle=OptionalPass.ABSORPTION,
              requires=frozenset({MAPPED})),
    _PassDesc("mha_merge", "④b", mha_merge.run, "mha_merge",
              mha_merge.post_validate, toggle=OptionalPass.MHA_MERGE,
              requires=frozenset({MAPPED})),
]

# Phase 3: Backend Annotation (⑤-⑤d)
_ANNOTATION_PASSES: list[_PassDesc] = [
    _PassDesc(
        "format_annotator", "⑤", format_annotator.run,
        "format", format_annotator.post_validate,
        requires=frozenset({MAPPED}), provides=frozenset({ANNOTATED}),
    ),
    _PassDesc(
        "format_planner", "⑤a", format_planner.run,
        "format_planner", format_planner.post_validate,
        toggle=OptionalPass.FORMAT_PLANNER,
        kind="analysis", requires=frozenset({ANNOTATED}),
    ),
    _PassDesc(
        "reformat_inserter", "⑤b", reformat_inserter.run,
        "reformat", reformat_inserter.post_validate,
        requires=frozenset({ANNOTATED}),
    ),
    _PassDesc(
        "storage_assigner", "⑤c", storage_assigner.run,
        "storage", storage_assigner.post_validate,
        toggle=OptionalPass.STORAGE_ASSIGNER,
        kind="analysis", requires=frozenset({MAPPED}),
    ),
    _PassDesc(
        "block_pad", "⑤d", block_pad.run,
        "block_pad", block_pad.post_validate,
        toggle=OptionalPass.BLOCK_PAD,
        validate_with_config=True,
        requires=frozenset({ANNOTATED}),
    ),
]

# Phase 4-5: Validation + Scheduling + Memory (⑥-⑧)
_LATE_PASSES: list[_PassDesc] = [
    _PassDesc(
        "validator", "⑥", validator.run, None,
        config_builder=lambda c: _build_validator_config(c["signatures"]),
        collect_errors=True,
        kind="analysis", requires=frozenset({MAPPED}),
    ),
    _PassDesc(
        "roofline_analyzer", "⑥b", roofline_analyzer.run, None,
        toggle=OptionalPass.ROOFLINE_ANALYZER,
        kind="analysis", requires=frozenset({MAPPED}),
    ),
    _PassDesc(
        "block_fuser", "⑥c", block_fuser.run, None,
        block_fuser.post_validate,
        toggle=OptionalPass.BLOCK_FUSER,
        post_hook=_global_tiler_post_hook,
        kind="analysis",
    ),
    _PassDesc(
        "fusion_planner", "⑥c", fusion_planner.run, None,
        toggle=OptionalPass.FUSION_PLANNER,
        kind="analysis",
    ),
    _PassDesc(
        "scheduler", "⑦", scheduler.run, None,
        scheduler.post_validate,
        requires=frozenset({MAPPED}), provides=frozenset({SCHEDULED}),
    ),
    _PassDesc(
        "global_tiler", "⑦b", global_tiler.run, None,
        toggle=OptionalPass.GLOBAL_TILER,
        post_hook=_global_tiler_post_hook,
        kind="analysis",
    ),
    _PassDesc(
        "memory_planner", "⑧", memory_planner.run, "hardware",
        memory_planner.post_validate,
        viz_hook=_memory_planner_viz,
        requires=frozenset({SCHEDULED, ANNOTATED}),
    ),
]



# ---- Pass 拓扑导出（供 viz 使用）----

from torch2c.integration._pass_descriptions import (  # noqa: E402
    _PASS_DESC,
    get_pass_topology as _get_pass_topology,
)


def get_pass_topology() -> dict:
    """导出 pass 拓扑结构，供可视化自动生成。

    Returns:
        {"phases": [...], "passes": [...]}
        每个 pass: {"name", "number", "phase", "optional", "input", "output", "desc"}
    """
    return _get_pass_topology(
        _OPTIMIZATION_PASSES, _ANNOTATION_PASSES, _LATE_PASSES,
    )



# ---- 工具函数 ----


def _load_cost_model(config_dir: str) -> dict:
    """加载代价模型配置，缺失时返回空 dict（使用内置默认值）。"""
    path = os.path.join(config_dir, "cost_model_config.yaml")
    if not os.path.exists(path):
        return {}
    return load_config(path)


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
        "format_planner": {
            "format_capabilities": hardware.get("format_capabilities", {}),
        },
        "reformat": {},
        "mha_merge": {
            **hardware.get("mha_merge", {}),
            "cost_model": _load_cost_model(config_dir),
            "hardware": {
                "last_dim_align": (hardware.get("block_pad", {})
                    .get("alignment", {}).get("nd", {})
                    .get("fp16", [1, 16]))[1],
                "l1_size_bytes": hardware.get("memory", {}).get(
                    "l1", {}).get("total_size_bytes", 16 * 1024 * 1024),
                "dma_bytes_per_cycle": hardware.get("compute", {}).get(
                    "dma_bytes_per_cycle", 256),
            },
        },
        "block_pad": hardware.get("block_pad", {}),
        "storage": {
            "enable_local_storage": True,
            **hardware.get("local_bypass", {}),
            **hardware.get("pipe_bypass", {}),
        },
        "hardware": hardware,
        "cost_model": _load_cost_model(config_dir),
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


def _resolve_pass_config(p: _PassDesc, configs: dict) -> dict:
    """为 Pass 解析配置：config_builder > config_key > 完整 configs。"""
    if p.config_builder:
        return p.config_builder(configs)
    if p.config_key is not None:
        return configs[p.config_key]
    return configs


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
    timing: dict = configs.setdefault("_pass_timing", {})
    for p in passes:
        if p.toggle and not pc.is_enabled(p.toggle):
            logger.info("Pass %s %s 已禁用，跳过", p.number, p.name)
            timing[p.name] = {"enabled": False}
            continue
        logger.info("Pass %s %s 开始", p.number, p.name)
        pass_config = _resolve_pass_config(p, configs)
        before = graph.to_dict() if debug_dump else None
        t0 = time.perf_counter()
        try:
            graph = p.run_fn(graph, pass_config)
        except CompilerError as exc:
            if p.collect_errors:
                collector.error(p.name, str(exc))
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timing[p.name] = {"enabled": True, "duration_ms": round(elapsed_ms, 1)}
        logger.info("Pass %s 完成 (%.1f ms)", p.number, elapsed_ms)
        # 重编号：确保 node ID 与 execution_order 一致
        id_map = graph.renumber()
        if id_map:
            logger.debug("Pass %s 后重编号: %d 个节点", p.name, len(id_map))
        if p.post_hook:
            p.post_hook(graph, configs)
        if debug_dump:
            _dump_pass_snapshot(graph, before, output_dir, p.number, p.name)
        if p.validate_fn:
            vfn = p.validate_fn
            if p.validate_with_config:
                _run_post_validation(collector, p.name, graph,
                                     lambda g, _vfn=vfn, _cfg=pass_config: _vfn(g, _cfg))
            else:
                _run_post_validation(collector, p.name, graph, vfn)
        # T12: 阶段契约校验（validate_stage）
        stage_errors = graph.validate_stage(p.name)
        for msg in stage_errors:
            collector.warn(p.name, f"stage_contract: {msg}")
        if p.viz_hook:
            p.viz_hook(graph, output_dir, cube_size, configs)
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
    pass_toggles: dict[str, bool] | str | None = None,
) -> str:
    """完整编译流水线：9 Pass 从 PyTorch 模型到 C 工程。返回输出目录路径。

    output_dir 默认为 output/<ModelClassName>。

    tile_override: 手动 tiling 参数，格式为 {node_id: {"tile_size": int, "num_buffers": int}}。
        省略的字段使用自动计算值。设为 None 或 {} 使用全自动 tiling。

    pass_toggles: 可选 Pass 开关覆盖，优先级高于 optimization_config.yaml。
        例如 {"absorption": False, "global_tiler": False}。
        也可传入 "minimal" 关闭所有可选 pass（只跑必需 pass）。
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
    # 特殊值 "minimal" / "none" 关闭所有可选 pass
    if pass_toggles == "minimal" or pass_toggles == "none":
        configs["pass_config"] = PassConfig.all_disabled()
    elif pass_toggles:
        base_pc: PassConfig = configs["pass_config"]
        merged = {**asdict(base_pc), **pass_toggles}
        configs["pass_config"] = PassConfig.from_dict(merged)
    collector = DiagnosticCollector()

    # T7: 启动时校验 pass 依赖拓扑
    dep_errors = _validate_pass_dependencies(
        [_OPTIMIZATION_PASSES, _ANNOTATION_PASSES, _LATE_PASSES],
    )
    for msg in dep_errors:
        logger.error("Pass 依赖校验失败: %s", msg)

    # Pass ① graph_capture（特殊：不是 run(graph, config) 模式）
    logger.info("Pass ① graph_capture 开始")
    graph = graph_capture.capture(model, dummy_input, mask=mask)
    _run_post_validation(collector, "graph_capture", graph, graph_capture.post_validate)
    if debug_dump:
        _dump_pass_snapshot(graph, {"nodes": {}, "tensors": {}, "execution_order": []},
                            output_dir, "①", "graph_capture")
    logger.info("Pass ① 完成: %s", graph.summary())

    cube_size = configs["hardware"]["fractal"]["cube_size"]
    model_name = type(model).__name__
    configs["_model_name"] = model_name  # viz hook 使用

    # Phase 2: Graph Optimization (②-④)
    graph = _run_pass_list(graph, _OPTIMIZATION_PASSES, configs, collector,
                           output_dir, cube_size, debug_dump)
    _run_phase_checkpoint(collector, "optimization", graph, debug_dump)

    # Phase 3: Backend Annotation (⑤-⑤d)
    graph = _run_pass_list(graph, _ANNOTATION_PASSES, configs, collector,
                           output_dir, cube_size, debug_dump)
    _run_phase_checkpoint(collector, "annotation", graph, debug_dump)

    # Phase 4-5: Validation + Scheduling + Memory (⑥-⑧)
    graph = _run_pass_list(graph, _LATE_PASSES, configs, collector,
                           output_dir, cube_size, debug_dump)

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
        # 导出 timing 数据
        timing = configs.get("_pass_timing", {})
        if timing:
            timing_path = os.path.join(output_dir, "debug", "pass_timing.json")
            with open(timing_path, "w") as f:
                json.dump(timing, f, indent=2, ensure_ascii=False)

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
    pass_toggles: dict[str, bool] | str | None = None,
) -> Graph:
    """仅运行 Pass ①-⑧（不含 codegen），返回编排后的 Graph。

    用于 benchmark / 策略对比，跳过 C 代码生成和 golden 验证。
    pass_toggles="minimal" 关闭所有可选 pass。
    """
    configs = _resolve_compile_configs(model, config_dir, target_dtype, target_format, None)
    if pass_toggles == "minimal" or pass_toggles == "none":
        configs["pass_config"] = PassConfig.all_disabled()
    elif pass_toggles:
        base_pc: PassConfig = configs["pass_config"]
        configs["pass_config"] = PassConfig.from_dict({**asdict(base_pc), **pass_toggles})
    collector = DiagnosticCollector()

    graph = graph_capture.capture(model, dummy_input, mask=mask)
    cube_size = configs["hardware"]["fractal"]["cube_size"]
    configs["_model_name"] = type(model).__name__

    graph = _run_pass_list(graph, _OPTIMIZATION_PASSES, configs, collector,
                           output_dir, cube_size)
    graph = _run_pass_list(graph, _ANNOTATION_PASSES, configs, collector,
                           output_dir, cube_size)
    graph = _run_pass_list(graph, _LATE_PASSES, configs, collector,
                           output_dir, cube_size)
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
    print(format_model_annotations(model, inputs=inputs))  # noqa: print

    graph = graph_capture.capture(model, dummy_input, mask=mask)
    print("\n" + graph.summary())  # noqa: print
    print("\n" + graph.format_npu_annotations())  # noqa: print
    return graph
