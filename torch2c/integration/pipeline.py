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
    _PassDesc("op_mapping", "②", op_mapping.run, "mapping", op_mapping.post_validate),
    _PassDesc("op_decomposition", "③", op_decomposition.run, "decomposition",
             op_decomposition.post_validate),
    _PassDesc("op_absorption", "④", op_absorption.run, "absorption",
             op_absorption.post_validate, toggle=OptionalPass.ABSORPTION),
    _PassDesc("mha_merge", "④b", mha_merge.run, "mha_merge",
             mha_merge.post_validate, toggle=OptionalPass.MHA_MERGE),
]

# Phase 3: Backend Annotation (⑤-⑤d)
_ANNOTATION_PASSES: list[_PassDesc] = [
    _PassDesc(
        "format_annotator", "⑤", format_annotator.run,
        "format", format_annotator.post_validate,
    ),
    _PassDesc(
        "format_planner", "⑤a", format_planner.run,
        "format_planner", format_planner.post_validate,
        toggle=OptionalPass.FORMAT_PLANNER,
    ),
    _PassDesc(
        "reformat_inserter", "⑤b", reformat_inserter.run,
        "reformat", reformat_inserter.post_validate,
    ),
    _PassDesc(
        "storage_assigner", "⑤c", storage_assigner.run,
        "storage", storage_assigner.post_validate,
        toggle=OptionalPass.STORAGE_ASSIGNER,
    ),
    _PassDesc(
        "block_pad", "⑤d", block_pad.run,
        "block_pad", block_pad.post_validate,
        toggle=OptionalPass.BLOCK_PAD,
        validate_with_config=True,
    ),
]

# Phase 4-5: Validation + Scheduling + Memory (⑥-⑧)
_LATE_PASSES: list[_PassDesc] = [
    _PassDesc(
        "validator", "⑥", validator.run, None,
        config_builder=lambda c: _build_validator_config(c["signatures"]),
        collect_errors=True,
    ),
    _PassDesc(
        "roofline_analyzer", "⑥b", roofline_analyzer.run, None,
        toggle=OptionalPass.ROOFLINE_ANALYZER,
    ),
    _PassDesc(
        "fusion_planner", "⑥c", fusion_planner.run, None,
        toggle=OptionalPass.FUSION_PLANNER,
    ),
    _PassDesc(
        "scheduler", "⑦", scheduler.run, None,
        scheduler.post_validate,
    ),
    _PassDesc(
        "global_tiler", "⑦b", global_tiler.run, None,
        toggle=OptionalPass.GLOBAL_TILER,
        post_hook=_global_tiler_post_hook,
    ),
    _PassDesc(
        "memory_planner", "⑧", memory_planner.run, "hardware",
        memory_planner.post_validate,
        viz_hook=_memory_planner_viz,
    ),
]


# ---- Pass 拓扑导出（供 viz 使用）----


_PHASE_LABELS = {
    "optimization": ("b_lowering", "Lowering"),
    "annotation": ("c_backend", "Backend"),
    "late": ("d_emission", "Emission"),
}


_PASS_DESC: dict[str, dict[str, str]] = {
    "graph_capture": {
        "input": "PyTorch nn.Module + dummy_input",
        "output": "Graph IR（ATen 算子图，含 shape/dtype/权重）",
        "desc": "通过 torch.export 捕获模型前向图，保留高级算子（layer_norm、softmax 等），"
                "不做自动分解。标注权重、模型输入输出、attention mask。",
    },
    "op_mapping": {
        "input": "ATen 算子图（op_type = aten.xxx）",
        "output": "NPU 算子图（npu_op = cube_xxx / vector_xxx / idma_xxx）",
        "desc": "1:1 映射 ATen 算子到 NPU 指令。Cube 单元处理矩阵乘（16×16×16 MAC），"
                "Vector 处理逐元素/归约（SIMD），IDMA 处理数据搬运（reshape/transpose）。"
                "未命中映射表的算子保留 is_mapped=False，留给 op_decomposition 裂解。",
    },
    "op_decomposition": {
        "input": "部分未映射的复合算子 + 已映射的逐元素算子",
        "output": "全部映射为 NPU 原子算子",
        "desc": "两阶段处理：(1) 裂解 — 将 is_mapped=False 的复合算子按 decompositions.yaml "
                "规则拆为多步原子操作；(2) 广播展开 — 逐元素算子（vector_add 等）的两个输入 "
                "shape 不匹配时插入 idma_broadcast 节点。",
    },
    "op_absorption": {
        "input": "独立的 bias/add 节点",
        "output": "bias 融入 matmul，减少节点数",
        "desc": "将 bias 加法吸收到前序 matmul 中（cube_matmul → cube_matmul_bias），"
                "省去独立 Vector 运算 + 中间 tensor 的 HBM 读写。权重 transpose 也通过 DMA "
                "ND→NZ 随路转换吸收。",
    },
    "mha_merge": {
        "input": "MHA 投影链（matmul→view→transpose→reshape）",
        "output": "保持 merged 或拆分为 per-head 投影",
        "desc": "对每个 MHA block 做成本分析：比较合并投影（padding 少但可能需要 tiling）"
                "和拆分投影（padding 多但免 tiling）的总开销，选择更优方案。",
    },
    "format_annotator": {
        "input": "格式未标注的 tensor（默认 nd）",
        "output": "每个 tensor 标注 format（nd/nz）和 dtype（fp16/fp32）",
        "desc": "根据算子标注（@npu 装饰器）和全局目标 dtype，为每个 tensor 设置 NPU 存储格式。"
                "Cube 权重需要 NZ 分形格式，激活默认 ND。",
    },
    "format_planner": {
        "input": "初始格式标注",
        "output": "全局最优格式分配",
        "desc": "考虑硬件格式能力约束（如 Cube src1 必须 NZ），分析 tensor 的生产者-消费者链路，"
                "选择使运行时格式转换次数最少的全局方案。",
    },
    "reformat_inserter": {
        "input": "可能存在格式冲突的图",
        "output": "插入显式格式转换节点",
        "desc": "在相邻算子 format 不匹配处插入 reformat 节点（通过 DMA 随路转换实现）。"
                "如果 format_planner 已消除冲突，则不插入。",
    },
    "storage_assigner": {
        "input": "所有 tensor 默认 storage=hbm",
        "output": "符合 bypass 条件的 tensor 标记为 local/pipe",
        "desc": "分析生产者-消费者的计算单元对，如果满足硬件 bypass 条件（如 cube→vector），"
                "则 tensor 不回写 HBM，直接走 L1 local buffer 或 pipe 直连，"
                "省去 2 次 DMA 搬运的带宽开销。",
    },
    "block_pad": {
        "input": "原始 shape 的 tensor",
        "output": "shape 对齐到硬件块尺寸的 tensor",
        "desc": "Cube 以 16×16 分形块为计算粒度，要求 tensor 最后维对齐到 16 的倍数、"
                "最后两维乘积对齐到 256。未对齐的 tensor 补零对齐，避免硬件处理边界碎片。",
    },
    "validator": {
        "input": "完成标注的图",
        "output": "校验通过或报错",
        "desc": "检查所有节点的 npu_op 是否在 c_api_signatures 支持列表中，"
                "确保图在目标硬件上可执行。不修改图。",
    },
    "roofline_analyzer": {
        "input": "已标注的图",
        "output": "每个节点标注计算/访存瓶颈",
        "desc": "计算每个算子的算术强度（FLOPs/Bytes），与硬件 roofline 拐点比较。"
                "计算受限的算子应优化计算效率，访存受限的应减少数据搬运（tiling/bypass/融合）。",
    },
    "fusion_planner": {
        "input": "独立的算子序列",
        "output": "标注融合组，中间 tensor 不落 HBM",
        "desc": "识别可融合的算子对（如 cube→vector 单消费者链），将它们标记为同一融合组。"
                "组内中间 tensor 留在 L1 不回写 HBM，减少 DMA 搬运。",
    },
    "scheduler": {
        "input": "无序的节点集合",
        "output": "确定 schedule_order 和 task_id",
        "desc": "基于数据依赖的拓扑排序，分配执行序号。"
                "无依赖的节点分配到不同 task_id，标记可并行。",
    },
    "global_tiler": {
        "input": "可能超出 L1 的大 tensor",
        "output": "标注 tiling 参数（tile_size, num_buffers）",
        "desc": "评估每个算子的 L1 峰值占用，如果超出 L1 容量则切分为多个 tile。"
                "双 buffer ping-pong 模式让 DMA 搬运与计算流水线重叠，隐藏访存延迟。",
    },
    "memory_planner": {
        "input": "未分配地址的 tensor",
        "output": "每个 tensor 的 HBM offset/size + DMA 计划",
        "desc": "基于生命周期分析分配 HBM 地址，避免冲突。"
                "生成 DMA load/store 指令序列，确定每个算子的数据搬运计划。",
    },
    "codegen": {
        "input": "完整编排的图 + DMA 计划",
        "output": "C99 工程（model_graph.c, CMakeLists.txt 等）",
        "desc": "将图翻译为 C 代码：每个算子生成三段式代码块（DMA 搬入 → NPU 算子调用 → DMA 搬出）。"
                "同时导出权重头文件和 golden 测试数据。",
    },
}


def get_pass_topology() -> dict:
    """导出 pass 拓扑结构，供可视化自动生成。

    Returns:
        {"phases": [...], "passes": [...]}
        每个 pass: {"name", "number", "phase", "optional", "input", "output", "desc"}
    """
    phases = [
        {"id": "a_capture", "label": "Capture"},
        {"id": "b_lowering", "label": "Lowering"},
        {"id": "c_backend", "label": "Backend"},
        {"id": "d_emission", "label": "Emission"},
    ]

    passes: list[dict] = []
    # graph_capture 特殊处理
    passes.append({
        "name": "graph_capture", "number": "①",
        "phase": "a_capture", "optional": False,
        **_PASS_DESC.get("graph_capture", {}),
    })

    phase_map = [
        (_OPTIMIZATION_PASSES, "b_lowering"),
        (_ANNOTATION_PASSES, "c_backend"),
        (_LATE_PASSES, "d_emission"),
    ]
    for pass_list, phase_id in phase_map:
        for p in pass_list:
            passes.append({
                "name": p.name,
                "number": p.number,
                "phase": phase_id,
                "optional": p.toggle is not None,
                **_PASS_DESC.get(p.name, {}),
            })

    # codegen 特殊处理
    passes.append({
        "name": "codegen", "number": "⑨",
        "phase": "d_emission", "optional": False,
        **_PASS_DESC.get("codegen", {}),
    })

    return {"phases": phases, "passes": passes}


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
        "format_planner": {
            "format_capabilities": hardware.get("format_capabilities", {}),
        },
        "reformat": {},
        "mha_merge": {
            **hardware.get("mha_merge", {}),
            "hardware": {
                "last_dim_align": (hardware.get("block_pad", {})
                    .get("alignment", {}).get("nd", {})
                    .get("fp16", [1, 16]))[1],
                "l1_size_bytes": hardware.get("memory", {}).get(
                    "l1", {}).get("total_size_bytes", 16 * 1024 * 1024),
                "dma_bytes_per_cycle": hardware.get("compute", {}).get(
                    "dma_bytes_per_cycle", 256),
                "matmul_launch_cycles": hardware.get("mha_merge", {}).get(
                    "matmul_launch_cycles", 100),
            },
        },
        "block_pad": hardware.get("block_pad", {}),
        "storage": {
            "enable_local_storage": True,
            **hardware.get("local_bypass", {}),
            **hardware.get("pipe_bypass", {}),
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
        merged = {**asdict(base_pc), **pass_toggles}
        configs["pass_config"] = PassConfig.from_dict(merged)
    collector = DiagnosticCollector()

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
    pass_toggles: dict[str, bool] | None = None,
) -> Graph:
    """仅运行 Pass ①-⑧（不含 codegen），返回编排后的 Graph。

    用于 benchmark / 策略对比，跳过 C 代码生成和 golden 验证。
    """
    configs = _resolve_compile_configs(model, config_dir, target_dtype, target_format, None)
    if pass_toggles:
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
