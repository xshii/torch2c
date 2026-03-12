"""_codegen_runner — Pass ⑨ codegen 执行辅助函数。"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

from torch2c.codegen import c_emitter, golden_exporter, weight_exporter
from torch2c.codegen._plan import CodegenPlan
from torch2c.codegen.c_project import (
    cmake_emitter,
    main_emitter,
    mock_emitter,
    utils_emitter,
)
from torch2c.common import Graph, c_mock_runtime_level, dtype_numpy, get_logger

logger = get_logger(__name__)


def _build_codegen_plan(graph: Graph, dma_plans: list) -> CodegenPlan:
    """构建 codegen 的类型化输入。"""
    return CodegenPlan(graph, dma_plans)


def _infer_io_dtypes(
    graph: Graph,
) -> tuple[list[type], list[type]]:
    """从 graph 推断每个 I/O tensor 的 numpy dtype。

    Returns:
        (input_numpy_dtypes, output_numpy_dtypes)
    """
    by_offset = lambda t: t.hbm_offset or 0  # noqa: E731
    inputs = sorted(
        [t for t in graph.tensors.values() if t.is_model_input], key=by_offset
    )
    outputs = sorted(
        [t for t in graph.tensors.values() if t.is_model_output], key=by_offset
    )
    in_dtypes = [dtype_numpy(t.dtype) if t.dtype else np.float16 for t in inputs]
    out_dtypes = [dtype_numpy(t.dtype) if t.dtype else np.float16 for t in outputs]
    return in_dtypes, out_dtypes


def _run_golden(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    numpy_dtype: type = np.float16,
    input_dtypes: list[type] | None = None,
    output_dtypes: list[type] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """前向推理获取 golden 输入/输出。

    支持 per-tensor dtype：input_dtypes/output_dtypes 覆盖 numpy_dtype。
    """
    model.eval()
    with torch.no_grad():
        out = model(dummy_input, mask) if mask is not None else model(dummy_input)

    # 输入
    in_dt_0 = input_dtypes[0] if input_dtypes else numpy_dtype
    inputs = [dummy_input.cpu().float().numpy().astype(in_dt_0)]
    if mask is not None:
        in_dt_1 = input_dtypes[1] if input_dtypes and len(input_dtypes) > 1 else numpy_dtype
        inputs.append(mask.cpu().float().numpy().astype(in_dt_1))

    # 输出
    if isinstance(out, torch.Tensor):
        out_dt = output_dtypes[0] if output_dtypes else numpy_dtype
        outputs = [out.cpu().float().numpy().astype(out_dt)]
    else:
        outputs = []
        for i, o in enumerate(out):
            out_dt = output_dtypes[i] if output_dtypes and i < len(output_dtypes) else numpy_dtype
            outputs.append(o.cpu().float().numpy().astype(out_dt))
    return inputs, outputs


def _emit_c_project(
    plan: CodegenPlan,
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
        runtime_debug_level=c_mock_runtime_level(),
    )
    cmake_emitter.run(output_dir)
    utils_emitter.run(output_dir)


def _compute_offsets(graph: Graph, attr: str) -> list[int]:
    """从 graph tensors 中按 hbm_offset 排序提取偏移列表。"""
    filtered = [t for t in graph.tensors.values() if getattr(t, attr, False)]
    filtered.sort(key=lambda t: t.hbm_offset or 0)
    return [t.hbm_offset or 0 for t in filtered]


def _np_to_str(np_dtype: type) -> str:
    return {np.float16: "fp16", np.float32: "fp32"}.get(np_dtype, "fp16")


def _export_weights_and_golden(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None,
    graph: Graph,
    output_dir: str,
    static_golden: bool,
) -> None:
    """导出权重和 golden 数据（文件模式 + 可选静态模式）。"""
    in_dtypes, out_dtypes = _infer_io_dtypes(graph)

    # 权重导出 — 每个权重使用 graph 中标注的 dtype
    weight_path = os.path.join(output_dir, "src", "model_weights.h")
    weights = [t for t in graph.tensors.values() if t.is_weight and t.name]
    weight_offsets = {t.name: t.hbm_offset or 0 for t in weights}
    per_weight_dtype = {t.name: t.dtype or "fp16" for t in weights}
    weight_exporter.export_weights(
        model.state_dict(), weight_path, dtype="fp16", offsets=weight_offsets,
        per_weight_dtype=per_weight_dtype,
    )

    # golden 前向推理 — per-tensor dtype
    inputs, outputs = _run_golden(
        model, dummy_input, mask,
        input_dtypes=in_dtypes, output_dtypes=out_dtypes,
    )

    # golden 数据：文件模式 — per-tensor dtype
    golden_dir = os.path.join(output_dir, "golden")
    in_dtype_strs = [_np_to_str(d) for d in in_dtypes]
    out_dtype_strs = [_np_to_str(d) for d in out_dtypes]
    golden_exporter.export_golden(
        inputs, outputs, golden_dir,
        dtype=in_dtype_strs, output_dtype=out_dtype_strs,
    )

    # 可选静态模式
    if static_golden:
        input_offsets = _compute_offsets(graph, "is_model_input")
        output_offsets = _compute_offsets(graph, "is_model_output")
        golden_exporter.export_golden_static(
            inputs, outputs, input_offsets, output_offsets,
            os.path.join(output_dir, "src", "model_golden.h"),
        )


def _copy_npu_mock(output_dir: str) -> None:
    """将 npu_cpu_mock 源码复制到 output_dir/npu_cpu_mock/，使交付件自包含。"""
    import shutil
    from torch2c.common import NPU_CPU_MOCK_DIR

    src = str(NPU_CPU_MOCK_DIR)
    dst = os.path.join(output_dir, "npu_cpu_mock")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    # 只复制 include/ 和 src/（不含 build/tests 等）
    os.makedirs(os.path.join(dst, "include"), exist_ok=True)
    os.makedirs(os.path.join(dst, "src"), exist_ok=True)
    for f in os.listdir(os.path.join(src, "include")):
        shutil.copy2(os.path.join(src, "include", f), os.path.join(dst, "include", f))
    for f in os.listdir(os.path.join(src, "src")):
        if f.endswith(".c"):
            shutil.copy2(os.path.join(src, "src", f), os.path.join(dst, "src", f))
    logger.info("npu_cpu_mock 已复制到 %s", dst)


def _compute_metrics(
    fp32_arr: np.ndarray, fp16_arr: np.ndarray,
) -> tuple[float, float]:
    """Compute max_abs_diff and cosine similarity between FP32 and FP16 arrays."""
    fp32_flat = fp32_arr.flatten().astype(np.float64)
    fp16_flat = fp16_arr.flatten().astype(np.float64)
    max_abs = float(np.max(np.abs(fp32_flat - fp16_flat)))
    dot = np.dot(fp32_flat, fp16_flat)
    norm_a = np.linalg.norm(fp32_flat)
    norm_b = np.linalg.norm(fp16_flat)
    cosine = float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 1.0
    return max_abs, cosine


def _build_module_to_node(graph: Graph) -> dict[str, str]:
    """Build mapping from module_path to node_id."""
    mapping: dict[str, str] = {}
    for nid, node in graph.nodes.items():
        if node.module_path:
            mapping[node.module_path] = nid
    return mapping


def _dump_intermediates(
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None,
    graph: Graph,
    output_dir: str,
) -> None:
    """Dump per-node intermediate tensors for precision diagnostics.

    Captures FP32 and FP16 intermediate outputs via forward hooks and
    saves them as .npy files in debug/intermediates/.
    """
    mod_to_node = _build_module_to_node(graph)
    if not mod_to_node:
        logger.warning("No module_path found in graph nodes; skipping intermediates dump")
        return

    inter_dir = os.path.join(output_dir, "debug", "intermediates")
    os.makedirs(inter_dir, exist_ok=True)

    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(mod_path: str):
        def hook(_module, _input, output):
            if isinstance(output, torch.Tensor):
                captured[mod_path] = output.detach().cpu().float()
        return hook

    # Register hooks on named modules that match graph nodes
    for name, mod in model.named_modules():
        if name in mod_to_node:
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input, mask) if mask is not None else model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Export intermediates and log metrics
    for mod_path, fp32_tensor in captured.items():
        nid = mod_to_node[mod_path]
        fp32_arr = fp32_tensor.numpy()
        fp16_arr = fp32_arr.astype(np.float16)
        np.save(os.path.join(inter_dir, f"{nid}.npy"), fp16_arr)
        max_abs, cosine = _compute_metrics(fp32_arr, fp16_arr)
        logger.info(
            "node %-12s  max_abs_diff=%.6f  cosine=%.6f  (%s)",
            nid, max_abs, cosine, mod_path,
        )

    logger.info("Intermediates dumped to %s (%d nodes)", inter_dir, len(captured))


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
    """Pass ⑨：C 代码生成 + 权重导出 + golden 数据 + mock 源码。"""
    logger.info("Pass ⑨ codegen 开始")
    plan = _build_codegen_plan(graph, dma_plans)
    # static 模式用首个输出 dtype 来决定 elem_size
    _, out_dtypes = _infer_io_dtypes(graph)
    out0_np = out_dtypes[0] if out_dtypes else np.float16

    _emit_c_project(
        plan, configs, config_dir, output_dir,
        out0_np, atol, cosine_tol, static_golden,
    )
    _export_weights_and_golden(
        model, dummy_input, mask, graph, output_dir, static_golden,
    )
    _copy_npu_mock(output_dir)
    logger.info("Pass ⑨ 完成")
