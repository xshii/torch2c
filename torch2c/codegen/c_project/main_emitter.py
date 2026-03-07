"""main_emitter — 生成 main.c：加载 golden 输入，执行模型，dump 输出并比对。

支持两种模式：
- file 模式（默认）：运行时从文件加载 golden 数据，依赖文件系统
- static 模式：golden 数据编译为静态数组，无文件系统依赖
"""

from __future__ import annotations

import os

from torch2c.common import get_logger

from .._helpers import load_template, write_file

logger = get_logger("codegen.main_emitter")


_DEFAULT_L1_BYTES = 16 * 1024 * 1024  # 16 MB


def _extract_io(plan: dict) -> tuple[list[dict], list[dict], int]:
    """提取输入/输出 tensor 信息和 HBM 总大小。"""
    tensors = plan["tensors"]
    by_offset = lambda t: t.get("hbm_offset", 0) or 0  # noqa: E731
    inputs = sorted([t for t in tensors.values() if t.get("is_model_input")], key=by_offset)
    outputs = sorted([t for t in tensors.values() if t.get("is_model_output")], key=by_offset)
    hbm_size = 0
    for t in tensors.values():
        end = (t.get("hbm_offset", 0) or 0) + (t.get("hbm_size", 0) or 0)
        hbm_size = max(hbm_size, end)
    return inputs, outputs, hbm_size


# ---- file 模式 ----


def _gen_load_inputs(inputs: list[dict]) -> str:
    lines: list[str] = []
    for i, t in enumerate(inputs):
        offset = t.get("hbm_offset", 0) or 0
        lines.append(f'    printf("Loading input {i}...\\n");')
        lines.append(
            f"    if (load_tensor(hbm, {offset}, "
            f'"golden/input_{i}.bin", "golden/input_{i}.desc") != 0) {{'
        )
        lines.append(f'        fprintf(stderr, "Failed to load input {i}\\n");')
        lines.append("        return 1;")
        lines.append("    }")
    return "\n".join(lines)


def _gen_compare_outputs(outputs: list[dict], atol: float, cosine_tol: float) -> str:
    lines: list[str] = []
    for i, t in enumerate(outputs):
        offset = t.get("hbm_offset", 0) or 0
        size = t.get("hbm_size", 0) or 0
        lines.append(f'    printf("Comparing output {i}...\\n");')
        lines.append(f'    dump_tensor(hbm, {offset}, {size}, "actual_output_{i}.bin");')
        lines.append(
            f"    if (compare_tensors("
            f'"actual_output_{i}.bin", "golden/output_{i}.bin", '
            f'"golden/output_{i}.desc", {atol}f, {cosine_tol}f, &result) != 0) {{'
        )
        lines.append(
            f"        printf("
            f'"  Output {i} FAIL: max_abs=%.6f cosine=%.6f\\n", '
            f"result.max_abs_diff, result.cosine_similarity);"
        )
        lines.append("        pass = 0;")
        lines.append("    } else {")
        lines.append(
            f"        printf("
            f'"  Output {i} PASS: max_abs=%.6f cosine=%.6f\\n", '
            f"result.max_abs_diff, result.cosine_similarity);"
        )
        lines.append("    }")
    return "\n".join(lines)


def _gen_debug_init(runtime_level: int) -> str:
    if runtime_level > 0:
        return f"    npu_debug_enable({runtime_level});\n"
    return ""


def _emit_file_mode(
    plan: dict, hw_config: dict, atol: float, cosine_tol: float,
    runtime_debug_level: int = 0,
) -> str:
    inputs, outputs, hbm_size = _extract_io(plan)
    l1_size = hw_config.get("l1_capacity", _DEFAULT_L1_BYTES)
    template = load_template("main.c.tmpl")
    return template.format(
        hbm_size=hbm_size,
        l1_size=l1_size,
        load_inputs=_gen_load_inputs(inputs),
        compare_outputs=_gen_compare_outputs(outputs, atol, cosine_tol),
        debug_init=_gen_debug_init(runtime_debug_level),
    )


# ---- static 模式 ----


def _gen_static_compare(outputs: list[dict], atol: float, elem_size: int) -> str:
    """生成静态模式的输出比对代码（内联，无文件 I/O）。"""
    lines: list[str] = []
    for i, t in enumerate(outputs):
        offset = t.get("hbm_offset", 0) or 0
        size = t.get("hbm_size", 0) or 0
        lines.append(f'    printf("Comparing output {i}...\\n");')
        lines.append("    {")
        lines.append(f"        float d = max_abs_diff(hbm + {offset}, golden_output_ptr({i}), {size}, {elem_size});")
        lines.append(f'        if (d > {atol}f) {{')
        lines.append(f'            printf("  Output {i} FAIL: max_abs=%.6f\\n", d);')
        lines.append("            pass = 0;")
        lines.append("        } else {")
        lines.append(f'            printf("  Output {i} PASS: max_abs=%.6f\\n", d);')
        lines.append("        }")
        lines.append("    }")
    return "\n".join(lines)


def _emit_static_mode(
    plan: dict, hw_config: dict, atol: float, elem_size: int,
    runtime_debug_level: int = 0,
) -> str:
    inputs, outputs, hbm_size = _extract_io(plan)
    l1_size = hw_config.get("l1_capacity", _DEFAULT_L1_BYTES)
    template = load_template("main_static.c.tmpl")
    return template.format(
        hbm_size=hbm_size,
        l1_size=l1_size,
        compare_outputs=_gen_static_compare(outputs, atol, elem_size),
        debug_init=_gen_debug_init(runtime_debug_level),
    )


# ---- 公共入口 ----


def emit_main_c(
    plan: dict, hw_config: dict, *, atol: float = 1e-2, cosine_tol: float = 0.999,
    static_mode: bool = False, elem_size: int = 2, runtime_debug_level: int = 0,
) -> str:
    """生成 main.c 内容。

    Args:
        static_mode: True 时生成无文件 I/O 的静态嵌入版本。
        elem_size: 静态模式下每个元素的字节数（fp16=2, fp32=4）。
        runtime_debug_level: C 侧运行时维测级别 (0=关闭)。
    """
    if static_mode:
        return _emit_static_mode(plan, hw_config, atol, elem_size, runtime_debug_level)
    return _emit_file_mode(plan, hw_config, atol, cosine_tol, runtime_debug_level)


def run(
    plan: dict, hw_config: dict, output_dir: str, *, atol: float = 1e-2, cosine_tol: float = 0.999,
    static_mode: bool = False, elem_size: int = 2, runtime_debug_level: int = 0,
) -> None:
    """生成 main.c 到 output_dir。"""
    mode_str = "static" if static_mode else "file"
    logger.info("main_emitter: 生成 main.c (mode=%s)", mode_str)
    content = emit_main_c(
        plan, hw_config, atol=atol, cosine_tol=cosine_tol,
        static_mode=static_mode, elem_size=elem_size,
        runtime_debug_level=runtime_debug_level,
    )
    write_file(os.path.join(output_dir, "main.c"), content)
