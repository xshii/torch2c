"""main_emitter — 生成 main.c：加载 golden 输入，执行模型，dump 输出并比对。"""

from __future__ import annotations

import os
from pathlib import Path

from npu_compiler.common import get_logger

from .._helpers import write_file

logger = get_logger("codegen.main_emitter")

_TMPL_DIR = Path(__file__).parent.parent / "templates"


def _load_template(name: str) -> str:
    with open(_TMPL_DIR / name, "r", encoding="utf-8") as f:
        return f.read()


_DEFAULT_L1_BYTES = 16 * 1024 * 1024  # 16 MB


def _gen_load_inputs(inputs: list[dict]) -> str:
    """生成 load_inputs C 代码块。"""
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
    """生成 compare_outputs C 代码块。"""
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


def emit_main_c(
    plan: dict, hw_config: dict, *, atol: float = 1e-2, cosine_tol: float = 0.999
) -> str:
    """生成 main.c 内容。"""
    tensors = plan["tensors"]
    by_offset = lambda t: t.get("hbm_offset", 0) or 0  # noqa: E731

    inputs = sorted([t for t in tensors.values() if t.get("is_model_input")], key=by_offset)
    outputs = sorted([t for t in tensors.values() if t.get("is_model_output")], key=by_offset)

    hbm_size = 0
    for t in tensors.values():
        end = (t.get("hbm_offset", 0) or 0) + (t.get("hbm_size", 0) or 0)
        hbm_size = max(hbm_size, end)

    l1_size = hw_config.get("l1_capacity", _DEFAULT_L1_BYTES)

    template = _load_template("main.c.tmpl")
    return template.format(
        hbm_size=hbm_size,
        l1_size=l1_size,
        load_inputs=_gen_load_inputs(inputs),
        compare_outputs=_gen_compare_outputs(outputs, atol, cosine_tol),
    )


def run(
    plan: dict, hw_config: dict, output_dir: str, *, atol: float = 1e-2, cosine_tol: float = 0.999
) -> None:
    """生成 main.c 到 output_dir。"""
    logger.info("main_emitter: 开始生成 main.c")
    content = emit_main_c(plan, hw_config, atol=atol, cosine_tol=cosine_tol)
    write_file(os.path.join(output_dir, "main.c"), content)
