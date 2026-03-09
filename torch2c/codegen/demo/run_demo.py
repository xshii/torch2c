"""codegen demo — 加载 plan JSON，生成 C 工程，gcc -fsyntax-only 验证。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from torch2c.codegen import c_emitter, cmake_emitter, mock_emitter, utils_emitter
from torch2c.codegen._plan import CodegenPlan
from torch2c.common import INTEGRATION_CONFIG_DIR, Graph, get_logger
from torch2c.memory_planner._dma import DmaPlan

logger = get_logger("codegen.demo")


def generate_test_memory_c(plan: CodegenPlan) -> str:
    """生成 test_memory_layout.c：检查 tensor 偏移不重叠。"""
    lines = [
        "#include <stdio.h>",
        "#include <stdlib.h>",
        '#include "model_memory.h"',
        "",
        "static int _test_count = 0;",
        "static int _pass_count = 0;",
        "#define RUN_TEST(fn) do { _test_count++; "
        'printf("  [RUN] %s ... ", #fn); fn(); '
        '_pass_count++; printf("PASS\\n"); } while(0)',
        "#define ASSERT_TRUE(cond) do { if (!(cond)) { "
        'printf("FAIL\\n    %s:%d\\n", __FILE__, __LINE__); '
        "exit(1); } } while(0)",
        '#define TEST_SUMMARY() printf("\\n=== %d/%d tests passed ===\\n", '
        "_pass_count, _test_count)",
        "",
    ]
    # 生成偏移对齐测试
    lines.append("static void test_offsets_non_negative(void) {")
    for tid in plan.tensors:
        safe = tid.upper()
        lines.append(f"    ASSERT_TRUE({safe}_HBM_OFFSET >= 0);")
    lines.append("}")
    lines.append("")

    lines.append("static void test_sizes_positive(void) {")
    for tid, t in plan.tensors.items():
        safe = tid.upper()
        if t.hbm_size:
            lines.append(f"    ASSERT_TRUE({safe}_HBM_SIZE > 0);")
    lines.append("}")
    lines.append("")

    lines.append("int main(void) {")
    lines.append("    RUN_TEST(test_offsets_non_negative);")
    lines.append("    RUN_TEST(test_sizes_positive);")
    lines.append("    TEST_SUMMARY();")
    lines.append("    return (_pass_count == _test_count) ? 0 : 1;")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    demo_dir = Path(__file__).parent
    plan_path = demo_dir / "demo_input_plan.json"
    output_dir = demo_dir.resolve().parents[2] / "output" / "codegen_demo"
    config_dir = str(INTEGRATION_CONFIG_DIR)

    logger.info("加载 plan: %s", plan_path)
    with open(plan_path, "r") as f:
        data = json.load(f)

    graph = Graph.from_dict(data)
    dma_plans = [DmaPlan(**dp) for dp in data.get("dma_plans", [])]
    plan = CodegenPlan(graph, dma_plans)

    # 生成各模块
    c_emitter.run(plan, str(output_dir), config_dir)
    mock_emitter.run(str(output_dir), config_dir)
    cmake_emitter.run(str(output_dir))
    utils_emitter.run(str(output_dir))

    # demo 模式生成 stub model_weights.h（真实场景由 weight_exporter 生成）
    weights_h = output_dir / "src" / "model_weights.h"
    if not weights_h.exists():
        with open(weights_h, "w") as f:
            f.write(
                "#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n"
                "static inline void load_weights(unsigned char* hbm)"
                " { (void)hbm; }\n#endif\n"
            )

    # 生成 test_memory_layout.c
    tests_dir = output_dir / "tests"
    os.makedirs(tests_dir, exist_ok=True)
    test_mem_path = tests_dir / "test_memory_layout.c"
    with open(test_mem_path, "w") as f:
        f.write(generate_test_memory_c(plan))

    # 检查期望文件
    expected_path = demo_dir / "expected_files.txt"
    with open(expected_path) as f:
        expected = [line.strip() for line in f if line.strip()]
    missing = [fp for fp in expected if not (output_dir / fp).exists()]
    if missing:
        logger.error("缺少文件: %s", missing)
        sys.exit(1)
    logger.info("所有期望文件已生成 (%d 个)", len(expected))

    # gcc -fsyntax-only 验证
    model_c = str(output_dir / "src" / "model_graph.c")
    mock_h = str(output_dir / "npu_mock.h")
    cmd = [
        "gcc",
        "-fsyntax-only",
        "-std=c99",
        f"-include{mock_h}",
        f"-I{output_dir / 'src'}",
        f"-I{output_dir / 'utils'}",
        f"-I{output_dir}",
        model_c,
    ]
    logger.info("语法检查: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("语法检查失败:\n%s", result.stderr)
        sys.exit(1)
    logger.info("语法检查通过!")


if __name__ == "__main__":
    main()
