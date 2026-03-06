"""验证生成的 C 工程：编译 + 链接 npu_cpu_mock + 运行 + golden 比对。"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

from npu_compiler.common import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
_NPU_MOCK_DIR = _PROJECT_ROOT / "npu_cpu_mock"

# npu_cpu_mock 的所有源文件
_MOCK_SOURCES = [
    "src/npu_dtype_utils.c",
    "src/npu_compute_elementwise.c",
    "src/npu_compute_matmul.c",
    "src/npu_compute_norm.c",
    "src/npu_compute_softmax.c",
    "src/npu_compute_transpose.c",
    "src/npu_dma.c",
    "src/npu_sync.c",
]

_NPU_MOCK_SHIM = """\
/* npu_mock.h — redirect to npu_api.h for CPU mock validation */
#ifndef NPU_MOCK_H
#define NPU_MOCK_H
#include "npu_api.h"
#endif
"""


def _find_cc() -> str | None:
    """查找可用的 C 编译器。"""
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return cc
    return None


def validate_c(output_dir: str) -> dict:
    """编译并运行生成的 C 工程，比对 golden 数据。

    Args:
        output_dir: compile() 的输出目录。

    Returns:
        包含 passed, stdout, stderr 的结果 dict。

    Raises:
        RuntimeError: 编译或运行失败。
    """
    cc = _find_cc()
    if cc is None:
        raise RuntimeError("未找到 C 编译器 (cc/gcc/clang)")

    mock_dir = str(_NPU_MOCK_DIR)
    if not os.path.isdir(mock_dir):
        raise RuntimeError(f"npu_cpu_mock 目录不存在: {mock_dir}")

    # 用 npu_api.h 包装替换 npu_mock.h（仅覆盖 stub 声明头文件）
    mock_h_path = os.path.join(output_dir, "npu_mock.h")
    logger.info("替换 npu_mock.h 为 npu_api.h 包装")
    with open(mock_h_path, "w") as f:
        f.write(_NPU_MOCK_SHIM)

    # 构建编译命令
    mock_sources = [os.path.join(mock_dir, src) for src in _MOCK_SOURCES]
    project_sources = [
        "main.c",
        "src/model_graph.c",
        "utils/data_loader.c",
        "utils/data_dumper.c",
        "utils/comparator.c",
    ]
    exe_name = "npu_model_run"

    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-O2",
        "-o",
        exe_name,
        *project_sources,
        *mock_sources,
        "-I.",
        "-Isrc",
        f"-I{os.path.join(mock_dir, 'include')}",
        "-lm",
    ]

    # 编译
    logger.info("编译 C 工程: %s", " ".join(cmd[:6]) + " ...")
    comp = subprocess.run(
        cmd,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if comp.returncode != 0:
        logger.error("编译失败:\n%s", comp.stderr)
        raise RuntimeError(f"C 编译失败 (exit {comp.returncode}): {comp.stderr}")

    logger.info("编译成功")

    # 运行
    exe_path = os.path.join(output_dir, exe_name)
    logger.info("运行 C 工程: %s", exe_path)
    run = subprocess.run(
        [f"./{exe_name}"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )

    passed = run.returncode == 0
    result = {
        "passed": passed,
        "stdout": run.stdout,
        "stderr": run.stderr,
        "returncode": run.returncode,
    }

    if passed:
        logger.info("C golden 比对通过!\n%s", run.stdout)
    else:
        logger.error("C golden 比对失败 (exit %d):\n%s\n%s", run.returncode, run.stdout, run.stderr)

    return result
