"""验证生成的 C 工程：编译 + 链接 npu_cpu_mock + 运行 + golden 比对。"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess

from torch2c.common import c_mock_compile_level, get_logger

logger = get_logger(__name__)


def _find_cc() -> str | None:
    """查找可用的 C 编译器。"""
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return cc
    return None


def validate_c(output_dir: str, *, c_debug_level: int | None = None) -> dict:
    """编译并运行生成的 C 工程，比对 golden 数据。

    使用 output_dir/npu_cpu_mock/ 中的本地 mock 源码（自包含交付件）。

    Returns:
        包含 passed, stdout, stderr 的结果 dict。
    """
    cc = _find_cc()
    if cc is None:
        raise RuntimeError("未找到 C 编译器 (cc/gcc/clang)")

    if c_debug_level is None:
        c_debug_level = c_mock_compile_level()

    mock_dir = os.path.join(output_dir, "npu_cpu_mock")
    if not os.path.isdir(mock_dir):
        raise RuntimeError(f"npu_cpu_mock 目录不存在: {mock_dir}")

    # 构建编译命令
    mock_sources = glob.glob(os.path.join(mock_dir, "src", "*.c"))
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
        f"-DNPU_DEBUG_LEVEL={c_debug_level}",
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
    logger.info("运行 C 工程: %s/%s", output_dir, exe_name)
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
        logger.error("C golden 比对失败 (exit %d):\n%s\n%s",
                     run.returncode, run.stdout, run.stderr)

    return result
