"""验证生成的 C 工程输出完整性。"""

from __future__ import annotations

import os

from npu_compiler.common import get_logger

logger = get_logger(__name__)

_REQUIRED_SRC_FILES = [
    "model_graph.c",
    "model_graph.h",
    "model_memory.h",
    "model_params.h",
    "model_weights.h",
]

# 这些文件位于 output_dir 根目录（不在 src/ 下）
_REQUIRED_ROOT_FILES = [
    "main.c",
    "npu_mock.h",
    "CMakeLists.txt",
    "utils/data_loader.c",
    "utils/data_loader.h",
    "utils/comparator.c",
    "utils/comparator.h",
    "utils/data_dumper.c",
    "utils/data_dumper.h",
    "utils/tensor_utils.h",
]

_REQUIRED_GOLDEN_FILES = [
    "golden/input_0.bin",
    "golden/input_0.desc",
    "golden/output_0.bin",
    "golden/output_0.desc",
]


def validate(output_dir: str) -> list[str]:
    """验证输出目录结构和文件完整性。

    Returns:
        错误列表（空列表表示全部通过）。
    """
    errors: list[str] = []
    src_dir = os.path.join(output_dir, "src")

    for fname in _REQUIRED_SRC_FILES:
        path = os.path.join(src_dir, fname)
        if not os.path.isfile(path):
            errors.append(f"缺少文件: {path}")
        elif os.path.getsize(path) == 0:
            errors.append(f"文件为空: {path}")

    for fname in _REQUIRED_ROOT_FILES:
        path = os.path.join(output_dir, fname)
        if not os.path.isfile(path):
            errors.append(f"缺少文件: {path}")

    for fname in _REQUIRED_GOLDEN_FILES:
        path = os.path.join(output_dir, fname)
        if not os.path.isfile(path):
            errors.append(f"缺少文件: {path}")

    # 检查 model_graph.c 包含算子块
    graph_c = os.path.join(src_dir, "model_graph.c")
    if os.path.isfile(graph_c):
        with open(graph_c, "r") as f:
            content = f.read()
        op_count = content.count("/* ===")
        if op_count == 0:
            errors.append("model_graph.c 中未找到算子块")
        else:
            logger.info("model_graph.c 包含 %d 个算子块", op_count)

    if errors:
        for err in errors:
            logger.error(err)
        raise RuntimeError(f"输出验证失败: {len(errors)} 个错误")

    logger.info("输出验证通过!")
    return errors
