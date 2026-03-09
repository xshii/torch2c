"""mock_emitter — 生成 npu_mock.h（重定向到 npu_api.h）。"""

from __future__ import annotations

import os

from torch2c.common import get_logger

from .._helpers import write_file

logger = get_logger("codegen.mock_emitter")

_NPU_MOCK_H = """\
/* npu_mock.h — redirect to npu_api.h (npu_cpu_mock provides implementation) */
#ifndef NPU_MOCK_H
#define NPU_MOCK_H
#include "npu_api.h"
#include "npu_debug.h"
#endif
"""


def emit_mock_h() -> str:
    """生成 npu_mock.h 内容。"""
    return _NPU_MOCK_H


def run(output_dir: str, config_dir: str | None = None) -> None:
    """生成 npu_mock.h 到 output_dir。"""
    logger.info("mock_emitter: 开始生成 npu_mock.h")
    write_file(os.path.join(output_dir, "npu_mock.h"), emit_mock_h())
