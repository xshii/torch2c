"""mock_emitter — 原 npu_mock.h 生成器，已废弃。

npu_mock.h 是一个仅含 #include "npu_api.h" + "npu_debug.h" 的间接层，
已被消除：生成的 C 代码直接 #include "npu_api.h"。
"""

from __future__ import annotations

from torch2c.common import get_logger

logger = get_logger("codegen.mock_emitter")


def run(output_dir: str, config_dir: str | None = None) -> None:
    """No-op：npu_mock.h 已被消除，保留接口兼容。"""
    logger.debug("mock_emitter: npu_mock.h 已废弃，跳过")
