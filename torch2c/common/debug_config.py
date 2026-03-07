"""debug_config — 维测配置加载与全局访问。

从 debug.yaml 加载维测开关，提供模块级查询接口。
"""

from __future__ import annotations

import os

from .config_loader import load_config
from .logger import get_logger

logger = get_logger("debug_config")

# 默认值（debug.yaml 缺失时使用）
_DEFAULTS: dict = {
    "torch_trace": {"enabled": False, "leaf_only": False},
    "memory_layout": {"enabled": False},
    "c_mock_trace": {"compile_level": 0, "runtime_level": 0},
}

_config: dict | None = None


def load_debug_config(config_dir: str) -> dict:
    """加载 debug.yaml，缺失时返回全关默认值。"""
    global _config
    path = os.path.join(config_dir, "debug.yaml")
    if os.path.isfile(path):
        raw = load_config(path)
        # 合并默认值（yaml 中缺省的字段用默认值补齐）
        _config = {}
        for section, defaults in _DEFAULTS.items():
            user = raw.get(section, {})
            if not isinstance(user, dict):
                user = {}
            _config[section] = {**defaults, **user}
        logger.info("已加载维测配置: %s", path)
    else:
        _config = {k: dict(v) for k, v in _DEFAULTS.items()}
        logger.debug("debug.yaml 不存在, 使用默认值（全部关闭）")
    return _config


def get_debug_config() -> dict:
    """获取已加载的维测配置，未加载时返回默认值。"""
    if _config is None:
        return {k: dict(v) for k, v in _DEFAULTS.items()}
    return _config


def torch_trace_enabled() -> bool:
    cfg = get_debug_config()
    return bool(cfg["torch_trace"]["enabled"])


def torch_trace_leaf_only() -> bool:
    cfg = get_debug_config()
    return bool(cfg["torch_trace"]["leaf_only"])


def memory_layout_enabled() -> bool:
    cfg = get_debug_config()
    return bool(cfg["memory_layout"]["enabled"])


def c_mock_compile_level() -> int:
    cfg = get_debug_config()
    return int(cfg["c_mock_trace"]["compile_level"])


def c_mock_runtime_level() -> int:
    cfg = get_debug_config()
    return int(cfg["c_mock_trace"]["runtime_level"])
