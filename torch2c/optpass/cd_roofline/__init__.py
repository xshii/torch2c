"""roofline — 算子计算强度分析 + 时延估算。"""

from .roofline_analyzer import run

# 导入内置 cost functions，触发 @register_cost_fn 注册
from . import _builtin_costs  # noqa: F401

__all__ = ["run"]
