"""torch2c_config — 模型编译配置装饰器。

用法：
    @torch2c_config(
        target_dtype="fp16",
        target_format="nz",
        compute_dtype_rules={
            "default": "fp16",
            "by_compute_unit": {"cube": "fp32"},
            "by_op": {"vector_softmax_part1": "fp32"},
        },
    )
    class MyModel(nn.Module):
        ...

compile() 会自动读取 model._torch2c_config，参数仍可覆盖。
"""

from __future__ import annotations

from typing import Any


def torch2c_config(
    *,
    target_dtype: str | None = None,
    target_format: str | None = None,
    compute_dtype: str | None = None,
    compute_dtype_rules: dict[str, Any] | None = None,
) -> Any:
    """为 nn.Module 类附加编译配置。"""
    cfg: dict[str, Any] = {}
    if target_dtype is not None:
        cfg["target_dtype"] = target_dtype
    if target_format is not None:
        cfg["target_format"] = target_format
    if compute_dtype is not None:
        cfg["compute_dtype"] = compute_dtype
    if compute_dtype_rules is not None:
        cfg["compute_dtype_rules"] = compute_dtype_rules

    def decorator(cls):
        cls._torch2c_config = cfg
        return cls

    return decorator


def get_model_config(model) -> dict[str, Any]:
    """从模型实例读取 torch2c_config，返回空 dict 若未配置。"""
    return getattr(model, "_torch2c_config", {})
