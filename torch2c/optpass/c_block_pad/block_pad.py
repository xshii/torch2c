"""block_pad — Pass⑤d：将 tensor shape 对齐到 cube 分形块尺寸。

规则（ndim >= 2）：
  - 最后 1 维对齐到 last_dim 的倍数（默认 16）
  - 最后 2 维的乘积对齐到 last_two_product 的倍数（默认 256）

规则（ndim == 1）：
  - 唯一维度对齐到 single_dim 的倍数（默认 = last_two_product）

scalar tensor (ndim == 0) 跳过。
特定 dtype 可覆盖默认对齐参数（by_dtype 优先于 default）。

配置示例（hardware_config.yaml）：
  block_pad:
    default:
      last_dim: 16
      last_two_product: 256
      single_dim: 256          # 1D tensor 对齐，省略时等于 last_two_product
    by_dtype:
      int8:
        last_dim: 32
        last_two_product: 1024

对齐后原始 shape 保存在 tensor.original_shape 中，供 codegen 使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)

# 硬编码兜底默认值（config 为空时使用）
_FALLBACK_LAST_DIM = 16
_FALLBACK_LAST_TWO_PRODUCT = 256


@dataclass(frozen=True)
class PadRule:
    """单个 dtype 的对齐规则。"""

    last_dim: int          # ndim >= 2 时最后 1 维对齐
    last_two_product: int  # ndim >= 2 时最后 2 维乘积对齐
    single_dim: int        # ndim == 1 时唯一维度对齐


def _parse_pad_rules(config: dict) -> tuple[PadRule, dict[str, PadRule]]:
    """解析 block_pad 配置，返回 (default_rule, {dtype: rule})。

    config 即 hardware_config.yaml 中 block_pad section 的内容。
    """
    default_raw = config.get("default", {})
    ld = default_raw.get("last_dim", _FALLBACK_LAST_DIM)
    ltp = default_raw.get("last_two_product", _FALLBACK_LAST_TWO_PRODUCT)
    sd = default_raw.get("single_dim", ltp)  # 省略时等于 last_two_product
    default = PadRule(last_dim=ld, last_two_product=ltp, single_dim=sd)

    by_dtype: dict[str, PadRule] = {}
    for dtype, overrides in (config.get("by_dtype") or {}).items():
        d_ld = overrides.get("last_dim", default.last_dim)
        d_ltp = overrides.get("last_two_product", default.last_two_product)
        d_sd = overrides.get("single_dim", d_ltp)
        by_dtype[dtype] = PadRule(last_dim=d_ld, last_two_product=d_ltp, single_dim=d_sd)
    return default, by_dtype


def _ceil_to(value: int, multiple: int) -> int:
    """向上取整到 multiple 的倍数。"""
    return math.ceil(value / multiple) * multiple


def _pad_shape(shape: list[int], rule: PadRule) -> list[int]:
    """计算 block-padded shape。

    Returns:
        新的 shape 列表（不修改原列表）。
    """
    ndim = len(shape)
    if ndim == 0:
        return list(shape)

    padded = list(shape)

    if ndim == 1:
        padded[0] = _ceil_to(padded[0], rule.single_dim)
        return padded

    # ndim >= 2: 最后 1 维对齐
    padded[-1] = _ceil_to(padded[-1], rule.last_dim)

    if rule.last_two_product > 0:
        # 最后 2 维乘积对齐
        product = padded[-2] * padded[-1]
        if product % rule.last_two_product != 0:
            factor = rule.last_two_product // math.gcd(
                padded[-1], rule.last_two_product,
            )
            padded[-2] = _ceil_to(padded[-2], factor)

    return padded


def _validate_shape(t_id: str, shape: list[int], rule: PadRule) -> list[str]:
    """校验单个 tensor 的 shape 是否满足 rule 约束。"""
    ndim = len(shape)
    if ndim == 0:
        return []
    errors: list[str] = []
    if ndim == 1:
        if shape[0] % rule.single_dim != 0:
            errors.append(
                f"tensor {t_id}: 1D 维度 {shape[0]} "
                f"不是 {rule.single_dim} 的倍数"
            )
        return errors
    if shape[-1] % rule.last_dim != 0:
        errors.append(
            f"tensor {t_id}: 最后 1 维 {shape[-1]} "
            f"不是 {rule.last_dim} 的倍数"
        )
    product = shape[-2] * shape[-1]
    if product % rule.last_two_product != 0:
        errors.append(
            f"tensor {t_id}: 最后 2 维乘积 {product} "
            f"不是 {rule.last_two_product} 的倍数"
        )
    return errors


def run(graph: Graph, config: dict) -> Graph:
    """对所有 tensor 的 shape 做 block 对齐。

    Args:
        graph: 经过 storage_assigner 的 Graph IR。
        config: block_pad 配置字典，可含 default / by_dtype。

    Returns:
        同一 Graph 对象（原地修改）。
    """
    default_rule, by_dtype = _parse_pad_rules(config)
    if by_dtype:
        logger.info(
            "block_pad 规则: default=%s, 特定dtype=%s",
            default_rule, {k: v for k, v in by_dtype.items()},
        )

    pad_count = 0
    for t in graph.tensors.values():
        if len(t.shape) == 0:
            continue
        rule = by_dtype.get(t.dtype, default_rule)
        padded = _pad_shape(t.shape, rule)
        if padded != t.shape:
            t.original_shape = list(t.shape)
            t.shape = padded
            pad_count += 1
            logger.debug(
                "tensor %s (%s): shape %s → %s",
                t.id, t.dtype, t.original_shape, t.shape,
            )
            # 在 producer 节点上记录 opt_log
            producer = graph.get_node(t.producer_node_id) if t.producer_node_id else None
            if producer:
                ndim = len(t.original_shape)
                if ndim == 1:
                    align_desc = f"1D 维度对齐到 {rule.single_dim} 的倍数"
                else:
                    align_desc = (
                        f"最后 1 维对齐到 {rule.last_dim} 的倍数，"
                        f"最后 2 维乘积对齐到 {rule.last_two_product} 的倍数"
                    )
                log_opt(
                    producer, "block_pad", "shape 对齐",
                    f"{t.id}: {t.original_shape}→{padded}。"
                    f"Cube 以 16×16 分形块为计算粒度，{align_desc}，"
                    f"避免硬件处理边界碎片时的效率损失",
                )
        else:
            t.original_shape = None

    logger.info("block_pad 完成: %d 个 tensor 的 shape 已对齐", pad_count)
    return graph


def post_validate(graph: Graph, config: dict | None = None) -> list[str]:
    """校验 block-pad 后的 shape 约束。

    Args:
        graph: 经过 block_pad 的 Graph IR。
        config: block_pad 配置字典，可含 default / by_dtype。
                省略或 None 时使用硬编码兜底规则。
    """
    if config:
        default_rule, by_dtype = _parse_pad_rules(config)
    else:
        default_rule = PadRule(
            last_dim=_FALLBACK_LAST_DIM,
            last_two_product=_FALLBACK_LAST_TWO_PRODUCT,
            single_dim=_FALLBACK_LAST_TWO_PRODUCT,
        )
        by_dtype = {}
    errors: list[str] = []
    for t in graph.tensors.values():
        rule = by_dtype.get(t.dtype, default_rule)
        errors.extend(_validate_shape(t.id, t.shape, rule))
    return errors
