"""block_pad — Pass⑤d：将 tensor shape 对齐到硬件块尺寸。

根据 tensor 的 format 和 dtype 查询二维对齐表，获取 dim[-2] 和 dim[-1]
各自的对齐要求。不同格式的块结构不同：

  - ND:  无分形块，dim[-1] 按 SIMD 宽度对齐，dim[-2] 通常不需对齐
  - NZ:  Fractal_NZ，列优先块 [c0, cube_size]
  - ZZ:  Fractal_Z， 行优先块 [cube_size, c0]
  - NN:  列优先块   [c0, cube_size]

c0 随 dtype 变化（fp16=16, int8=32），因此同格式不同 dtype 对齐不同。

配置示例（hardware_config.yaml）：
  block_pad:
    alignment:           # format → dtype → [dim[-2]对齐, dim[-1]对齐]
      nd:
        fp16: [1, 16]
        int8: [1, 32]
      nz:
        fp16: [16, 16]
        int8: [32, 16]
      zz:
        fp16: [16, 16]
        int8: [16, 32]
    fallback: [16, 16]   # 未匹配时兜底
    single_dim: 256       # 1D tensor 对齐

对齐后原始 shape 保存在 tensor.original_shape 中，供 codegen 使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)

# 硬编码兜底默认值（config 为空时使用）
_FALLBACK_ALIGN = (16, 16)
_FALLBACK_SINGLE_DIM = 256


@dataclass(frozen=True)
class AlignRule:
    """单个 (format, dtype) 组合的对齐规则。"""

    dim_neg2: int   # dim[-2] 对齐倍数（1 表示不对齐）
    dim_neg1: int   # dim[-1] 对齐倍数
    single_dim: int  # 1D tensor 唯一维度对齐


def _ceil_to(value: int, multiple: int) -> int:
    """向上取整到 multiple 的倍数。"""
    if multiple <= 1:
        return value
    return math.ceil(value / multiple) * multiple


def parse_alignment_table(
    config: dict,
) -> tuple[dict[tuple[str, str], AlignRule], AlignRule]:
    """解析 block_pad 配置，返回 (lookup_table, fallback_rule)。

    lookup_table key = (format, dtype)，value = AlignRule。
    """
    alignment = config.get("alignment", {})
    fb = config.get("fallback", list(_FALLBACK_ALIGN))
    single_dim = config.get("single_dim", _FALLBACK_SINGLE_DIM)
    fallback = AlignRule(dim_neg2=fb[0], dim_neg1=fb[1], single_dim=single_dim)

    table: dict[tuple[str, str], AlignRule] = {}
    for fmt, dtype_map in alignment.items():
        if not isinstance(dtype_map, dict):
            continue
        for dtype, pair in dtype_map.items():
            table[(fmt, dtype)] = AlignRule(
                dim_neg2=pair[0], dim_neg1=pair[1], single_dim=single_dim,
            )

    return table, fallback


def get_align_rule(
    table: dict[tuple[str, str], AlignRule],
    fallback: AlignRule,
    fmt: str,
    dtype: str,
) -> AlignRule:
    """查表获取对齐规则，未命中时返回 fallback。"""
    return table.get((fmt, dtype), fallback)


def pad_shape(shape: list[int], rule: AlignRule) -> list[int]:
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

    # ndim >= 2: 两个维度独立对齐
    padded[-1] = _ceil_to(padded[-1], rule.dim_neg1)
    padded[-2] = _ceil_to(padded[-2], rule.dim_neg2)

    return padded


def _validate_shape(t_id: str, shape: list[int], rule: AlignRule) -> list[str]:
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
    if rule.dim_neg1 > 1 and shape[-1] % rule.dim_neg1 != 0:
        errors.append(
            f"tensor {t_id}: dim[-1]={shape[-1]} "
            f"不是 {rule.dim_neg1} 的倍数"
        )
    if rule.dim_neg2 > 1 and shape[-2] % rule.dim_neg2 != 0:
        errors.append(
            f"tensor {t_id}: dim[-2]={shape[-2]} "
            f"不是 {rule.dim_neg2} 的倍数"
        )
    return errors


def run(graph: Graph, config: dict) -> Graph:
    """对所有 tensor 的 shape 做 format-aware block 对齐。

    Args:
        graph: 经过 format_annotator 的 Graph IR（tensor.format 已标注）。
        config: block_pad 配置字典，含 alignment / fallback / single_dim。

    Returns:
        同一 Graph 对象（原地修改）。
    """
    table, fallback = parse_alignment_table(config)

    pad_count = 0
    for t in graph.tensors.values():
        if len(t.shape) == 0:
            continue
        rule = get_align_rule(table, fallback, t.format, t.dtype)
        padded = pad_shape(t.shape, rule)
        if padded != t.shape:
            t.original_shape = list(t.shape)
            t.shape = padded
            pad_count += 1
            logger.debug(
                "tensor %s (format=%s, dtype=%s): shape %s → %s",
                t.id, t.format, t.dtype, t.original_shape, t.shape,
            )
            # 在 producer 节点上记录 opt_log
            producer = graph.get_node(t.producer_node_id) if t.producer_node_id else None
            if producer:
                log_opt(
                    producer, "block_pad", "shape 对齐",
                    f"{t.id}: {t.original_shape}→{padded}。"
                    f"format={t.format}, dtype={t.dtype} 要求 "
                    f"dim[-2] 对齐到 {rule.dim_neg2}、"
                    f"dim[-1] 对齐到 {rule.dim_neg1}",
                )
        else:
            t.original_shape = None

    logger.info("block_pad 完成: %d 个 tensor 的 shape 已对齐", pad_count)
    return graph


def post_validate(graph: Graph, config: dict | None = None) -> list[str]:
    """校验 block-pad 后的 shape 约束。"""
    if config:
        table, fallback = parse_alignment_table(config)
    else:
        fallback = AlignRule(
            dim_neg2=_FALLBACK_ALIGN[0],
            dim_neg1=_FALLBACK_ALIGN[1],
            single_dim=_FALLBACK_SINGLE_DIM,
        )
        table = {}

    errors: list[str] = []
    for t in graph.tensors.values():
        rule = get_align_rule(table, fallback, t.format, t.dtype)
        errors.extend(_validate_shape(t.id, t.shape, rule))
    return errors
