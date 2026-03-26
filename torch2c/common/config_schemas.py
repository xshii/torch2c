"""config_schemas — 类型安全的 pass 配置定义。

每个 dataclass 对应一个 pass 的 config 结构。提供 from_raw(dict) 将原始 dict
（来自 YAML / pipeline._load_configs()）转为类型安全对象。

用法（渐进迁移）：
    # 旧代码（仍然工作）
    threshold = config.get("prefer_merged_threshold", 0.9)

    # 新代码
    cfg = MhaMergeConfig.from_raw(config)
    threshold = cfg.prefer_merged_threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_DEFAULT_L1_SIZE = 16 * 1024 * 1024  # 16 MB
_DEFAULT_FALLBACK = [16, 16]


# ---------------------------------------------------------------------------
# MhaMergeConfig — mha_merge pass (④b)
# ---------------------------------------------------------------------------

@dataclass
class MhaMergeConfig:
    """MHA merge/split 决策参数。

    pipeline._load_configs() 构建的 mha_merge dict 包含：
      - prefer_merged_threshold, max_batch_for_split  (来自 hardware.mha_merge)
      - hardware.last_dim_align, hardware.l1_size_bytes, hardware.dma_bytes_per_cycle
      - cost_model  (来自 cost_model_config.yaml, 透传给 roofline)
    """

    prefer_merged_threshold: float = 0.9
    max_batch_for_split: int = 1
    last_dim_align: int = 16
    l1_size_bytes: int = _DEFAULT_L1_SIZE
    dma_bytes_per_cycle: int = 256
    cost_model: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, d: dict[str, Any]) -> MhaMergeConfig:
        """从 pipeline 生成的 mha_merge 原始 dict 构建。

        兼容两种结构：
          - 扁平结构：所有键在顶层
          - 嵌套结构：硬件参数在 ``d["hardware"]`` 子 dict 中
        """
        hw = d.get("hardware", {})
        return cls(
            prefer_merged_threshold=d.get("prefer_merged_threshold", 0.9),
            max_batch_for_split=d.get("max_batch_for_split", 1),
            last_dim_align=hw.get("last_dim_align",
                                  d.get("last_dim_align", 16)),
            l1_size_bytes=hw.get("l1_size_bytes",
                                 d.get("l1_size_bytes", _DEFAULT_L1_SIZE)),
            dma_bytes_per_cycle=hw.get("dma_bytes_per_cycle",
                                       d.get("dma_bytes_per_cycle", 256)),
            cost_model=d.get("cost_model", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """转回 dict 供旧代码使用。保持 pipeline 的嵌套布局。"""
        return {
            "prefer_merged_threshold": self.prefer_merged_threshold,
            "max_batch_for_split": self.max_batch_for_split,
            "hardware": {
                "last_dim_align": self.last_dim_align,
                "l1_size_bytes": self.l1_size_bytes,
                "dma_bytes_per_cycle": self.dma_bytes_per_cycle,
            },
            "cost_model": self.cost_model,
        }


# ---------------------------------------------------------------------------
# FormatConfig — format_annotator pass (⑤)
# ---------------------------------------------------------------------------

@dataclass
class FormatConfig:
    """Format/dtype 标注参数。

    config 可选键：
        target_dtype:         存储 dtype（如 "fp16"），None 继承模型原始值
        target_format:        存储 format（如 "nz"），None 继承模型原始值
        compute_dtype:        全局计算精度（如 "fp32"），None 跟随 target_dtype
        compute_dtype_rules:  分层计算精度规则 dict
    """

    target_dtype: str | None = None
    target_format: str | None = None
    compute_dtype: str | None = None
    compute_dtype_rules: dict | None = None

    @classmethod
    def from_raw(cls, d: dict[str, Any]) -> FormatConfig:
        """从 pipeline 生成的 format 原始 dict 构建。"""
        return cls(
            target_dtype=d.get("target_dtype"),
            target_format=d.get("target_format"),
            compute_dtype=d.get("compute_dtype"),
            compute_dtype_rules=d.get("compute_dtype_rules"),
        )

    def to_dict(self) -> dict[str, Any]:
        """转回 dict。仅包含非 None 键，与 pipeline 行为一致。"""
        result: dict[str, Any] = {}
        if self.target_dtype is not None:
            result["target_dtype"] = self.target_dtype
        if self.target_format is not None:
            result["target_format"] = self.target_format
        if self.compute_dtype is not None:
            result["compute_dtype"] = self.compute_dtype
        if self.compute_dtype_rules is not None:
            result["compute_dtype_rules"] = self.compute_dtype_rules
        return result


# ---------------------------------------------------------------------------
# BlockPadConfig — block_pad pass (⑤d)
# ---------------------------------------------------------------------------

@dataclass
class BlockPadConfig:
    """Block padding 对齐参数。

    直接对应 hardware_config.yaml 的 ``block_pad`` 节：
        alignment:  format -> dtype -> [dim[-2]对齐, dim[-1]对齐]
        fallback:   未匹配时兜底 [dim[-2], dim[-1]]
        single_dim: 1D tensor 对齐值
    """

    alignment: dict = field(default_factory=dict)
    fallback: list[int] = field(default_factory=lambda: [16, 16])
    single_dim: int = 256

    @classmethod
    def from_raw(cls, d: dict[str, Any]) -> BlockPadConfig:
        """从 hardware_config.yaml 的 block_pad 节构建。"""
        raw_fallback = d.get("fallback")
        # YAML 可能加载为 tuple，统一转 list
        fallback = list(raw_fallback) if raw_fallback is not None else list(_DEFAULT_FALLBACK)
        return cls(
            alignment=d.get("alignment", {}),
            fallback=fallback,
            single_dim=d.get("single_dim", 256),
        )

    def get_align(self, fmt: str, dtype: str) -> list[int]:
        """查询指定 format x dtype 的对齐值，未匹配返回 fallback。"""
        return self.alignment.get(fmt, {}).get(dtype, list(self.fallback))

    def to_dict(self) -> dict[str, Any]:
        """转回 dict。"""
        return {
            "alignment": self.alignment,
            "fallback": self.fallback,
            "single_dim": self.single_dim,
        }
