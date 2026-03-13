"""PassConfig — 可选 Pass 的类型安全开关。

用法：
    from torch2c.common.pass_config import OptionalPass, PassConfig

    # 全部启用（默认）
    cfg = PassConfig()

    # 选择性关闭
    cfg = PassConfig(absorption=False, global_tiler=False)

    # 从 yaml dict 构建
    cfg = PassConfig.from_dict({"absorption": False})

    # 检查
    cfg.is_enabled(OptionalPass.ABSORPTION)  # False
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum, auto


class OptionalPass(Enum):
    """可选 Pass 枚举。

    每个成员对应一个可关闭的 Pass，成员名与 PassConfig 字段名一致（大写）。
    """

    ABSORPTION = auto()         # ④ op_absorption
    MHA_MERGE = auto()          # ④b mha_merge
    FORMAT_PLANNER = auto()     # ⑤a format_planner
    STORAGE_ASSIGNER = auto()   # ⑤c storage_assigner
    BLOCK_PAD = auto()          # ⑤d block_pad
    ROOFLINE_ANALYZER = auto()  # ⑥b roofline_analyzer
    FUSION_PLANNER = auto()     # ⑥c fusion_planner
    GLOBAL_TILER = auto()       # ⑦b global_tiler


@dataclass
class PassConfig:
    """可选 Pass 开关配置，所有字段默认 True。"""

    absorption: bool = True
    mha_merge: bool = True
    format_planner: bool = True
    storage_assigner: bool = True
    block_pad: bool = True
    roofline_analyzer: bool = True
    fusion_planner: bool = True
    global_tiler: bool = True

    def is_enabled(self, pass_id: OptionalPass) -> bool:
        """检查指定 Pass 是否启用。"""
        return getattr(self, pass_id.name.lower())

    @classmethod
    def from_dict(cls, d: dict) -> PassConfig:
        """从字典构建，忽略未知字段，缺失字段用默认值。"""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def all_disabled(cls) -> PassConfig:
        """全部关闭（用于测试最小管线）。"""
        return cls(**{f.name: False for f in fields(cls)})

    @classmethod
    def all_enabled(cls) -> PassConfig:
        """全部启用（等价于默认构造）。"""
        return cls()
