"""format_capabilities 配置解析 — 将 YAML 配置转为 UnitCapability 数据结构。

YAML 格式示例：
    format_capabilities:
      cube:
        src0: nd          # 按位置：第 0 个非 absorbed 输入
        src1: nz          # 按位置：第 1 个非 absorbed 输入
        dst: [nd, nz]     # 输出可选
      vector:
        src: nd            # 无编号 = 所有输入统一要求
        dst: [nd, nz]
      by_op:              # 算子级别覆盖（优先级高于计算单元级别）
        vector_ln:
          src: [nd, nz]
          dst: nd

规则：
  - 带编号 src0/src1：按 inputs 位置匹配（跳过 absorbed）
  - 无编号 src：所有输入统一要求
  - 列表 [nd, nz]：多格式可选
  - 标量 nd：唯一要求
  - by_op 中的算子规则优先级高于计算单元级别规则
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitCapability:
    """计算单元的格式能力描述。

    src_ports: 输入端口的格式要求。
        key=None 表示所有输入统一要求（无编号 src）。
        key=0/1/... 表示按位置匹配（src0/src1/...）。
    dst: 输出支持的格式集合。
    """

    src_ports: dict[int | None, frozenset[str]]
    dst: frozenset[str]

    def get_src_formats(self, position: int) -> frozenset[str]:
        """获取指定输入位置的允许格式集合。"""
        # 优先按位置查找，否则用统一要求
        if position in self.src_ports:
            return self.src_ports[position]
        if None in self.src_ports:
            return self.src_ports[None]
        # 没有约束 → 任意格式
        return frozenset({"nd", "nz"})


def _parse_formats(value: str | list[str]) -> frozenset[str]:
    """将 YAML 值（标量或列表）解析为 frozenset。"""
    if isinstance(value, list):
        return frozenset(value)
    return frozenset({value})


def _parse_one_capability(spec: dict) -> UnitCapability:
    """将单个能力 spec dict 解析为 UnitCapability。"""
    src_ports: dict[int | None, frozenset[str]] = {}
    dst = frozenset({"nd"})  # 默认

    for key, value in spec.items():
        if key == "dst":
            dst = _parse_formats(value)
        elif key == "src":
            src_ports[None] = _parse_formats(value)
        elif key.startswith("src"):
            idx = int(key[3:])
            src_ports[idx] = _parse_formats(value)

    return UnitCapability(src_ports=src_ports, dst=dst)


@dataclass(frozen=True)
class FormatCapabilities:
    """格式能力查找表，支持算子级别覆盖计算单元级别规则。

    查找优先级：by_op[npu_op] > by_unit[compute_unit]。
    """

    by_unit: dict[str, UnitCapability]
    by_op: dict[str, UnitCapability]

    # 未配置的计算单元/算子默认无约束（任意 src/dst）
    _DEFAULT = UnitCapability(
        src_ports={None: frozenset({"nd", "nz"})},
        dst=frozenset({"nd", "nz"}),
    )

    def get(self, compute_unit: str, npu_op: str | None = None) -> UnitCapability:
        """按优先级查找：算子级别 > 计算单元级别 > 默认无约束。"""
        if npu_op and npu_op in self.by_op:
            return self.by_op[npu_op]
        return self.by_unit.get(compute_unit, self._DEFAULT)


def parse_capabilities(raw: dict) -> FormatCapabilities:
    """解析 format_capabilities 配置块。

    Args:
        raw: hardware_config.yaml 中 format_capabilities 字段的值。

    Returns:
        FormatCapabilities 查找表。
    """
    by_unit: dict[str, UnitCapability] = {}
    by_op: dict[str, UnitCapability] = {}

    for name, spec in raw.items():
        if name == "by_op":
            # 算子级别覆盖
            for op_name, op_spec in spec.items():
                by_op[op_name] = _parse_one_capability(op_spec)
        else:
            # 计算单元级别
            by_unit[name] = _parse_one_capability(spec)

    return FormatCapabilities(by_unit=by_unit, by_op=by_op)
