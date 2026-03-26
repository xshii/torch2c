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
    src_preference: 输入端口的格式偏好顺序（保留 YAML 列表顺序）。
        YAML 中 ``src0: [zz, nd]`` 表示端口 0 优先接受 zz。
        标量 ``src1: nz`` 等价于只有一个元素的列表。
    dst_preference: 输出格式的偏好顺序（保留 YAML dst 列表顺序）。
        YAML 中 ``dst: [zz, nz, nd]`` 表示 producer 优先输出 zz。
    """

    src_ports: dict[int | None, frozenset[str]]
    dst: frozenset[str]
    src_preference: dict[int | None, tuple[str, ...]] = None  # type: ignore[assignment]
    dst_preference: tuple[str, ...] = None  # type: ignore[assignment]

    def __post_init__(self):
        # frozen=True 下用 object.__setattr__ 初始化默认值
        if self.src_preference is None:
            object.__setattr__(self, "src_preference", {})
        if self.dst_preference is None:
            object.__setattr__(self, "dst_preference", ())

    def get_src_formats(self, position: int) -> frozenset[str]:
        """获取指定输入位置的允许格式集合。"""
        # 优先按位置查找，否则用统一要求
        if position in self.src_ports:
            return self.src_ports[position]
        if None in self.src_ports:
            return self.src_ports[None]
        # 没有约束 → 任意格式
        return frozenset({"nd", "nz"})

    def get_src_preferred(self, position: int) -> str | None:
        """获取指定输入位置的首选格式（YAML 列表中的第一个）。

        返回 None 表示无偏好（单一格式或未配置）。
        """
        pref = self.src_preference
        if not pref:
            return None
        # 优先按位置查找，否则用统一要求
        order = pref.get(position) or pref.get(None)
        if order and len(order) >= 2:
            return order[0]  # 列表第一个 = 最优先
        return None

    def get_dst_preferred(self) -> str | None:
        """获取 producer 首选的输出格式（YAML dst 列表中的第一个）。

        返回 None 表示无偏好（单一格式或未配置）。
        """
        if self.dst_preference and len(self.dst_preference) >= 2:
            return self.dst_preference[0]
        return None


def _parse_formats(value: str | list[str]) -> frozenset[str]:
    """将 YAML 值（标量或列表）解析为 frozenset。"""
    if isinstance(value, list):
        return frozenset(value)
    return frozenset({value})


def _parse_order(value: str | list[str]) -> tuple[str, ...]:
    """将 YAML 值解析为顺序 tuple（保留列表顺序作为偏好）。"""
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _parse_one_capability(spec: dict) -> UnitCapability:
    """将单个能力 spec dict 解析为 UnitCapability。"""
    src_ports: dict[int | None, frozenset[str]] = {}
    src_preference: dict[int | None, tuple[str, ...]] = {}
    dst = frozenset({"nd"})  # 默认
    dst_preference: tuple[str, ...] = ()

    for key, value in spec.items():
        if key == "dst":
            dst = _parse_formats(value)
            dst_preference = _parse_order(value)
        elif key == "src":
            src_ports[None] = _parse_formats(value)
            src_preference[None] = _parse_order(value)
        elif key.startswith("src"):
            idx = int(key[3:])
            src_ports[idx] = _parse_formats(value)
            src_preference[idx] = _parse_order(value)

    return UnitCapability(
        src_ports=src_ports, dst=dst,
        src_preference=src_preference, dst_preference=dst_preference,
    )


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
