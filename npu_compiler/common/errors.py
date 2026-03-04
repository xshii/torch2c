"""统一异常定义与诊断收集。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """诊断严重级别。"""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class CompileDiagnostic:
    """单条编译诊断信息。"""

    phase: str
    severity: Severity
    message: str


class DiagnosticCollector:
    """编译诊断收集器，汇总各 Pass 的 warning/error。"""

    def __init__(self) -> None:
        self._items: list[CompileDiagnostic] = []

    def warn(self, phase: str, message: str) -> None:
        self._items.append(CompileDiagnostic(phase, Severity.WARNING, message))

    def error(self, phase: str, message: str) -> None:
        self._items.append(CompileDiagnostic(phase, Severity.ERROR, message))

    def has_errors(self) -> bool:
        return any(d.severity == Severity.ERROR for d in self._items)

    @property
    def items(self) -> list[CompileDiagnostic]:
        return list(self._items)

    def summary(self) -> str:
        warnings = sum(1 for d in self._items if d.severity == Severity.WARNING)
        errors = sum(1 for d in self._items if d.severity == Severity.ERROR)
        return f"诊断汇总: {errors} error(s), {warnings} warning(s)"


class CompilerError(Exception):
    """编译器基类异常。"""


class ConfigError(CompilerError):
    """配置文件错误。"""


class MappingError(CompilerError):
    """算子映射失败。"""


class DecompositionError(CompilerError):
    """算子裂解失败。"""


class AbsorptionError(CompilerError):
    """参数吸收失败。"""


class ValidationError(CompilerError):
    """合法性校验失败。"""


class MemoryPlanError(CompilerError):
    """内存编排失败。"""


class CodegenError(CompilerError):
    """代码生成失败。"""
