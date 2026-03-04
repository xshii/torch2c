"""common — 基础设施模块。"""

from .config_loader import load_config
from .dtypes import DTYPE_INFO, DtypeInfo, dtype_bytes, dtype_c_enum
from .errors import (
    AbsorptionError,
    CodegenError,
    CompileDiagnostic,
    CompilerError,
    ConfigError,
    DecompositionError,
    DiagnosticCollector,
    MappingError,
    MemoryPlanError,
    Severity,
    ValidationError,
)
from .graph_ir import Graph, Node, Tensor
from .logger import get_logger, setup_logging
from .pass_protocol import CompilerPass

__all__ = [
    "Graph",
    "Node",
    "Tensor",
    "get_logger",
    "setup_logging",
    "load_config",
    "CompilerError",
    "ConfigError",
    "MappingError",
    "DecompositionError",
    "AbsorptionError",
    "ValidationError",
    "MemoryPlanError",
    "CodegenError",
    "Severity",
    "CompileDiagnostic",
    "DiagnosticCollector",
    "DtypeInfo",
    "DTYPE_INFO",
    "dtype_bytes",
    "dtype_c_enum",
    "CompilerPass",
]
