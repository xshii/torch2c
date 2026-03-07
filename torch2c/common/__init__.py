"""common — 基础设施模块。"""

from .config_loader import load_config
from .dtypes import DTYPE_INFO, DTYPE_NUMPY, DtypeInfo, dtype_bytes, dtype_c_enum, dtype_numpy
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
from .compile_config import get_model_config, torch2c_config
from .npu_annotate import (
    NpuSpec,
    format_annotation,
    format_model_annotations,
    get_input_annotation,
    get_npu_annotations,
    npu,
    npu_input,
)
from .pass_protocol import CompilerPass
from .paths import (
    DEFAULT_OUTPUT_DIR,
    HARDWARE_CONFIG_PATH,
    INTEGRATION_CONFIG_DIR,
    NPU_CPU_MOCK_DIR,
    PACKAGE_ROOT,
    PROJECT_ROOT,
)

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DTYPE_INFO",
    "DTYPE_NUMPY",
    "AbsorptionError",
    "CodegenError",
    "CompileDiagnostic",
    "CompilerError",
    "CompilerPass",
    "ConfigError",
    "DecompositionError",
    "DiagnosticCollector",
    "DtypeInfo",
    "Graph",
    "HARDWARE_CONFIG_PATH",
    "INTEGRATION_CONFIG_DIR",
    "MappingError",
    "MemoryPlanError",
    "NPU_CPU_MOCK_DIR",
    "Node",
    "NpuSpec",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "Severity",
    "Tensor",
    "ValidationError",
    "get_input_annotation",
    "get_model_config",
    "get_npu_annotations",
    "npu",
    "npu_input",
    "torch2c_config",
    "format_annotation",
    "format_model_annotations",
    "dtype_bytes",
    "dtype_c_enum",
    "dtype_numpy",
    "get_logger",
    "load_config",
    "setup_logging",
]
