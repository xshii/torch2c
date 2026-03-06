"""codegen — Pass⑨：C代码生成。

子模块：
  c_emitter         — 图执行逻辑 model_graph.c/h
  weight_exporter   — 权重导出 model_weights.h
  golden_exporter   — golden 数据导出
  c_project/        — 比对、CMake、mock、辅助工具 C 代码生成
"""

from . import c_emitter, golden_exporter, weight_exporter
from .c_project import cmake_emitter, mock_emitter, utils_emitter

__all__ = [
    "c_emitter",
    "cmake_emitter",
    "golden_exporter",
    "mock_emitter",
    "utils_emitter",
    "weight_exporter",
]
