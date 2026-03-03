"""c_project — 比对、CMake 集成及辅助工具 C 代码生成。"""

from .cmake_emitter import emit_cmake, run as cmake_run
from .main_emitter import emit_main_c, run as main_run
from .mock_emitter import emit_mock_h, run as mock_run
from .utils_emitter import run as utils_run

__all__ = [
    "cmake_emitter",
    "main_emitter",
    "mock_emitter",
    "utils_emitter",
    "emit_cmake",
    "emit_main_c",
    "emit_mock_h",
    "cmake_run",
    "main_run",
    "mock_run",
    "utils_run",
]
