"""c_project — 比对、CMake 集成及辅助工具 C 代码生成。"""

from .cmake_emitter import emit_cmake
from .cmake_emitter import run as cmake_run
from .main_emitter import emit_main_c
from .main_emitter import run as main_run
from .mock_emitter import emit_mock_h
from .mock_emitter import run as mock_run
from .utils_emitter import run as utils_run

__all__ = [
    "cmake_emitter",
    "cmake_run",
    "emit_cmake",
    "emit_main_c",
    "emit_mock_h",
    "main_emitter",
    "main_run",
    "mock_emitter",
    "mock_run",
    "utils_emitter",
    "utils_run",
]
