"""cmake_emitter — 生成 CMakeLists.txt。"""

from __future__ import annotations

import os

from torch2c.common import get_logger

logger = get_logger("codegen.cmake_emitter")

_CMAKE_TEMPLATE = """\
cmake_minimum_required(VERSION 3.10)
project(npu_model C)

set(CMAKE_C_STANDARD 99)
add_compile_options(-Wall -Wextra)

# NPU SDK: 真机用 NPU_SDK_PATH，本地验证用内置 npu_cpu_mock
set(NPU_SDK_PATH "" CACHE PATH "Path to NPU SDK (leave empty for CPU mock)")
if(NPU_SDK_PATH)
    set(NPU_INCLUDE_DIR ${NPU_SDK_PATH}/include)
    link_directories(${NPU_SDK_PATH}/lib)
    set(NPU_MOCK_SOURCES "")
else()
    set(NPU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/npu_cpu_mock/include)
    file(GLOB NPU_MOCK_SOURCES npu_cpu_mock/src/*.c)
endif()

option(NPU_DEBUG_DUMP "Enable debug dump" OFF)
if(NPU_DEBUG_DUMP)
    add_definitions(-DNPU_DEBUG_DUMP)
endif()
set(NPU_DEBUG_LEVEL "0" CACHE STRING "NPU debug level (0=off)")
add_definitions(-DNPU_DEBUG_LEVEL=${NPU_DEBUG_LEVEL})

add_executable(npu_model_run
    main.c src/model_graph.c
    utils/data_loader.c utils/data_dumper.c utils/comparator.c
    ${NPU_MOCK_SOURCES})
target_include_directories(npu_model_run PRIVATE
    ${NPU_INCLUDE_DIR} src utils)
if(NPU_SDK_PATH)
    target_link_libraries(npu_model_run npu_runtime)
endif()
target_link_libraries(npu_model_run m)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_executable(test_data_loader tests/test_data_loader.c utils/data_loader.c)
    target_include_directories(test_data_loader PRIVATE utils)
    add_test(NAME test_data_loader COMMAND test_data_loader)

    add_executable(test_comparator tests/test_comparator.c utils/comparator.c utils/data_loader.c)
    target_include_directories(test_comparator PRIVATE utils)
    target_link_libraries(test_comparator m)
    add_test(NAME test_comparator COMMAND test_comparator)

    add_executable(test_memory_layout tests/test_memory_layout.c)
    target_include_directories(test_memory_layout PRIVATE src)
    add_test(NAME test_memory_layout COMMAND test_memory_layout)
endif()
"""


def emit_cmake() -> str:
    return _CMAKE_TEMPLATE


def run(output_dir: str) -> None:
    logger.info("cmake_emitter: 开始生成 CMakeLists.txt")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "CMakeLists.txt")
    with open(path, "w") as f:
        f.write(emit_cmake())
    logger.info("cmake_emitter: 已生成 %s", path)
