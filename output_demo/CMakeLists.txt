cmake_minimum_required(VERSION 3.10)
project(npu_model C)

set(NPU_SDK_PATH "" CACHE PATH "Path to NPU SDK")
if(NOT NPU_SDK_PATH)
    set(NPU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    message(WARNING "Using mock headers (compile-only)")
else()
    set(NPU_INCLUDE_DIR ${NPU_SDK_PATH}/include)
    link_directories(${NPU_SDK_PATH}/lib)
endif()

set(CMAKE_C_STANDARD 99)
add_compile_options(-Wall -Wextra)

option(NPU_DEBUG_DUMP "Enable debug dump" OFF)
if(NPU_DEBUG_DUMP)
    add_definitions(-DNPU_DEBUG_DUMP)
endif()

add_executable(npu_model_run
    main.c src/model_graph.c
    utils/data_loader.c utils/data_dumper.c utils/comparator.c)
target_include_directories(npu_model_run PRIVATE
    ${NPU_INCLUDE_DIR} src utils)
if(NPU_SDK_PATH)
    target_link_libraries(npu_model_run npu_runtime)
endif()
target_link_libraries(npu_model_run m)

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
