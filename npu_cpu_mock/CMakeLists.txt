cmake_minimum_required(VERSION 3.14)
project(npu_cpu_mock C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Library
add_library(npu_cpu_mock STATIC
    src/npu_dtype_utils.c
    src/npu_compute_elementwise.c
    src/npu_compute_matmul.c
    src/npu_compute_norm.c
    src/npu_compute_rmsnorm.c
    src/npu_compute_softmax.c
    src/npu_compute_transpose.c
    src/npu_dma.c
    src/train/train_vector_loss.c
    src/train/train_vector_backward.c
    src/train/train_vector_optimizer.c
)
target_include_directories(npu_cpu_mock PUBLIC include)
target_link_libraries(npu_cpu_mock PRIVATE m)

# Tests
enable_testing()
add_subdirectory(tests)
