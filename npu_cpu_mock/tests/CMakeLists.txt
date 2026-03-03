set(TEST_SOURCES
    test_fp16
    test_elementwise
    test_matmul
    test_norm
    test_softmax
    test_transpose
)

foreach(t ${TEST_SOURCES})
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE npu_cpu_mock m)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${t} COMMAND ${t})
endforeach()
