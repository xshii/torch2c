set(TRAIN_TEST_SOURCES
    test_train_loss
    test_train_backward
    test_train_optimizer
)

foreach(t ${TRAIN_TEST_SOURCES})
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE npu_cpu_mock m)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${t} COMMAND ${t})
endforeach()
