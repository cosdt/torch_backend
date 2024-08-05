if(BUILD_TEST)
  set(TORCH_BACKEND_NPU_TEST_SOURCES
    ${PROJECT_SOURCE_DIR}/test/cpp/common/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_generator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_context_test.cpp)

  add_executable(test_npu ${TORCH_BACKEND_NPU_TEST_SOURCES})
  target_link_libraries(test_npu PRIVATE torch_npu gtest_main gtest)
  add_test(NAME test_npu COMMAND $<TARGET_FILE:test_npu>)
endif()

if(INSTALL_TEST)
  install(TARGETS test_npu DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC)
    install(FILES $<TARGET_PDB_FILE:test_npu> DESTINATION bin OPTIONAL)
  endif()
endif()