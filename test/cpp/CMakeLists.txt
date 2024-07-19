if(BUILD_TEST)
  file(GLOB _TEST_SRCS core/*.cpp npu/*.cpp)
  message(STATUS "Build test files: " ${_TEST_SRCS})

  set(TORCH_BACKEND_CORE_TEST_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/common/main.cpp ${_TEST_SRCS})

  add_executable(test_core ${TORCH_BACKEND_CORE_TEST_SOURCES})
  target_link_libraries(test_core PRIVATE torch_npu gtest_main gtest)
  add_test(NAME test_core COMMAND $<TARGET_FILE:test_core>)
endif()

if(INSTALL_TEST)
  install(TARGETS test_core DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC)
    install(
      FILES $<TARGET_PDB_FILE:test_core>
      DESTINATION bin
      OPTIONAL)
  endif()
endif()
