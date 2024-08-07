if(BUILD_TEST)
  set(TORCH_BACKEND_TEST_SOURCES
    ${PROJECT_SOURCE_DIR}/test/cpp/common/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context_test.cpp)

  add_executable(test_backend ${TORCH_BACKEND_TEST_SOURCES})
  target_link_libraries(test_backend PRIVATE torch_backend gtest_main gtest)
  add_test(NAME test_backend COMMAND $<TARGET_FILE:test_backend>)
endif()

if(INSTALL_TEST)
  install(TARGETS test_backend DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC)
    install(FILES $<TARGET_PDB_FILE:test_backend> DESTINATION bin OPTIONAL)
  endif()
endif()
