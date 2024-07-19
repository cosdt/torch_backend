# ---[ Googletest
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  include_directories(
    BEFORE SYSTEM
    ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
  include_directories(
    BEFORE SYSTEM
    ${PROJECT_SOURCE_DIR}/third_party/googletest/googlemock/include)

  # test/cpp/core
  add_subdirectory(${PROJECT_SOURCE_DIR}/test/cpp ${CMAKE_BINARY_DIR}/test_core)
endif()
