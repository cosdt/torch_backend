cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_BACKEND CXX C)

# ---[ Include.
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)
# TODO(FFFrog):
# It will be removed when the decoupling is complete
include(cmake/CXXCompilerFlag.cmake)

# ---[ Options.
# Note to developers: if you add an option below, make sure you also add it to
# cmake/Summary.cmake so that the summary prints out the option values.
option(BUILD_TEST "Build C++ test binaries (need gtest)" OFF)
cmake_dependent_option(INSTALL_TEST "Install C++ test binaries if BUILD_TEST is on"
        ON "BUILD_TEST" OFF)

# ---[ Build Directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---[ RPATH.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib/:$ORIGIN/")

# ---[ Default Variables.
set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_LIBDIR lib)

# ---[ Find Packages
set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)

# --[ CXX Flags.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEVICE_NAMESPACE=acl_adapter")

# ---[ Global Include Directories
if(DEFINED PYTORCH_INSTALL_DIR)
  message(STATUS "PYTORCH_INSTALL_DIR: ${PYTORCH_INSTALL_DIR}")
  include_directories(${PYTORCH_INSTALL_DIR}/include)
else()
  message(FATAL_ERROR "Can not find PyTorch directory")
endif()

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

# ---[ Variables.
set(BACKEND_LIB torch_backend)
set(PYTHON_BACKEND_LIB torch_backend_python)

include_directories(${PROJECT_SOURCE_DIR})
# TODO(FFFrog):
# It will be removed when the decoupling is complete
include_directories(${PROJECT_SOURCE_DIR}/backends/npu/acl/include)
include_directories(${PROJECT_SOURCE_DIR}/backends/npu)

link_directories(${PYTORCH_INSTALL_DIR}/lib)
# TODO(FFFrog):
# It will be removed when the decoupling is complete
link_directories(${PROJECT_SOURCE_DIR}/npu/acl/libs)

set(PYTHON_SRCS)
set(BACKEND_SRCS)

add_subdirectory(${PROJECT_SOURCE_DIR}/torch_backend/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
# TODO(FFFrog):
# Fake and npu will be compiled into a single lib when the decoupling is complete
add_subdirectory(${PROJECT_SOURCE_DIR}/backends/fake)
add_subdirectory(${PROJECT_SOURCE_DIR}/backends/npu)

# add torch_backend_python lib
add_library(${PYTHON_BACKEND_LIB} SHARED ${PYTHON_SRCS})
target_link_libraries(${PYTHON_BACKEND_LIB} PRIVATE torch_python torch_backend)
install(TARGETS ${PYTHON_BACKEND_LIB} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# add torch_backend lib
add_library(${BACKEND_LIB} SHARED ${BACKEND_SRCS})
# TODO(FFFrog):
# It will be removed when the decoupling is complete
target_link_libraries(${BACKEND_LIB} PUBLIC hccl ascendcl acl_op_compiler ge_runner graph)
target_link_libraries(${BACKEND_LIB} PUBLIC torch torch_cpu c10)
target_compile_options(${BACKEND_LIB} PRIVATE "-DC10_BUILD_MAIN_LIB")

set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if ("${ATEN_THREADING}" STREQUAL "OMP")
  target_compile_definitions(${BACKEND_LIB} PUBLIC "-DAT_PARALLEL_OPENMP=1")
elseif ("${ATEN_THREADING}" STREQUAL "NATIVE")
  target_compile_definitions(${BACKEND_LIB} PUBLIC "-DAT_PARALLEL_NATIVE=1")
elseif ("${ATEN_THREADING}" STREQUAL "TBB")
  target_compile_definitions(${BACKEND_LIB} PUBLIC "-DAT_PARALLEL_NATIVE_TBB=1")
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()
install(TARGETS ${BACKEND_LIB} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

include(cmake/Dependencies.cmake)
