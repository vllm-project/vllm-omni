# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(ulysses_symmetric_memory_fused)
set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu")
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  message(FATAL_ERROR
    "SOC_VERSION is required. Pass the detected Ascend target with "
    "-DSOC_VERSION=<device name>.")
endif()
set(SOC_VERSION "${SOC_VERSION}" CACHE STRING "Target Ascend SoC" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(ASCEND_PLATFORM "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(ASCEND_PLATFORM "x86_64-linux")
else()
  message(FATAL_ERROR "Unsupported host architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(ASCENDC_CMAKE "${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_PLATFORM}/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
if(NOT EXISTS "${ASCENDC_CMAKE}")
  message(FATAL_ERROR "AscendC CMake entrypoint not found: ${ASCENDC_CMAKE}")
endif()
include("${ASCENDC_CMAKE}")
if(NOT SHMEM_ROOT)
  message(FATAL_ERROR "Set SHMEM_ROOT to the installed CANN SHMEM package")
endif()
ascendc_library(ulysses_symmetric_memory_fused SHARED kernel.cpp)
ascendc_compile_definitions(ulysses_symmetric_memory_fused PRIVATE ASCENDC_DUMP=0)
ascendc_include_directories(ulysses_symmetric_memory_fused PRIVATE ${SHMEM_ROOT}/include ${SHMEM_ROOT}/include/device ${SHMEM_ROOT}/src/device ${ASCEND_CANN_PACKAGE_PATH}/include)
# CANN's packaging glob skips hidden components in the mirrored source path.
# Copy only in that case, avoiding duplicate objects for ordinary checkouts.
if(CMAKE_CURRENT_SOURCE_DIR MATCHES "/\\.[^/]+/")
add_custom_command(TARGET ulysses_symmetric_memory_fused PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_BINARY_DIR}/ulysses_symmetric_memory_fused_host_dir/objects-Release/host_bisheng_obj${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp.o"
    "${CMAKE_BINARY_DIR}/ulysses_symmetric_memory_fused_host_dir/proto_kernel.o")
endif()
