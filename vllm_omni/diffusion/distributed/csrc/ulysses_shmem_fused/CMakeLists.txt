# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(ulysses_shmem_fused)
set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "Target Ascend SoC")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
if(NOT SHMEM_ROOT)
  message(FATAL_ERROR "Set SHMEM_ROOT to the installed CANN SHMEM package")
endif()
ascendc_library(ulysses_shmem_fused SHARED kernel.cpp)
ascendc_compile_definitions(ulysses_shmem_fused PRIVATE ASCENDC_DUMP=0)
ascendc_include_directories(ulysses_shmem_fused PRIVATE ${SHMEM_ROOT}/include ${SHMEM_ROOT}/include/device ${SHMEM_ROOT}/src/device ${ASCEND_CANN_PACKAGE_PATH}/include)
# CANN's packaging glob skips hidden components in the mirrored source path.
# Copy only in that case, avoiding duplicate objects for ordinary checkouts.
if(CMAKE_CURRENT_SOURCE_DIR MATCHES "/\\.[^/]+/")
add_custom_command(TARGET ulysses_shmem_fused PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_BINARY_DIR}/ulysses_shmem_fused_host_dir/objects-Release/host_bisheng_obj${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp.o"
    "${CMAKE_BINARY_DIR}/ulysses_shmem_fused_host_dir/proto_kernel.o")
endif()
