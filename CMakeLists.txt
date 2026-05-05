# Description:    CMake file for PortLBM
#                 Real-time visualized and GPU-accelerated Lattice Boltzmann simulations
#
# Author:         Marcel Graf
#
# Copyright:      (c) Marcel Graf
#
# Phase-1 library refactor applied: May 2025

cmake_minimum_required(VERSION 3.18.2)

project(PortLBM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose one of the following build types: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

# ---- Options ---------------------------------------------------------------

option(WITH_VISUALIZATION   "Build with the ImGui/ImPlot visualization framework"    OFF)
option(WITH_NAN_PROTECTION  "Enable guards against NaN/Inf in macroscopic variables" ON)
option(USE_FLOAT            "Use float instead of double as real_type"               OFF)
option(FORCE_USE_CPU        "Force host execution even when a GPU is available"      OFF)
option(BENCHMARK_MODE       "Build the benchmark driver"                             OFF)
option(PORTLBM_BUILD_EXECUTABLE "Build the parallel_lbm driver executable"          ON)

# ---- Dependencies ----------------------------------------------------------

find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)

if(WITH_VISUALIZATION)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
endif()

# ---- Library target: portlbm_core ------------------------------------------

add_library(portlbm_core STATIC)
add_library(PortLBM::portlbm_core ALIAS portlbm_core)

# Source files and internal include dirs are set in the subdirectories
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/include")

# Public include root: consumers do  #include <lbm/portlbm.hpp>  etc.
target_include_directories(portlbm_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(portlbm_core
  PUBLIC
    nlohmann_json::nlohmann_json
    fmt::fmt
)

# Compile definitions — PUBLIC so that header-only templates (GPU kernels,
# real_type, device-selector) pick up the same flags as the library .cpp files.
# USE_FLOAT changes real_type in the public header constants.hpp and therefore
# affects the ABI; propagating it via PUBLIC is intentional.
target_compile_definitions(portlbm_core
  PUBLIC
    $<$<BOOL:${WITH_NAN_PROTECTION}>:WITH_NAN_PROTECTION>
    $<$<BOOL:${USE_FLOAT}>:USE_FLOAT>
    $<$<BOOL:${FORCE_USE_CPU}>:FORCE_USE_CPU>
    $<$<BOOL:${WITH_VISUALIZATION}>:WITH_VISUALIZATION>
)

if(WITH_VISUALIZATION)
  target_link_libraries(portlbm_core PUBLIC glfw OpenGL::GL)
endif()

# SYCL: PUBLIC so compile options propagate to the executable that instantiates
# the header-only kernel templates.
target_compile_options(portlbm_core PUBLIC ${SYCL_FLAGS})
add_sycl_to_target(TARGET portlbm_core)

# ---- Version header (1.5) --------------------------------------------------

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lbm/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/lbm/version.hpp"
  @ONLY
)

# ---- Executable target (1.1, 1.7) ------------------------------------------

if(PORTLBM_BUILD_EXECUTABLE)
  add_executable(parallel_lbm src/main_gpu.cpp)
  target_link_libraries(parallel_lbm PRIVATE portlbm_core)

  # benchmark.cpp is application-level code; it is NOT compiled into the library.
  # It is added only to the driver executable when BENCHMARK_MODE is requested.
  if(BENCHMARK_MODE)
    target_sources(parallel_lbm PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/lbm/benchmark/benchmark.cpp"
    )
    target_compile_definitions(parallel_lbm PRIVATE BENCHMARK_MODE)
  endif()

  # The executable instantiates header-only SYCL kernel templates.
  add_sycl_to_target(TARGET parallel_lbm)
endif()

# ---- Install and export (1.4) -----------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS portlbm_core
  EXPORT  PortLBMTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Install public headers; exclude vendored external/ sources (ImGui, ImPlot)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/lbm"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
  PATTERN "external" EXCLUDE
)

# Install the generated version header
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/lbm/version.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lbm"
)

install(EXPORT PortLBMTargets
  FILE      PortLBMTargets.cmake
  NAMESPACE PortLBM::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PortLBM"
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PortLBMConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PortLBM"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfigVersion.cmake"
  VERSION     "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PortLBM"
)
