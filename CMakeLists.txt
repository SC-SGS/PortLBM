# CMake file for PortLBM — GPU-accelerated Lattice Boltzmann simulations Copyright (c) Marcel Graf,
# Alexander Strack

cmake_minimum_required(VERSION 3.18.2)

project(
  PortLBM
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING
            "Choose one of the following build types: Debug Release RelWithDebInfo MinSizeRel"
            FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

# ---- Options ---------------------------------------------------------------

option(WITH_VISUALIZATION "Build with ImGui/ImPlot visualization" OFF)
option(WITH_NAN_PROTECTION "Guard against NaN/Inf in macroscopic variables" ON)
option(USE_FLOAT "Use float instead of double as real_type" OFF)
option(FORCE_USE_CPU "Force host execution even when a GPU is available" OFF)
option(BENCHMARK_MODE "Build the benchmark driver" OFF)
option(PORTLBM_BUILD_EXECUTABLE "Build the portlbm driver executable" ON)
option(PORTLBM_BUILD_TESTS "Build the Catch2 unit and integration tests" OFF)
option(PORTLBM_FORMAT_ONLY
       "Configure formatting targets only — skips all dependencies and compilation" OFF)

# ---- Formatting targets (configured first so FORMAT_ONLY can return early) --

find_program(CLANG_FORMAT_PROGRAM NAMES clang-format-18 clang-format)

if(CLANG_FORMAT_PROGRAM)
  file(GLOB_RECURSE FORMAT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/lbm/*.cpp"
       "${CMAKE_CURRENT_SOURCE_DIR}/include/lbm/*.hpp")
  list(FILTER FORMAT_SOURCES EXCLUDE REGEX ".*/external/.*")

  add_custom_target(
    check-clang-format
    COMMAND ${CLANG_FORMAT_PROGRAM} --dry-run --Werror ${FORMAT_SOURCES}
    COMMENT "Checking code formatting"
    VERBATIM)

  add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT_PROGRAM} -i ${FORMAT_SOURCES}
    COMMENT "Formatting code"
    VERBATIM)
else()
  message(STATUS "[PortLBM] clang-format not found — formatting targets unavailable")
endif()

find_program(CMAKE_FORMAT_PROGRAM NAMES cmake-format)

if(CMAKE_FORMAT_PROGRAM)
  file(
    GLOB_RECURSE
    CMAKE_FORMAT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/**/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")

  add_custom_target(
    check-cmake-format
    COMMAND ${CMAKE_FORMAT_PROGRAM} --check ${CMAKE_FORMAT_SOURCES}
    COMMENT "Checking CMake formatting"
    VERBATIM)

  add_custom_target(
    fix-cmake-format
    COMMAND ${CMAKE_FORMAT_PROGRAM} -i ${CMAKE_FORMAT_SOURCES}
    COMMENT "Fixing CMake formatting"
    VERBATIM)
else()
  message(STATUS "[PortLBM] cmake-format not found — CMake formatting targets unavailable")
endif()

if(PORTLBM_FORMAT_ONLY)
  return()
endif()

# ---- Dependencies ----------------------------------------------------------

include(FetchContent)

# nlohmann_json — fall back to FetchContent if not on system
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "[PortLBM] nlohmann_json not found — fetching via FetchContent")
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE)
  set(JSON_BuildTests
      OFF
      CACHE INTERNAL "")
  set(JSON_Install
      OFF
      CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
  set(PORTLBM_JSON_FETCHED TRUE)
endif()

# fmt — fall back to FetchContent if not on system
find_package(fmt QUIET)
if(NOT fmt_FOUND)
  message(STATUS "[PortLBM] fmt not found — fetching via FetchContent")
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
    GIT_SHALLOW TRUE)
  set(FMT_INSTALL
      OFF
      CACHE INTERNAL "")
  FetchContent_MakeAvailable(fmt)
  set(PORTLBM_FMT_FETCHED TRUE)
endif()

# Search _deps/acpp-install first (populated by install_adaptivecpp.sh), then fall back to whatever
# AdaptiveCpp_DIR or CMAKE_PREFIX_PATH provides.
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/_deps/acpp-install")
find_package(AdaptiveCpp CONFIG REQUIRED)

if(WITH_VISUALIZATION)
  find_package(OpenGL REQUIRED)
  # glfw3 — fall back to FetchContent if not on system
  find_package(glfw3 QUIET)
  if(NOT glfw3_FOUND)
    message(STATUS "[PortLBM] glfw3 not found — fetching via FetchContent")
    message(STATUS "[PortLBM] Note: building GLFW from source requires X11 dev headers.")
    message(
      STATUS
        "[PortLBM]   Ubuntu/Debian: sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev"
    )
    message(
      STATUS "[PortLBM]   If this fails, consider installing glfw via Spack: spack install glfw")
    FetchContent_Declare(
      glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.3.8
      GIT_SHALLOW TRUE)
    set(GLFW_BUILD_DOCS
        OFF
        CACHE INTERNAL "")
    set(GLFW_BUILD_TESTS
        OFF
        CACHE INTERNAL "")
    set(GLFW_BUILD_EXAMPLES
        OFF
        CACHE INTERNAL "")
    set(GLFW_INSTALL
        OFF
        CACHE INTERNAL "")
    set(GLFW_BUILD_WAYLAND
        OFF
        CACHE INTERNAL "")
    FetchContent_MakeAvailable(glfw)
    set(PORTLBM_GLFW_FETCHED TRUE)
  endif()
endif()

# ---- Library target --------------------------------------------------------

add_library(portlbm_core STATIC)
add_library(PortLBM::portlbm_core ALIAS portlbm_core)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(
  portlbm_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/lbm> $<INSTALL_INTERFACE:include>)

# Fetched targets are not IMPORTED, so CMake cannot serialize them into the install export. Wrap
# them in BUILD_INTERFACE; PortLBMConfig.cmake.in's find_dependency() covers consumers of the
# installed library.
if(PORTLBM_JSON_FETCHED)
  target_link_libraries(portlbm_core PUBLIC $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
else()
  target_link_libraries(portlbm_core PUBLIC nlohmann_json::nlohmann_json)
endif()

if(PORTLBM_FMT_FETCHED)
  target_link_libraries(portlbm_core PUBLIC $<BUILD_INTERFACE:fmt::fmt>)
else()
  target_link_libraries(portlbm_core PUBLIC fmt::fmt)
endif()

# PUBLIC so header-only GPU kernel templates and real_type see the same flags. USE_FLOAT changes
# real_type and therefore the ABI — PUBLIC is intentional.
target_compile_definitions(
  portlbm_core
  PUBLIC $<$<BOOL:${WITH_NAN_PROTECTION}>:WITH_NAN_PROTECTION> $<$<BOOL:${USE_FLOAT}>:USE_FLOAT>
         $<$<BOOL:${FORCE_USE_CPU}>:FORCE_USE_CPU>
         $<$<BOOL:${WITH_VISUALIZATION}>:WITH_VISUALIZATION>)

if(WITH_VISUALIZATION)
  if(PORTLBM_GLFW_FETCHED)
    target_link_libraries(portlbm_core PUBLIC $<BUILD_INTERFACE:glfw> OpenGL::GL)
  else()
    target_link_libraries(portlbm_core PUBLIC glfw OpenGL::GL)
  endif()
endif()

# PUBLIC so SYCL compile options propagate to the executable's kernel templates.
target_compile_options(portlbm_core PUBLIC ${SYCL_FLAGS})
add_sycl_to_target(TARGET portlbm_core)

# ---- Version header --------------------------------------------------------

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/lbm/version.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/lbm/version.hpp" @ONLY)

# ---- Executable target -----------------------------------------------------

if(PORTLBM_BUILD_EXECUTABLE)
  add_executable(portlbm src/main_gpu.cpp)
  target_link_libraries(portlbm PRIVATE portlbm_core)

  # benchmark.cpp is application-level; not compiled into the library.
  if(BENCHMARK_MODE)
    target_sources(portlbm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/lbm/benchmark/benchmark.cpp")
    target_compile_definitions(portlbm PRIVATE BENCHMARK_MODE)
  endif()

  add_sycl_to_target(TARGET portlbm)
endif()

# ---- Tests -----------------------------------------------------------------

if(PORTLBM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Install and export ----------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
  TARGETS portlbm_core
  EXPORT PortLBMTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Exclude vendored external/ headers (ImGui, ImPlot) from the install tree.
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/lbm"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
  PATTERN "external" EXCLUDE)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/lbm/version.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lbm")

install(
  EXPORT PortLBMTargets
  FILE PortLBMTargets.cmake
  NAMESPACE PortLBM::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PortLBM")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PortLBMConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PortLBM")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/PortLBMConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PortLBM")
