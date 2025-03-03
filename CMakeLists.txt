# Description:   CMake file for Bachelor thesis
#                Real-time visualized and GPU-accelerated Lattice Boltzmann simulations
#
# Author:        Marcel Graf
#
# Copyright:     (C) 2024 Marcel Graf
#
# Last modified: December 27, 2024

cmake_minimum_required(VERSION 3.18.2)

project(parallel_lbm CXX)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)

  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose one of the following build types: Debug Release RelWithDebInfo MinSizeRel" FORCE)
  
endif(NOT CMAKE_BUILD_TYPE)

# Compiler options
#set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-Wall -Wextra -pedantic -ggdb3")

# Options
option(WITH_VISUALIZATION "Build with the visualization framework" OFF)
option(WITH_NAN_PROTECTION "Build with security against NANs" ON)
option(USE_FLOAT "If enabled, the project is build using float instead of double as the real type" OFF)

# Find crucial packages
find_package(HPX REQUIRED)
find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)

# Executable
add_executable(${PROJECT_NAME} src/main_gpu.cpp)

# Add source files and include directories
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} PUBLIC HPX::hpx HPX::wrap_main nlohmann_json::nlohmann_json fmt::fmt)

# If the option for live visualization is set, find the necessary packages
if(WITH_VISUALIZATION)

  add_definitions(-DWITH_VISUALIZATION)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC glfw OpenGL::GL)

endif(WITH_VISUALIZATION)

if(WITH_NAN_PROTECTION)
  add_definitions(-DWITH_NAN_PROTECTION)
endif(WITH_NAN_PROTECTION)

if(USE_FLOAT)
  add_definitions(-DUSE_FLOAT)
endif(USE_FLOAT)

# SYCL
target_compile_options(${PROJECT_NAME} PUBLIC ${SYCL_FLAGS})
add_sycl_to_target(TARGET ${PROJECT_NAME})
