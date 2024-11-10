cmake_minimum_required(VERSION 3.18.2)
project(parallel_lbm CXX)

set(CMAKE_CXX_STANDARD 20)

option(WITH_VISUALIZATION "Build with the visualization framework" ON)

find_package(HPX REQUIRED)
find_package(AdaptiveCpp CONFIG REQUIRED)

find_package(nlohmann_json 3.11.2 REQUIRED)
find_package(fmt REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

file(GLOB_RECURSE source_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE header_files "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

add_executable(gpu_lattice_boltzmann main_gpu.cpp ${source_files} ${header_files})
target_compile_options(gpu_lattice_boltzmann PUBLIC ${SYCL_FLAGS})
target_link_libraries(gpu_lattice_boltzmann PUBLIC HPX::hpx HPX::wrap_main nlohmann_json::nlohmann_json fmt::fmt glfw OpenGL::GL)
add_sycl_to_target(TARGET gpu_lattice_boltzmann SOURCES ${source_files} ${header_files})