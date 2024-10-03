cmake_minimum_required(VERSION 3.18.2)
project(parallel_lbm CXX)
find_package(HPX REQUIRED)

set(HEADER_FILES ### General
                 include/access.hpp
                 include/boundaries.hpp 
                 include/collision.hpp
                 include/defines.hpp
                 include/file_interaction.hpp
                 include/lbm_execution.hpp
                 include/macroscopic.hpp
                 include/utils.hpp
                 ### Sequential implementations
                 include/simulation.hpp
                 # include/sequential_swap.hpp
                 # include/sequential_shift.hpp
                 include/sequential_two_lattice.hpp
                 # include/sequential_two_step.hpp
                 ### Parallel implementations
                 include/parallel_framework.hpp
                 include/parallel_two_lattice.hpp
                 # include/parallel_two_lattice_framework.hpp
                 # include/parallel_two_step_framework.hpp
                 # include/parallel_swap_framework.hpp
                 # include/parallel_shift_framework.hpp
                 ### GPU
                 include/gpu_two_lattice.hpp
                 )

set(SOURCE_FILES ###General
                 src/access.cpp
                 src/boundaries.cpp 
                 src/collision.cpp
                 src/defines.cpp
                 src/file_interaction.cpp
                 src/lbm_execution.cpp
                 src/macroscopic.cpp
                 ### Sequential implementations
                 src/simulation.cpp
                 # src/sequential_shift.cpp
                 # src/sequential_swap.cpp
                 src/sequential_two_lattice.cpp
                 # src/sequential_two_step.cpp
                 ### Parallel implementations
                 src/parallel_framework.cpp
                 src/parallel_two_lattice.cpp
                 # src/parallel_two_lattice_framework.cpp
                 # src/parallel_two_step_framework.cpp
                 # src/parallel_swap_framework.cpp
                 # src/parallel_shift_framework.cpp
                 ### GPU
                 #src/gpu_two_lattice.cpp
                 )

# add_executable(benchmark main_benchmark.cpp ${SOURCE_FILES})
# target_link_libraries(benchmark HPX::hpx HPX::wrap_main)

# add_executable(lattice_boltzmann main_global.cpp ${SOURCE_FILES})
# target_link_libraries(lattice_boltzmann HPX::hpx HPX::wrap_main)

add_executable(gpu_lattice_boltzmann main_gpu.cpp ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(gpu_lattice_boltzmann HPX::hpx HPX::wrap_main)