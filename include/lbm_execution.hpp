#ifndef LBM_EXECUTION_HPP
#define LBM_EXECUTION_HPP

#include "defines.hpp"
#include "simulation.hpp"
#include "file_interaction.hpp"

#include "sequential_two_lattice.hpp"
// #include "sequential_two_step.hpp"
// #include "sequential_swap.hpp"
// #include "sequential_shift.hpp"

#include "parallel_two_lattice.hpp"
// #include "parallel_two_lattice_framework.hpp"
// #include "parallel_two_step_framework.hpp"
// #include "parallel_swap_framework.hpp"
// #include "parallel_shift_framework.hpp"

void debug_prints
(
    const std::vector<double> &distribution_values,
    const std::vector<unsigned int> &nodes,
    const std::vector<unsigned int> &fluid_nodes,
    const std::vector<bool> &phase_information,
    const border_swap_information &swap_info
);

void debug_prints
(
    const std::vector<double> &distribution_values,
    const std::vector<unsigned int> &nodes,
    const std::vector<unsigned int> &fluid_nodes,
    const std::vector<bool> &phase_information,
    const std::vector<border_swap_information> &swap_info
);

void debug_prints
(
    const std::vector<double> &distribution_values,
    const std::vector<unsigned int> &nodes,
    const std::vector<unsigned int> &fluid_nodes,
    const std::vector<bool> &phase_information
);

void setup_global_variables(const Settings &settings);

void select_and_execute(const std::string &algorithm);

void execute_sequential_two_lattice();

void execute_sequential_two_step();

void execute_sequential_swap();

void execute_sequential_shift();

void execute_parallel_two_lattice();

void execute_parallel_two_lattice_framework();

void execute_parallel_two_step();

void execute_parallel_swap();

void execute_parallel_shift();




#endif
