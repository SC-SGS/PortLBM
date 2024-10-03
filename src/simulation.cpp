/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Implementation of the updated version of the lattice Boltzmann simulation.hpp first introduced 
 *              in my SimTech project work:
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann.
 *              Initially, this file only contained operations to set up an example domain.
 *              Additional functionality was added to support the updated structure that suits the GPU implementation better. 
 * 
 * @version     2.0
 * 
 * @date        2024-09-27
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#include "../include/simulation.hpp"

Properties::Properties
(
    const bool debug_mode,
    const bool results_to_csv,
    const double relaxation_time,
    const unsigned int time_steps,
    const unsigned int vertical_nodes,
    const unsigned int horizontal_nodes,
    const unsigned int non_buffered_node_count,
    const unsigned int buffered_node_count,
    const unsigned int subdomain_height,
    const unsigned int subdomain_width,
    const unsigned int subdomain_count_vertical,
    const unsigned int subdomain_count_horizontal,
    const double inlet_velocity_x,
    const double inlet_velocity_y,
    const double outlet_velocity_x,
    const double outlet_velocity_y,
    const std::string &&data_layout,
    const std::string &&algorithm
) 
:
debug_mode(debug_mode),
results_to_csv(results_to_csv),
relaxation_time(relaxation_time),
time_steps(time_steps),
vertical_nodes(vertical_nodes),
horizontal_nodes(horizontal_nodes),
non_buffered_node_count(non_buffered_node_count),
buffered_node_count(buffered_node_count),
subdomain_height(subdomain_height),
subdomain_width(subdomain_width),
subdomain_count_vertical(subdomain_count_vertical),
subdomain_count_horizontal(subdomain_count_horizontal),
inlet_velocity_x(inlet_velocity_x),
inlet_velocity_y(inlet_velocity_y),
outlet_velocity_x(outlet_velocity_x),
outlet_velocity_y(outlet_velocity_y),
data_layout(data_layout),
algorithm(algorithm)
{};

SimulationResults::SimulationResults
(
    const Properties &properties
)
:
densities(std::make_shared<std::vector<double>>(properties.time_steps * properties.buffered_node_count, -1.0f)),
pressures(std::make_shared<std::vector<double>>(properties.time_steps * properties.buffered_node_count, -1.0f)),
x_velocities(std::make_shared<std::vector<double>>(properties.time_steps * properties.buffered_node_count, 0.0f)),
y_velocities(std::make_shared<std::vector<double>>(properties.time_steps * properties.buffered_node_count, 0.0f))
{};

LegacyData::LegacyData
(
    const border_swap_information &bsi, 
    const access_function &acc_f
)
:
bsi(std::make_shared<border_swap_information>(bsi)),
acc_f(acc_f)
{};
