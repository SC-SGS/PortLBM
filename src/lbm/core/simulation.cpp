/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This source file contains the defintion of crucial functionality of the SYCL lattice Boltzmann simulations.
 * 
 * @version     3.0
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/core/simulation.hpp"
#include "../../../include/lbm/file_interaction/file_interaction.hpp"
#include "../../../include/lbm/exceptions/exceptions.hpp"

lbm::core::Properties::Properties
(
    // Algorithmic properties
    const std::string &&algorithm,
    const std::string &&data_layout,
    const bool debug_mode,
    const bool results_to_csv,
    const double relaxation_time,
    const unsigned int time_steps,
    // Domain properties
    const unsigned int vertical_nodes,
    const unsigned int horizontal_nodes,
    // Inlets
    const double inlet_velocity_x,
    const double inlet_velocity_y,
    const double inlet_density,
    // Outlets
    const double outlet_velocity_x,
    const double outlet_velocity_y,
    const double outlet_density
) 
:
// Algorithmic properties
algorithm(algorithm),
data_layout(data_layout),
debug_mode(debug_mode),
results_to_csv(results_to_csv),
relaxation_time(relaxation_time),
time_steps(time_steps),
// Domain properties
vertical_nodes(vertical_nodes+2),
horizontal_nodes(horizontal_nodes+2),
non_buffered_node_count((vertical_nodes + 2) * (horizontal_nodes + 2)),
buffered_node_count((vertical_nodes + 2) * (horizontal_nodes + 2)),
domain_node_count(vertical_nodes * horizontal_nodes),
// Inlets
inlet_velocity_x(inlet_velocity_x),
inlet_velocity_y(inlet_velocity_y),
inlet_density(inlet_density),
// Outlets
outlet_velocity_x(outlet_velocity_x),
outlet_velocity_y(outlet_velocity_y),
outlet_density(outlet_density)
{};

std::string lbm::core::Properties::to_string() const
{
    return fmt::format
    (
        "Algorithmic properties: \n"
        "\tAlgorithm: {} \n"
        "\tData layout: {} \n"
        "\tDebug mode: {} \n"
        "\tResults to CSV: {} \n"
        "\tRelaxation time: {:.6f} \n"
        "\tTime steps: {} \n"
        "Domain properties: \n"
        "\tVertical nodes: {} \n"
        "\tHorizontal nodes: {} \n"
        "\tNode count: {} \n"
        "Inlets: \n"
        "\tVelocity: ({:.6f}, {:.6f})\n"
        "\tDensity: {:.6f}\n"
        "Outlets: \n"
        "\tVelocity: ({:.6f}, {:.6f})\n"
        "\tDensity: {:.6f}\n\n",
        algorithm,
        data_layout,
        debug_mode,
        results_to_csv,
        relaxation_time,
        time_steps,
        vertical_nodes,
        horizontal_nodes,
        non_buffered_node_count,
        inlet_velocity_x,
        inlet_velocity_y,
        inlet_density,
        outlet_velocity_x,
        outlet_velocity_y,
        outlet_density
        );
}

lbm::core::ExpandedDomainData::ExpandedDomainData
(
    const unsigned int buffered_node_count,
    const unsigned int subdomain_height,
    const unsigned int subdomain_width,
    const unsigned int subdomain_count_vertical,
    const unsigned int subdomain_count_horizontal  
)
:
buffered_node_count(buffered_node_count),
subdomain_height(subdomain_height),
subdomain_width(subdomain_width),
subdomain_count_vertical(subdomain_count_vertical),
subdomain_count_horizontal(subdomain_count_horizontal)
{};

lbm::core::Results::Results(const size_t &size)
:
densities(std::make_unique<std::vector<double>>(size, -1.0f)),
x_velocities(std::make_unique<std::vector<double>>(size, 0.0f)),
y_velocities(std::make_unique<std::vector<double>>(size, 0.0f)),
absolute_velocities(std::make_unique<std::vector<double>>(size, 0.0f))
{};

lbm::core::Results::Results
(
    const std::vector<double> &densities,
    const std::vector<double> &pressures,
    const std::vector<double> &x_velocities,
    const std::vector<double> &y_velocities,
    const std::vector<double> &absolute_velocities  
)
:
densities(std::make_unique<std::vector<double>>(densities)),
x_velocities(std::make_unique<std::vector<double>>(x_velocities)),
y_velocities(std::make_unique<std::vector<double>>(y_velocities)),
absolute_velocities(std::make_unique<std::vector<double>>(absolute_velocities))
{};

lbm::core::Data::Data(const size_t &buffered_node_count)
:
phase_information(std::make_unique<std::vector<uint8_t>>(buffered_node_count, 0)),
is_buffer(std::make_unique<std::vector<uint8_t>>(buffered_node_count, 0)),
distribution_values_0(std::make_unique<std::vector<double>>(buffered_node_count * 9, 0.0f)),
distribution_values_1(std::make_unique<std::vector<double>>(buffered_node_count * 9, 0.0f)),
boundary_interactions(std::make_unique<std::vector<unsigned int>>(buffered_node_count * 9, 0))
{};

lbm::core::Simulation::Simulation()
:
properties(std::move(lbm::file_interaction::json_to_properties())),
data(std::make_unique<Data>(properties->buffered_node_count)),
results(std::make_unique<Results>(properties->domain_node_count))
{};
