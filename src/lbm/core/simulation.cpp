/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This source file contains the defintion of crucial functionality of the SYCL lattice Boltzmann simulations.
 * 
 * @version     4.0
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
    const Obstacle obstacle,
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
obstacle(obstacle),
// Domain properties
vertical_nodes(vertical_nodes+2),
horizontal_nodes(horizontal_nodes+2),
non_buffered_node_count((vertical_nodes + 2) * (horizontal_nodes + 2)),
buffered_node_count((vertical_nodes + 2) * (horizontal_nodes + 2)),
domain_node_count(vertical_nodes * (horizontal_nodes + 2)),
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


lbm::core::Results::Results(const size_t &size, sycl::queue &queue)
:
queue(std::make_shared<sycl::queue>(queue)),

densities_cpu(std::make_unique<std::vector<double>>(size, -1.0f)),
densities_gpu(sycl::malloc_device<double>(size, queue)),

x_velocities_cpu(std::make_unique<std::vector<double>>(size, 0.0f)),
x_velocities_gpu(sycl::malloc_device<double>(size, queue)),

y_velocities_cpu(std::make_unique<std::vector<double>>(size, 0.0f)),
y_velocities_gpu(sycl::malloc_device<double>(size, queue)),

absolute_velocities_cpu(std::make_unique<std::vector<double>>(size, 0.0f)),
absolute_velocities_gpu(sycl::malloc_device<double>(size, queue))
{
    queue.fill(densities_gpu, -1.0, size).wait();
};


lbm::core::Data::Data(const size_t &total_node_count, const sycl::queue &queue, bool two_lattice)
:
queue(std::make_shared<sycl::queue>(queue)),
phase_information(sycl::malloc_device<int8_t>(total_node_count, queue)),
distribution_values_0(sycl::malloc_device<double>(9 * total_node_count, queue))
{
    if(two_lattice) distribution_values_1 = sycl::malloc_device<double>(9 * total_node_count, queue);
    else distribution_values_1 = NULL;
};


lbm::core::Simulation::Simulation(sycl::queue &queue)
:
properties(std::make_unique<Properties>(file_interaction::json_to_properties())),
results(std::make_unique<Results>(properties->domain_node_count, queue))
{
    if(properties->algorithm == "gpu-two-lattice-linear" || properties->algorithm == "gpu-two-lattice")
        data = std::make_unique<Data>(properties->buffered_node_count, queue, true);
    else
        data = std::make_unique<Data>(properties->buffered_node_count, queue, false);
};
