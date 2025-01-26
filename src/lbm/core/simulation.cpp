/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This source file contains the defintion of crucial functionality of the SYCL lattice Boltzmann simulations.
 * 
 * @version     4.1
 * 
 * @date        January 2025
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
    const std::string &&scenario,
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
scenario(scenario),
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


lbm::core::Control::Control(const unsigned int max_iterations)
:
stopped(false),
current_iteration(0),
max_iterations(max_iterations),
progress(0),
timer(std::make_unique<hpx::chrono::high_resolution_timer>()),
last_frametime(0)
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
        "\tScenario: {}\n"
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
        scenario,
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


lbm::core::DecomposedDomain::DecomposedDomain
(
    const unsigned int unexpanded_horizontal_nodes,
    const unsigned int unexpanded_vertical_nodes,
    const size_t max_work_group_size,
    const bool buffered 
)
{
    // Is max_work_group_size uneven (and larger than 1)?
    if (max_work_group_size & 0x1)
    {
        // ==> stripe subdomains with extents 1 and max_work_group_size
        subdomain_width = max_work_group_size;
        subdomain_height = 1;
    }
    else // max_work_group_size is guaranteed to be even, that is, a multiple of 2
    {
        size_t power_4 = power_functions::which_power_of_4(max_work_group_size);
        size_t power_2 = power_functions::which_power_of_2(max_work_group_size);

        if (power_4) // Is max_work_group_size a power of four? 
        {
            // ==> square subdomains

            size_t size = 1;
            for(int i = 0; i < power_4; ++i) { size *= 2; }

            subdomain_width = size;
            subdomain_height = size; 
        }
        else if (power_2) // Is max_work_group_size a power of two?
        {
            // ==> as close to square as possible, with preference for horizontal length 
            size_t power_height = power_2 / 2;
            size_t height = 1;
            for(int i = 0; i < power_height; ++i) { height *= 2; }

            subdomain_height = height;
            subdomain_width = 2 * height; 
        }
        else // max_work_group_size is a multiple of two
        {
            //  ==> stripe subdomains with extents 2 and max_work_group_size / 2
            subdomain_height = 2;
            subdomain_width = max_work_group_size / 2; 
        }
    }

    subdomain_count_vertical = ((unexpanded_vertical_nodes / subdomain_height) + 1);
    if(!(unexpanded_vertical_nodes % subdomain_height)) { subdomain_count_vertical--; }
    expanded_vertical_nodes = (subdomain_height + buffered) * subdomain_count_vertical - buffered;
    
    subdomain_count_horizontal = ((unexpanded_horizontal_nodes / subdomain_width) + 1);
    if(!(unexpanded_vertical_nodes % subdomain_width)) { subdomain_count_horizontal--; }
    expanded_horizontal_nodes = (subdomain_width + buffered) * subdomain_count_horizontal - buffered;

    expanded_node_count = expanded_vertical_nodes * expanded_horizontal_nodes;
};

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
    queue.fill(x_velocities_gpu, 0.0, size).wait();
    queue.fill(y_velocities_gpu, 0.0, size).wait();
    queue.fill(absolute_velocities_gpu, 0.0, size).wait();

    densities_cpu->shrink_to_fit();
    x_velocities_cpu->shrink_to_fit();
    y_velocities_cpu->shrink_to_fit();
    absolute_velocities_cpu->shrink_to_fit();
};


lbm::core::Data::Data(const size_t &total_node_count, const sycl::queue &queue, bool two_lattice)
:
queue(std::make_shared<sycl::queue>(queue)),
phase_information(sycl::malloc_device<int8_t>(total_node_count, queue)),
distribution_values_0(sycl::malloc_device<double>(9 * total_node_count, queue))
{
    if(two_lattice) distribution_values_1 = sycl::malloc_device<double>(9 * total_node_count, queue);
    else distribution_values_1 = nullptr;
};


lbm::core::Simulation::Simulation(sycl::queue &queue)
:
properties(std::make_unique<Properties>(file_interaction::json_to_properties())),
results(std::make_unique<Results>(properties->domain_node_count, queue)),
control(std::make_unique<Control>(properties->time_steps))
{
    if(properties->algorithm == "gpu-two-lattice-linear")
    {
        data = std::make_unique<Data>(properties->buffered_node_count, queue, true);
    }
    else if(properties->algorithm == "gpu-two-lattice")
    {
        decomposed_domain = std::make_unique<DecomposedDomain>(
            properties->horizontal_nodes,
            properties->vertical_nodes,
            queue.get_device().get_info<sycl::info::device::max_work_group_size>(),
            false
        );
        data = std::make_unique<Data>(decomposed_domain->expanded_node_count, queue, true);
    }
    else
    {
        decomposed_domain = std::make_unique<DecomposedDomain>(
            properties->horizontal_nodes,
            properties->vertical_nodes,
            queue.get_device().get_info<sycl::info::device::max_work_group_size>(),
            true
        );
        data = std::make_unique<Data>(decomposed_domain->expanded_node_count, queue, false);
    }
        
};
