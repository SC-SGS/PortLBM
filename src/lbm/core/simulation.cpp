/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This source file contains the defintion of crucial functionality of the SYCL lattice Boltzmann 
 *              simulations.
 * 
 * @version     4.4
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/core/simulation.hpp"
#include "../../../include/lbm/file_interaction/file_interaction.hpp"
#include "../../../include/lbm/exceptions/exceptions.hpp"

// Properties /////////////////////////////////////////////////////////////////////////////////////////////////////////

lbm::core::Properties::Properties
(
    // Algorithmic properties
    const std::string &&algorithm,
    const std::string &&data_layout,
    const bool debug_mode,
    const unsigned int work_group_size,
    const unsigned int time_steps,
    const unsigned int frame_update_interval,
    // Domain properties
    const std::string &&scenario,
    const unsigned int vertical_nodes,
    const unsigned int horizontal_nodes,
    // Physical
    const real_type inlet_velocity_x,
    const real_type inlet_velocity_y,
    const real_type inlet_density,
    const real_type outlet_velocity_x,
    const real_type outlet_velocity_y,
    const real_type outlet_density,
    const real_type relaxation_time
) 
:
// Algorithmic properties
algorithm(algorithm),
data_layout(data_layout),
debug_mode(debug_mode),
work_group_size(work_group_size),
time_steps(time_steps),
frame_update_interval(frame_update_interval),
// Domain properties
scenario(scenario),
vertical_nodes(vertical_nodes + 2),
horizontal_nodes(horizontal_nodes + 2),
total_unexpanded_node_count((vertical_nodes + 2) * (horizontal_nodes + 2)),
domain_node_count(vertical_nodes * horizontal_nodes),
// Physical
inlet_velocity_x(inlet_velocity_x),
inlet_velocity_y(inlet_velocity_y),
inlet_density(inlet_density),
outlet_velocity_x(outlet_velocity_x),
outlet_velocity_y(outlet_velocity_y),
outlet_density(outlet_density),
relaxation_time(relaxation_time)
{};

// Control ////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        "\tWork group size: {} \n"
        "\tDebug mode: {} \n"
        "\tTime steps: {} \n"
        "\tFrame update interval: {} \n"
        "Domain properties: \n"
        "\tScenario: {}\n"
        "\tVertical nodes: {} \n"
        "\tHorizontal nodes: {} \n"
        "\tNode count: {} \n"
        "Physical: \n"
        "\tInlet velocity: ({:.6f}, {:.6f})\n"
        "\tInlet density: {:.6f}\n"
        "\tOutlet velocity: ({:.6f}, {:.6f})\n"
        "\tOutlet density: {:.6f}\n"
        "\tRelaxation time: {:.6f}\n\n",
        algorithm,
        data_layout,
        work_group_size,
        debug_mode,
        time_steps,
        frame_update_interval,
        scenario,
        vertical_nodes,
        horizontal_nodes,
        total_unexpanded_node_count,
        inlet_velocity_x,
        inlet_velocity_y,
        inlet_density,
        outlet_velocity_x,
        outlet_velocity_y,
        outlet_density,
        relaxation_time
    );
}

// Domain /////////////////////////////////////////////////////////////////////////////////////////////////////////////

lbm::core::Domain::Domain
(
    unsigned int total_node_count,
    unsigned int vertical_nodes,
    unsigned int horizontal_nodes,
    unsigned int subdomain_vertical_nodes,
    unsigned int subdomain_horizontal_nodes,
    unsigned int subdomain_count_vertical,
    unsigned int subdomain_count_horizontal
)
:
total_node_count(total_node_count),
vertical_nodes(vertical_nodes),
horizontal_nodes(horizontal_nodes),
subdomain_vertical_nodes(subdomain_vertical_nodes),
subdomain_horizontal_nodes(subdomain_horizontal_nodes),
subdomain_count_vertical(subdomain_count_vertical),
subdomain_count_horizontal(subdomain_count_horizontal)
{};


lbm::core::SwapDomain::SwapDomain
(
    const unsigned int unexpanded_horizontal_nodes,
    const unsigned int unexpanded_vertical_nodes,
    const size_t work_group_size
)
:
Domain(0, 0, 0, 0, 0, 0, 0)
{
    // Is work_group_size uneven (and larger than 1)?
    if (work_group_size & 0x1)
    {
        // ==> stripe subdomains with extents 1 and work_group_size
        subdomain_horizontal_nodes = work_group_size;
        subdomain_vertical_nodes = 1;
    }
    else // work_group_size is guaranteed to be even, that is, a multiple of 2
    {
        size_t power_4 = power_functions::which_power_of_4(work_group_size);
        size_t power_2 = power_functions::which_power_of_2(work_group_size);

        if (power_4) // Is work_group_size a power of four? 
        {
            // ==> square subdomains

            size_t size = 1;
            for(int i = 0; i < power_4; ++i) { size *= 2; }

            subdomain_horizontal_nodes = size;
            subdomain_vertical_nodes = size; 
        }
        else if (power_2) // Is work_group_size a power of two?
        {
            // ==> as close to square as possible, with preference for horizontal length 
            size_t power_height = power_2 / 2;
            size_t height = 1;
            for(int i = 0; i < power_height; ++i) { height *= 2; }

            subdomain_vertical_nodes = height;
            subdomain_horizontal_nodes = 2 * height; 
        }
        else // work_group_size is a multiple of two
        {
            //  ==> stripe subdomains with extents 2 and work_group_size / 2
            subdomain_vertical_nodes = 2;
            subdomain_horizontal_nodes = work_group_size / 2; 
        }
    }
    subdomain_vertical_nodes -= 1;
    subdomain_horizontal_nodes -= 2; // should be 2

    subdomain_count_vertical = (((unexpanded_vertical_nodes - 2) / subdomain_vertical_nodes) + 1);
    if(!((unexpanded_vertical_nodes - 2) % subdomain_vertical_nodes)) { subdomain_count_vertical--; }
    vertical_nodes = (subdomain_vertical_nodes + 1) * subdomain_count_vertical + 1;
    
    subdomain_count_horizontal = (((unexpanded_horizontal_nodes - 2) / subdomain_horizontal_nodes) + 1);
    if(!((unexpanded_horizontal_nodes - 2) % subdomain_horizontal_nodes)) { subdomain_count_horizontal--; }
    horizontal_nodes = (subdomain_horizontal_nodes + 1) * subdomain_count_horizontal + 1;

    total_node_count = vertical_nodes * horizontal_nodes;
};


lbm::core::DecomposedTwoLatticeDomain::DecomposedTwoLatticeDomain
(
    const unsigned int unexpanded_horizontal_nodes,
    const unsigned int unexpanded_vertical_nodes,
    const size_t work_group_size
)
:
Domain(0, 0, 0, 0, 0, 0, 0)
{
    // Is work_group_size uneven (and larger than 1)?
    if (work_group_size & 0x1)
    {
        // ==> stripe subdomains with extents 1 and work_group_size
        subdomain_horizontal_nodes = work_group_size;
        subdomain_vertical_nodes = 1;
    }
    else // work_group_size is guaranteed to be even, that is, a multiple of 2
    {
        size_t power_4 = power_functions::which_power_of_4(work_group_size);
        size_t power_2 = power_functions::which_power_of_2(work_group_size);

        if (power_4) // Is work_group_size a power of four? 
        {
            // ==> square subdomains

            size_t size = 1;
            for(int i = 0; i < power_4; ++i) { size *= 2; }

            subdomain_horizontal_nodes = size;
            subdomain_vertical_nodes = size; 
        }
        else if (power_2) // Is work_group_size a power of two?
        {
            // ==> as close to square as possible, with preference for horizontal length 
            size_t power_height = power_2 / 2;
            size_t height = 1;
            for(int i = 0; i < power_height; ++i) { height *= 2; }

            subdomain_vertical_nodes = height;
            subdomain_horizontal_nodes = 2 * height; 
        }
        else // work_group_size is a multiple of two
        {
            //  ==> stripe subdomains with extents 2 and work_group_size / 2
            subdomain_vertical_nodes = 2;
            subdomain_horizontal_nodes = work_group_size / 2; 
        }
    }

    subdomain_count_vertical = (((unexpanded_vertical_nodes - 2) / subdomain_vertical_nodes) + 1);
    if(!((unexpanded_vertical_nodes - 2) % subdomain_vertical_nodes)) { subdomain_count_vertical--; }
    vertical_nodes = (subdomain_vertical_nodes) * subdomain_count_vertical + 2;
    
    subdomain_count_horizontal = (((unexpanded_horizontal_nodes - 2) / subdomain_horizontal_nodes) + 1);
    if(!((unexpanded_horizontal_nodes - 2) % subdomain_horizontal_nodes)) { subdomain_count_horizontal--; }
    horizontal_nodes = (subdomain_horizontal_nodes) * subdomain_count_horizontal + 2;

    total_node_count = vertical_nodes * horizontal_nodes;
};


lbm::core::NonDecomposedTwoLatticeDomain::NonDecomposedTwoLatticeDomain
(
    const unsigned int horizontal_nodes,
    const unsigned int vertical_nodes
)
:
Domain
(
    horizontal_nodes * vertical_nodes,
    vertical_nodes,
    horizontal_nodes,
    vertical_nodes,
    horizontal_nodes,
    1,
    1
)
{};


lbm::core::Results::Results(const size_t &size, sycl::queue &queue)
:
queue(std::make_shared<sycl::queue>(queue)),

densities_cpu(std::make_unique<std::vector<real_type>>(size, -1.0f)),
densities_gpu(sycl::malloc_device<real_type>(size, queue)),

x_velocities_cpu(std::make_unique<std::vector<real_type>>(size, 0.0f)),
x_velocities_gpu(sycl::malloc_device<real_type>(size, queue)),

y_velocities_cpu(std::make_unique<std::vector<real_type>>(size, 0.0f)),
y_velocities_gpu(sycl::malloc_device<real_type>(size, queue)),

absolute_velocities_cpu(std::make_unique<std::vector<real_type>>(size, 0.0f)),
absolute_velocities_gpu(sycl::malloc_device<real_type>(size, queue))
{
    #ifdef USE_FLOAT
    queue.fill(densities_gpu, -1.0f, size).wait();
    queue.fill(x_velocities_gpu, 0.0f, size).wait();
    queue.fill(y_velocities_gpu, 0.0f, size).wait();
    queue.fill(absolute_velocities_gpu, 0.0f, size).wait();
    #else
    queue.fill(densities_gpu, -1.0, size).wait();
    queue.fill(x_velocities_gpu, 0.0, size).wait();
    queue.fill(y_velocities_gpu, 0.0, size).wait();
    queue.fill(absolute_velocities_gpu, 0.0, size).wait();
    #endif

    densities_cpu->shrink_to_fit();
    x_velocities_cpu->shrink_to_fit();
    y_velocities_cpu->shrink_to_fit();
    absolute_velocities_cpu->shrink_to_fit();
};


lbm::core::Data::Data(const size_t total_node_count, sycl::queue &queue, const bool two_lattice)
:
queue(std::make_shared<sycl::queue>(queue)),
phase_information(sycl::malloc_device<int8_t>(total_node_count, queue)),
distribution_values_0(sycl::malloc_device<real_type>(9 * total_node_count, queue))
{
    queue.fill(phase_information, (int8_t)-1, total_node_count).wait();
    if(two_lattice) distribution_values_1 = sycl::malloc_device<real_type>(9 * total_node_count, queue);
    else distribution_values_1 = nullptr;
};

// Simulation /////////////////////////////////////////////////////////////////////////////////////////////////////////

lbm::core::Simulation::Simulation(sycl::queue &queue)
:
properties(std::make_unique<Properties>(file_interaction::json_to_properties())),
results(std::make_unique<Results>(properties->domain_node_count, queue)),
control(std::make_unique<Control>(properties->time_steps))
{
    size_t max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();

    if(properties->work_group_size > max_work_group_size)
    {
        size_t wrong_size = properties->work_group_size;
        properties->work_group_size = max_work_group_size;
        properties->horizontal_nodes -= 2;
        properties->vertical_nodes -= 2;

        lbm::file_interaction::properties_to_json(*properties);
        
        throw lbm::exceptions::json::PropertyArgumentException
        (
            fmt::format
            (
                "Detected illegal work group size of {} that exceeds the maximum work group size of {}. "
                "JSON property \"workGroupSize\" has been set to {} to enable graceful program restart. ",
                wrong_size,
                max_work_group_size,
                max_work_group_size
            )
        );
    }
    if(properties->algorithm == "gpu-two-lattice-linear")
    {
        domain = std::make_unique<NonDecomposedTwoLatticeDomain>(
            properties->horizontal_nodes,
            properties->vertical_nodes
        );
        data = std::make_unique<Data>(properties->total_unexpanded_node_count, queue, true);
    }
    else if(properties->algorithm == "gpu-two-lattice")
    {
        domain = std::make_unique<DecomposedTwoLatticeDomain>(
            properties->horizontal_nodes,
            properties->vertical_nodes,
            properties->work_group_size
        );
        data = std::make_unique<Data>(domain->total_node_count, queue, true);
    }
    // soon:
    // else if(properties->algorithm == "gpu-two-lattice-optimized")
    // {
    //     domain = std::make_unique<DecomposedTwoLatticeDomain>(
    //         properties->horizontal_nodes,
    //         properties->vertical_nodes,
    //         properties->work_group_size
    //     );
    //     data = std::make_unique<Data>(domain->total_node_count, queue, false);
    // }
    else
    {
        domain = std::make_unique<SwapDomain>(
            properties->horizontal_nodes,
            properties->vertical_nodes,
            properties->work_group_size
        );
        data = std::make_unique<Data>(domain->total_node_count, queue, false);
    }
        
};
