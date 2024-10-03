#include "../include/parallel_swap_framework.hpp"
#include "../include/file_interaction.hpp"

#include <iostream>

#include <hpx/algorithm.hpp>



/**
 * @brief Performs the parallel two-step algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes a vector containing the first and last element of an iterator over all fluid nodes within each subdomain
 * @param distribution_values the vector containing the distribution values of all nodes
 * @param bsi see documentation of border_swap_information
 * @param access_function the access function according to which the values are to be accessed
 * @param iterations this many iterations will be performed
 */
void parallel_swap_framework::run
(  
    const std::vector<start_end_it_tuple> &fluid_nodes,       
    std::vector<double> &distribution_values, 
    const border_swap_information &bsi,
    const access_function access_function,
    const unsigned int iterations
)
{
    // Initializations relevant for buffering
    std::vector<std::tuple<unsigned int, unsigned int>> buffer_ranges;
    std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> y_values;
    parallel_framework::buffer_dimension_initializations(buffer_ranges, y_values);

    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    /* Parallelization framework */
    for(auto time = 0; time < iterations; ++time)
    {
        result[time] = parallel_swap_framework::stream_and_collide
        (fluid_nodes, bsi, distribution_values, access_function, y_values, buffer_ranges);
    }

    if(RESULTS_TO_CSV)
    {
        parallel_domain_sim_data_to_csv(result, "results.csv");
    }
}

/**
 * @brief Performs the parallel two-step algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes a vector containing the first and last element of an iterator over all fluid nodes within each subdomain
 * @param distribution_values the vector containing the distribution values of all nodes
 * @param bsi see documentation of border_swap_information
 * @param access_function the access function according to which the values are to be accessed
 * @param iterations this many iterations will be performed
 */
void parallel_swap_framework::run_debug
(  
    const std::vector<start_end_it_tuple> &fluid_nodes,       
    std::vector<double> &distribution_values, 
    const border_swap_information &bsi,
    const access_function access_function,
    const unsigned int iterations
)
{
    to_console::print_run_greeting("parallel swap algorithm", iterations);
    
    // Initializations relevant for buffering
    std::vector<std::tuple<unsigned int, unsigned int>> buffer_ranges;
    std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> y_values;
    parallel_framework::buffer_dimension_initializations(buffer_ranges, y_values);

    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    /* Parallelization framework */
    for(auto time = 0; time < iterations; ++time)
    {
        std::cout << "\033[33mIteration " << time << ":\033[0m";

        result[time] = parallel_swap_framework::stream_and_collide_debug
        (fluid_nodes, bsi, distribution_values, access_function, y_values, buffer_ranges);

        std::cout << "\tFinished iteration " << time << std::endl;
    }

    if(RESULTS_TO_CSV)
    {
        parallel_domain_sim_data_to_csv(result, "results.csv");
    }

    to_console::buffered::print_simulation_results(result);
    std::cout << "All done, exiting simulation. " << std::endl;
}

/**
 * @brief Performs the streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 *        This variant will print several debug comments to the console.
 * 
 * @param fluid_nodes a vector containing the first and last element of an iterator over all fluid nodes within each subdomain
 * @param bsi see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values
 * @param access_function the access to node values will be performed according to this access function
 * @param y_values a tuple containing the y values of all regular layers (0) and all buffer layers (1)
 * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
 * @return sim_data_tuple see documentation of sim_data_tuple
 */
sim_data_tuple parallel_swap_framework::stream_and_collide
(
    const std::vector<start_end_it_tuple> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &distribution_values,    
    const access_function access_function,
    const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
    const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
)
{
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    /* Border node initialization */
    hpx::for_each
    (
        hpx::execution::par, 
        bsi.begin(), 
        bsi.end(), 
        [&](std::vector<unsigned int> node)
        {
        for(auto it = node.begin() + 1; it < node.end(); ++it)
            {
                sequential_swap::perform_swap_step(distribution_values, node[0], access_function, *it);
            }
        });

    /* Buffer update */
    hpx::experimental::for_loop(
        hpx::execution::par, 0, BUFFER_COUNT, 
        [&](unsigned int buffer_index)
        {
            parallel_swap_framework::swap_buffer_update(buffer_ranges[buffer_index], distribution_values, access_function);
        });

    hpx::experimental::for_loop(
        hpx::execution::par, 0, SUBDOMAIN_COUNT, 
        [&](unsigned int subdomain)
        {
            for(auto node = std::get<0>(fluid_nodes[subdomain]); node <= std::get<1>(fluid_nodes[subdomain]); ++node)
            {
                // Swapping step
                sequential_swap::perform_swap_step(distribution_values, *node, access_function, sequential_swap::ACTIVE_STREAMING_DIRECTIONS);

                // Restore precious order in here
                sequential_swap::restore_order(distribution_values, *node, access_function);

                /* Perform collision for all fluid nodes */
                collision::perform_collision(*node, distribution_values, access_function, velocities, densities);
            }
        });

    /* Update ghost nodes */
    parallel_framework::update_velocity_input_density_output(y_values, distribution_values, velocities, densities, access_function);

    sequential_swap::restore_inout_correctness(distribution_values, access_function);

    /* Buffer correction */
    parallel_framework::outstream_buffer_update(distribution_values, y_values, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs the streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 *        This variant will print several debug comments to the console.
 * 
 * @param fluid_nodes a vector containing the first and last element of an iterator over all fluid nodes within each subdomain
 * @param bsi see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values
 * @param access_function the access to node values will be performed according to this access function
 * @param y_values a tuple containing the y values of all regular layers (0) and all buffer layers (1)
 * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
 * @return sim_data_tuple see documentation of sim_data_tuple
 */
sim_data_tuple parallel_swap_framework::stream_and_collide_debug
(
    const std::vector<start_end_it_tuple> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &distribution_values,    
    const access_function access_function,
    const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
    const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
)
{
    std::cout << "Distribution values before stream and collide: " << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    // Border node initialization
    for(const auto node : bsi)
    {
        for(auto it = node.begin() + 1; it < node.end(); ++it)
        {
            sequential_swap::perform_swap_step(distribution_values, node[0], access_function, *it);
        }
    }
    std::cout << "Distribution values after border node initialization: " << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, access_function);

    /* Buffer update */
    std::cout << "Copying to buffer" << std::endl;
    for(auto buffer_index = 0; buffer_index < BUFFER_COUNT; ++buffer_index)
    {
        parallel_swap_framework::swap_buffer_update(buffer_ranges[buffer_index], distribution_values, access_function);
    }
    std::cout << "Distribution values after buffer update: " << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, access_function);

    // Swapping step
    for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
    {
        for(auto node = std::get<0>(fluid_nodes[subdomain]); node <= std::get<1>(fluid_nodes[subdomain]); ++node)
        {
            sequential_swap::perform_swap_step(distribution_values, *node, access_function, sequential_swap::ACTIVE_STREAMING_DIRECTIONS);
        }
    }

    std::cout << "Distribution values after swap for every node: " << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    // Restore precious order in here
    for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
    {
        for(auto node = std::get<0>(fluid_nodes[subdomain]); node <= std::get<1>(fluid_nodes[subdomain]); ++node)
        {
            sequential_swap::restore_order(distribution_values, *node, access_function);
        }
    }

    std::cout << "Distribution values after ORDER has been restored for every node: " << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, access_function);

    /* Perform collision for all fluid nodes */
    for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
    {
        for(auto node = std::get<0>(fluid_nodes[subdomain]); node <= std::get<1>(fluid_nodes[subdomain]); ++node)
        {
            collision::perform_collision(*node, distribution_values, access_function, velocities, densities);
        }
    }
    std::cout << "Distribution values after collision: " << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    /* Update ghost nodes */
    parallel_framework::update_velocity_input_density_output(y_values, distribution_values, velocities, densities, access_function);
    std::cout << "Distribution values after ghost node update: " << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    sequential_swap::restore_inout_correctness(distribution_values, access_function);

    /* Buffer correction */
    parallel_framework::outstream_buffer_update(distribution_values, y_values, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs an update for the buffer with the specified boundaries.
 *        It prepares the subdomain-wise streaming and performs the swap step for the uppermost row of each subdomain.
 * 
 * @param buffer_bounds a tuple containing the indices of the first and the last node of the buffer
 * @param distribution_values a vector containing all distribution values
 * @param access_function the access to node values will be performed according to this access function
 */
void parallel_swap_framework::swap_buffer_update
(
    const std::tuple<unsigned int, unsigned int> &buffer_bounds,
    std::vector<double> &distribution_values,
    const access_function access_function
)
{
    unsigned int start = std::get<0>(buffer_bounds);
    unsigned int end = std::get<1>(buffer_bounds);

    // Clone values facing southward from subdomain above
    for(auto buffer_node = start; buffer_node <= end; ++buffer_node)
    {
        for(auto direction : {0,1,2})
        {
            distribution_values[access_function(buffer_node, direction)] = distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 7), direction)];
        }
    }

    // Perform streaming across buffer
    for(auto buffer_node = start + 1; buffer_node <= end - 1; ++buffer_node)
    {
        for(auto direction : {6,7,8})
        {
            distribution_values[access_function(lbm_access::get_neighbor(buffer_node, direction), invert_direction(direction))] = 
            distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 1), direction)];
        }
    }
}
