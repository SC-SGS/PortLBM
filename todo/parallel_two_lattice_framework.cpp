#include "../include/parallel_two_lattice_framework.hpp"

#include <iostream>

#include <hpx/algorithm.hpp>

/**
 * @brief Performs the framework-based parallel two-lattice algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector of tuples of iterators pointing at the first and last fluid node of each domain
 * @param boundary_nodes see documentation of border_swap_information
 * @param distribution_values_0 source for even time steps and destination for odd time steps
 * @param distribution_values_1 source for odd time steps and destination for even time steps
 * @param access_function the access function according to which distribution values are to be accessed
 * @param iterations this many iterations will be performed
 */
void parallel_two_lattice_framework::run
(  
    const std::vector<start_end_it_tuple> &fluid_nodes,       
    const border_swap_information &boundary_nodes,
    std::vector<double> &distribution_values_0, 
    std::vector<double> &distribution_values_1,   
    const access_function access_function,
    const unsigned int iterations
)
{
     std::vector<double> temp;

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
        result[time] = parallel_two_lattice_framework::stream_and_collide
        (fluid_nodes, boundary_nodes, distribution_values_0, distribution_values_1, access_function, y_values, buffer_ranges);

        temp = std::move(distribution_values_0);
        distribution_values_0 = std::move(distribution_values_1);
        distribution_values_1 = std::move(temp);
    }

    if(RESULTS_TO_CSV)
    {
        parallel_domain_sim_data_to_csv(result, "results.csv");
    }
}

/**
 * @brief Performs the framework-based parallel two-lattice algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector of tuples of iterators pointing at the first and last fluid node of each domain
 * @param boundary_nodes see documentation of border_swap_information
 * @param distribution_values_0 source for even time steps and destination for odd time steps
 * @param distribution_values_1 source for odd time steps and destination for even time steps
 * @param access_function the access function according to which distribution values are to be accessed
 * @param iterations this many iterations will be performed
 */
void parallel_two_lattice_framework::run_debug
(  
    const std::vector<start_end_it_tuple> &fluid_nodes,       
    const border_swap_information &boundary_nodes,
    std::vector<double> &distribution_values_0, 
    std::vector<double> &distribution_values_1,   
    const access_function access_function,
    const unsigned int iterations
)
{
     std::vector<double> temp;

    to_console::print_run_greeting("parallel two-lattice algorithm (framework version)", iterations);

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

        result[time] = parallel_two_lattice_framework::stream_and_collide_debug
        (fluid_nodes, boundary_nodes, distribution_values_0, distribution_values_1, access_function, y_values, buffer_ranges);

        std::cout << "\tFinished iteration " << time << std::endl;

        temp = std::move(distribution_values_0);
        distribution_values_0 = std::move(distribution_values_1);
        distribution_values_1 = std::move(temp);
    }

    if(RESULTS_TO_CSV)
    {
        parallel_domain_sim_data_to_csv(result, "results.csv");
    }

    to_console::buffered::print_simulation_results(result);
    std::cout << "All done, exiting simulation. " << std::endl;
}

/**
 * @brief Performs the combined streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 * 
 * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain.
 * @param bsi see documentation of border_swap_information
 * @param source a vector containing the distribution values of the previous time step
 * @param destination the distribution values will be written to this vector after performing both steps.
 * @param access_function the function used to access the distribution values
 * @param y_values a tuple containing the y values of all regular layers (0) and all buffer layers (1)
 * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
 * @return see documentation of sim_data_tuple
 */
sim_data_tuple parallel_two_lattice_framework::stream_and_collide
(
    const std::vector<start_end_it_tuple> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &source, 
    std::vector<double> &destination,    
    const access_function access_function,
    const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
    const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
)
{
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    // Global boundary update
    parallel_framework::emplace_bounce_back_values(bsi, source, access_function);

    // Buffer update
    hpx::experimental::for_loop
    (
        hpx::execution::par, 0, BUFFER_COUNT,
        [&](unsigned int buffer_index)
        {
            parallel_framework::copy_to_buffer(buffer_ranges[buffer_index], source, access_function);
        }
    );

    hpx::experimental::for_loop
    (
        hpx::execution::par, 0, SUBDOMAIN_COUNT, 
        [&](unsigned int subdomain)
        {
            for(auto it = std::get<0>(fluid_nodes[subdomain]); it <= std::get<1>(fluid_nodes[subdomain]); ++it)
            {
                /* Streaming step */
                sequential_two_lattice::tl_stream(
                    source, 
                    destination, 
                    access_function, 
                    *it);

                /* Collision step */
                collision::perform_collision(
                    *it, 
                    destination, 
                    access_function, 
                    velocities,
                    densities);          
            }
        }
    );

    parallel_framework::update_velocity_input_density_output(y_values, destination, velocities, densities, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief This method is a serialized debug version of perform_tl_stream_and_collide_parallel.
 *        It acts as a proof-of-concept method that is suitable for testing such that errors related to
 *        the framework itself rather than the actual parallelization can be spotted.
 *        This variant of the combined streaming and collision step will print several debug comments to the console.
 * 
 * @param fluid_nodes A vector of tuples of iterators pointing at the first and last fluid node of each domain
 * @param bsi see documentation of border_swap_information
 * @param distribution_values_0 source for even time steps and destination for odd time steps
 * @param distribution_values_1 source for odd time steps and destination for even time steps
 * @param access_function the access function according to which distribution values are to be accessed
 * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
 * @return see documentation of sim_data_tuple
 */
sim_data_tuple parallel_two_lattice_framework::stream_and_collide_debug
(
    const std::vector<start_end_it_tuple> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &source, 
    std::vector<double> &destination,    
    const access_function access_function,
    const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
    const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
)
{
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    velocity current_velocity = {0,0};

    std::vector<double> densities(TOTAL_NODE_COUNT, -1);
    double current_density = 0;

    std::vector<double> current_distributions(DIRECTION_COUNT, 0);

    start_end_it_tuple bounds;

    std::cout << "\t TL stream and collide: initializations and declarations performed." << std::endl;

    // Global boundary update
    parallel_framework::emplace_bounce_back_values(bsi, source, access_function);

    // Buffer update
    for(auto buffer_index = 0; buffer_index < BUFFER_COUNT; ++buffer_index)
    {
        parallel_framework::copy_to_buffer(buffer_ranges[buffer_index], source, access_function);
    }

    for (auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
    {
        std::cout << "\t\033[33mPerforming iteration for subdomain " << subdomain << "\033[0m" << std::endl;
        std::cout << std::endl;

        bounds = fluid_nodes[subdomain];

        std::cout << "\t SOURCE before stream and collide: " << std::endl;
        to_console::buffered::print_distribution_values(source, access_function);

        std::cout << "DESTINATION as received by perform_tl_stream_and_collide: " << std::endl;
        to_console::buffered::print_distribution_values(destination, access_function);

        /* Streaming step */
        for(auto it = std::get<0>(bounds); it <= std::get<1>(bounds); ++it)
        {
            sequential_two_lattice::tl_stream(
                source, 
                destination, 
                access_function, 
                *it);
        }
        std::cout << "DESTINATION after streaming: " << std::endl;
        to_console::buffered::print_distribution_values(destination, access_function);

        /* Collision step */
            for(auto it = std::get<0>(bounds); it <= std::get<1>(bounds); ++it)
            {
                collision::perform_collision(
                    *it, 
                    destination, 
                    access_function, 
                    velocities,
                    densities);          
            }
        std::cout << "\t DESTINATION after collision: " << std::endl;
        to_console::buffered::print_distribution_values(destination, access_function);
    }

    parallel_framework::update_velocity_input_density_output(y_values, destination, velocities, densities, access_function);
    std::cout << "Updated inlet and outlet ghost nodes." <<std::endl;
    to_console::buffered::print_distribution_values(destination, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}
