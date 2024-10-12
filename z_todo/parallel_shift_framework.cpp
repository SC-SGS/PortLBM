#include "../include/parallel_shift_framework.hpp"

#include <iostream>

#include <hpx/algorithm.hpp>

/**
 * @brief Performs the parallel shift algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
 * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
 *                            see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
 * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
 *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param iterations          this many iterations will be performed
 */
void parallel_shift_framework::run
(  
    const std::vector<start_end_it_tuple> &fluid_nodes,       
    const std::vector<border_swap_information> &boundary_nodes,
    std::vector<double> &distribution_values,   
    const access_function access_function,
    const unsigned int iterations
)
{
    std::vector<std::tuple<unsigned int, unsigned int>> buffer_ranges;
    for (auto buffer_index = 0; buffer_index < BUFFER_COUNT; ++buffer_index)
    {
        buffer_ranges.push_back(parallel_framework::get_buffer_node_range(buffer_index));
    }

    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0))
    );

    /* Parallelization framework */
    for(auto time = 0; time < iterations; ++time)
    {
        result[time] = parallel_shift_framework::stream_and_collide
        (fluid_nodes, boundary_nodes, distribution_values, access_function, buffer_ranges, time);
    }

    if(RESULTS_TO_CSV)
    {
        parallel_domain_sim_data_to_csv(result, "results.csv");
    }
}

/**
 * @brief Performs the parallel shift algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
 * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
 *                            see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
 * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
 *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param iterations          this many iterations will be performed
 */
void parallel_shift_framework::run_debug
(  
    const std::vector<start_end_it_tuple> &fluid_nodes,       
    const std::vector<border_swap_information> &boundary_nodes,
    std::vector<double> &distribution_values,   
    const access_function access_function,
    const unsigned int iterations
)
{
    to_console::print_run_greeting("parallel shift algorithm", iterations);

    std::vector<std::tuple<unsigned int, unsigned int>> buffer_ranges;
    for (auto buffer_index = 0; buffer_index < BUFFER_COUNT; ++buffer_index)
    {
        buffer_ranges.push_back(parallel_framework::get_buffer_node_range(buffer_index));
    }

    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0))
    );

    /* Parallelization framework */
    for(auto time = 0; time < iterations; ++time)
    {
        std::cout << "\033[33mIteration " << time << ":\033[0m";
        
        result[time] = parallel_shift_framework::stream_and_collide_debug
        (fluid_nodes, boundary_nodes, distribution_values, access_function, buffer_ranges, time);

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
 * @brief Performs a combined collision and streaming step for the specified fluid node.
 * 
 * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
 * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
 *                            see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
 * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
 *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param buffer_ranges       a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
 * @param iteration           the iteration the algorithm is currently processing
 */
sim_data_tuple parallel_shift_framework::stream_and_collide
(
    const std::vector<start_end_it_tuple> &fluid_nodes,
    const std::vector<border_swap_information> &bsi,
    std::vector<double> &distribution_values, 
    const access_function access_function,
    const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges,
    const unsigned int iteration
)
{
    unsigned int read_offset = 0;
    unsigned int write_offset = 0;
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    if((iteration % 2) == 0)
    {
        read_offset = 0;
        write_offset = (SHIFT_OFFSET);

        // Emplace bounce-back values
        hpx::experimental::for_loop
        (
            hpx::execution::par, 0, SUBDOMAIN_COUNT, 
            [&](unsigned int subdomain)
            {
                unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
                parallel_shift_framework::emplace_bounce_back_values(bsi[subdomain], distribution_values, access_function, subdomain_offset + read_offset);
            }
        );

        // Buffer update
        hpx::experimental::for_loop
        (
            hpx::execution::par, 0, BUFFER_COUNT, 
            [&](unsigned int buffer)
            {
                int buffer_offset = (buffer + 1) * (SHIFT_OFFSET);
                parallel_shift_framework::buffer_update_even_time_step(buffer_ranges[buffer], distribution_values, access_function, buffer_offset);
            }
        );

        hpx::experimental::for_loop
        (
            hpx::execution::par, 0, SUBDOMAIN_COUNT, 
            [&](unsigned int subdomain)
            {
                unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
                for(auto it = std::get<1>(fluid_nodes[subdomain]); it >= std::get<0>(fluid_nodes[subdomain]); --it)
                {
                    sequential_shift::shift_stream(distribution_values, access_function, *it, read_offset + subdomain_offset, write_offset + subdomain_offset);
                    parallel_shift_framework::perform_collision(*it, distribution_values, access_function, velocities, densities, write_offset + subdomain_offset);
                }
            }
        );
    }
    else
    {
        read_offset = (SHIFT_OFFSET);
        write_offset = 0;

        // Emplace bounce-back values
        hpx::experimental::for_loop
        (
            hpx::execution::par, 0, SUBDOMAIN_COUNT, 
            [&](unsigned int subdomain)
            {
                unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
                parallel_shift_framework::emplace_bounce_back_values(bsi[subdomain], distribution_values, access_function, subdomain_offset + read_offset);
            }
        );

        // Buffer update
        hpx::experimental::for_loop
        (
            hpx::execution::par, 0, BUFFER_COUNT, 
            [&](unsigned int buffer)
            {
                int buffer_offset = (buffer + 1) * (SHIFT_OFFSET);
                parallel_shift_framework::buffer_update_odd_time_step(buffer_ranges[buffer], distribution_values, access_function, buffer_offset);
            }
        );

        hpx::experimental::for_loop
        (
            hpx::execution::par, 0, SUBDOMAIN_COUNT, 
            [&](unsigned int subdomain)
            {
                unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
                for(auto it = std::get<0>(fluid_nodes[subdomain]); it <= std::get<1>(fluid_nodes[subdomain]); ++it)
                {
                    sequential_shift::shift_stream(distribution_values, access_function, *it, read_offset + subdomain_offset, write_offset + subdomain_offset);
                    parallel_shift_framework::perform_collision(*it, distribution_values, access_function, velocities, densities, write_offset + subdomain_offset);
                }
            }
        );
    }

    /* Update ghost nodes */
    parallel_shift_framework::update_velocity_input_density_output(distribution_values, velocities, densities, access_function, write_offset);
    
    sim_data_tuple result{velocities, densities};
    return result;
}

/**
 * @brief Performs a combined collision and streaming step for the specified fluid node.
 *        This algorithm prints out various debug comments.
 * 
 * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
 * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
 *                            see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
 * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
 *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param buffer_ranges       a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
 * @param iteration           the iteration the algorithm is currently processing
 */
sim_data_tuple parallel_shift_framework::stream_and_collide_debug
(
    const std::vector<start_end_it_tuple> &fluid_nodes,
    const std::vector<border_swap_information> &bsi,
    std::vector<double> &distribution_values, 
    const access_function access_function,
    const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges,
    const unsigned int iteration
)
{
    unsigned int current_node = 0;
    unsigned int read_offset = 0;
    unsigned int write_offset = 0;
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    if((iteration % 2) == 0)
    {
        read_offset = 0;
        write_offset = (SHIFT_OFFSET);

        std::cout << "Beginning with iteration " << iteration << std::endl;
        std::cout << "Is even?: " << true << std::endl;
        std::cout << "Read offset: " << read_offset << std::endl;
        std::cout << "Write offset: " << write_offset << std::endl;
        std::cout << std::endl;

        std::cout << "Distributions before stream and collide:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, read_offset, buffer_ranges);

        // Emplace bounce-back values
        for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
        {
            std::cout << "Proceeding with subdomain " << subdomain << std::endl;
            unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
            parallel_shift_framework::emplace_bounce_back_values(bsi[subdomain], distribution_values, access_function, subdomain_offset + read_offset);
        }
        std::cout << "Distribution values after bounce-back update:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, read_offset, buffer_ranges);

        // Buffer update
        for(auto buffer = 0; buffer < BUFFER_COUNT; ++buffer)
        {
            int buffer_offset = (buffer+1) * (SHIFT_OFFSET);
            parallel_shift_framework::buffer_update_even_time_step(buffer_ranges[buffer], distribution_values, access_function, buffer_offset);
        }

        std::cout << "Distribution values after buffer update:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, read_offset, buffer_ranges);

        for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
        {
            unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
            for(auto it = std::get<1>(fluid_nodes[subdomain]); it >= std::get<0>(fluid_nodes[subdomain]); --it)
            {
                sequential_shift::shift_stream(distribution_values, access_function, *it, read_offset + subdomain_offset, write_offset + subdomain_offset);
            }
        }

        std::cout << "Distribution values after streaming:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, write_offset, buffer_ranges);

        for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
        {
            unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
            for(auto it = std::get<1>(fluid_nodes[subdomain]); it >= std::get<0>(fluid_nodes[subdomain]); --it)
            {
                parallel_shift_framework::perform_collision(*it, distribution_values, access_function, velocities, densities, write_offset + subdomain_offset);
            }
        }

        std::cout << "Distribution values after collision:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, write_offset, buffer_ranges);
    }
    else
    {
        read_offset = (SHIFT_OFFSET);
        write_offset = 0;

        std::cout << "Beginning with iteration " << iteration << std::endl;
        std::cout << "Is even?: " << false << std::endl;
        std::cout << "Read offset: " << read_offset << std::endl;
        std::cout << "Write offset: " << write_offset << std::endl;
        std::cout << std::endl;

        parallel_shift_framework::print_distribution_values(distribution_values, access_function, read_offset, buffer_ranges);

        // Emplace bounce-back values
        for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
        {
            unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
            parallel_shift_framework::emplace_bounce_back_values(bsi[subdomain], distribution_values, access_function, subdomain_offset + read_offset);
        }

        std::cout << "Distribution values after bounce-back update:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, read_offset, buffer_ranges);

        // Buffer update
        for(auto buffer = 0; buffer < BUFFER_COUNT; ++buffer)
        {
            int buffer_offset = (buffer+1) * (SHIFT_OFFSET);
            parallel_shift_framework::buffer_update_odd_time_step(buffer_ranges[buffer], distribution_values, access_function, buffer_offset);
        }

        std::cout << "Distribution values after buffer update:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, read_offset, buffer_ranges);

        for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
        {
            unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
            for(auto it = std::get<0>(fluid_nodes[subdomain]); it <= std::get<1>(fluid_nodes[subdomain]); ++it)
            {
                sequential_shift::shift_stream(distribution_values, access_function, *it, read_offset + subdomain_offset, write_offset + subdomain_offset);
            }
        }

        std::cout << "Distribution values after streaming:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, write_offset, buffer_ranges);

        for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
        {
            unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
            for(auto it = std::get<0>(fluid_nodes[subdomain]); it <= std::get<1>(fluid_nodes[subdomain]); ++it)
            {
                parallel_shift_framework::perform_collision(*it, distribution_values, access_function, velocities, densities, write_offset + subdomain_offset);
            }
        }

        std::cout << "Distribution values after collision:" << std::endl;
        parallel_shift_framework::print_distribution_values(distribution_values, access_function, write_offset, buffer_ranges);
    }

    /* Update ghost nodes */
    parallel_shift_framework::update_velocity_input_density_output(distribution_values, velocities, densities, access_function, write_offset);

    std::cout << "Distribution values after input update" << std::endl;
    parallel_shift_framework::print_distribution_values(distribution_values, access_function, write_offset, buffer_ranges);
    
    sim_data_tuple result{velocities, densities};
    return result;
}

/**
 * @brief Performs the pre-iteration buffer initialization for the buffer with the specified boundaries for an even time step.
 *        Southbound values from the nodes above remain within the buffer whereas northbound values from the nodes below will
 *        instead be written directly into the overlapping shift area.
 * 
 * @param buffer_bounds a tuple containing the first and last index of the buffer
 * @param distribution_values a vector containing all distribution values
 * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
 *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param buffer_offset the shift-related offset of this buffer
 */
void parallel_shift_framework::buffer_update_even_time_step
(
    const std::tuple<unsigned int, unsigned int> &buffer_bounds,
    std::vector<double> &distribution_values,
    access_function access_function,
    const unsigned int buffer_offset
)
{
    unsigned int start = std::get<0>(buffer_bounds);
    unsigned int end = std::get<1>(buffer_bounds);

    for(auto buffer_node = start; buffer_node <= end; ++buffer_node)
    {
        for(auto direction : {6,7,8})
        {
            distribution_values[access_function(buffer_node + buffer_offset, direction)] = 
            distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 1) + buffer_offset - (SHIFT_OFFSET), direction)];
        }
        for(auto direction : {0,1,2})
        {
            distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 1) + buffer_offset - 1, direction)] = 
            distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 7) + buffer_offset, direction)];
        }
    }
}

/**
 * @brief Performs the pre-iteration buffer initialization for the buffer with the specified boundaries for an odd time step.
 *        Northbound values from the nodes below remain within the buffer whereas southbound values from the nodes above will
 *        instead be written directly into the overlapping shift area.
 * 
 * @param buffer_bounds a tuple containing the first and last index of the buffer
 * @param distribution_values a vector containing all distribution values
 * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
 *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param buffer_offset the shift-related offset of this buffer
 */
void parallel_shift_framework::buffer_update_odd_time_step
(
    const std::tuple<unsigned int, unsigned int> &buffer_bounds,
    std::vector<double> &distribution_values,
    access_function access_function,
    const unsigned int buffer_offset
)
{
    unsigned int start = std::get<0>(buffer_bounds);
    unsigned int end = std::get<1>(buffer_bounds);

    for(auto buffer_node = start; buffer_node <= end; ++buffer_node)
    {
        for(auto direction : {6,7,8})
        {
            distribution_values[access_function(buffer_node + buffer_offset + SHIFT_OFFSET, direction)] = 
            distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 1) + buffer_offset, direction)];
        }
        for(auto direction : {0,1,2})
        {
            distribution_values[access_function(buffer_node + buffer_offset, direction)] =
            distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 7) + buffer_offset + SHIFT_OFFSET, direction)];
        }
    }
}

/**
 * @brief Sets up a suitable domain for parallel computation. The domain is a rectangle with
 *        dimensions specified in the defines file where the outermost nodes are ghost nodes.
 *        The upper and lower ghost nodes are solid whereas the leftmost and rightmost columns are fluid
 *        nodes that mark the inlet and outlet respectively.
 *        Notice that all data will be written to the parameters which are assumed to be empty initially.
 * 
 * @param distribution_values a vector containing all distribution values.
 * @param nodes a vector containing all node indices, including those of solid nodes and ghost nodes.
 * @param fluid_nodes a vector containing the indices of all fluid nodes.
 * @param phase_information a vector containing the phase information of all nodes where true means solid.
 * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
 *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 */
void parallel_shift_framework::setup_parallel_domain
(    
    std::vector<double> &distribution_values,
    std::vector<unsigned int> &nodes,
    std::vector<unsigned int> &fluid_nodes,
    std::vector<bool> &phase_information,
    access_function access_function
)
{
    distribution_values.assign(SHIFT_DISTRIBUTION_VALUE_COUNT * DIRECTION_COUNT, 0); 
    std::vector<double> regular_values = maxwell_boltzmann_distribution(VELOCITY_VECTORS.at(4), 1);
    std::vector<double> inlet_values = maxwell_boltzmann_distribution(INLET_VELOCITY, INLET_DENSITY);
    std::vector<double> outlet_values = maxwell_boltzmann_distribution(OUTLET_VELOCITY, OUTLET_DENSITY);

    /* Set up vector of all nodes for direct access */
    for(auto i = 0; i < TOTAL_NODE_COUNT; ++i)
    {
        nodes.push_back(i);
    }

    hpx::experimental::for_loop
    (
        hpx::execution::par, 0, SUBDOMAIN_COUNT, 
        [&](unsigned int subdomain)
        {
            unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
            int last_node = 0;
            for(auto y = subdomain * SUBDOMAIN_HEIGHT + subdomain; y < (subdomain + 1) * SUBDOMAIN_HEIGHT + subdomain; ++y)
            {
                for(auto x = 0; x < HORIZONTAL_NODES; ++x)
                {
                    last_node = lbm_access::get_node_index(x,y);
                    if(x == 0)
                    {
                        lbm_access::set_distribution_values_of(inlet_values, distribution_values, last_node + subdomain_offset, access_function);
                    }
                    else if(x == HORIZONTAL_NODES - 1)
                    {
                        lbm_access::set_distribution_values_of(outlet_values, distribution_values, last_node + subdomain_offset, access_function);
                    }
                    else
                    {
                        lbm_access::set_distribution_values_of(regular_values, distribution_values, last_node + subdomain_offset, access_function);
                    }
                }
            }
        }
    );

    /* Phase information vector */
    phase_information.assign(TOTAL_NODE_COUNT, false);
        hpx::experimental::for_loop
    (
        hpx::execution::par, 0, HORIZONTAL_NODES, 
        [&](unsigned int x)
        {
            phase_information[lbm_access::get_node_index(x,0)] = true;
            phase_information[lbm_access::get_node_index(x,VERTICAL_NODES - 1)] = true;
        }
    );

    /* Fluid nodes vector */
    for(auto y = 1; y < VERTICAL_NODES - 1; ++y)
    {
        for(auto x = 1; x < HORIZONTAL_NODES - 1; ++x)
        {
            fluid_nodes.push_back(lbm_access::get_node_index(x,y));
        }
    }
}

/**
 * @brief Updates the ghost nodes that represent inlet and outlet edges.
 *        When updating, a velocity border condition will be considered for the input
 *        and a density border condition for the output.
 *        The corresponding values are constants defined in "../include/"defines.hpp".
 * 
 * @param distribution_values the updated distribution values will be written to this vector
 * @param velocities a vector containing the velocities of all nodes
 * @param densities a vector containing the densities of all nodes
 * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
 *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param offset the write offset of the current iteration
 */
void parallel_shift_framework::update_velocity_input_density_output
(
    std::vector<double> &distribution_values,
    std::vector<velocity> &velocities,
    std::vector<double> &densities, 
    const access_function access_function,
    const unsigned int offset
)
{
        hpx::experimental::for_loop
        (
            hpx::execution::par, 0, SUBDOMAIN_COUNT, 
            [&](unsigned int subdomain)
            {
                unsigned int subdomain_offset = subdomain * (SHIFT_OFFSET);
                int last_node = 0;
                std::vector<double> current_dist_vals(DIRECTION_COUNT, 0);
                int current_border_node = 0;
                velocity v = {0,0};

                for(auto y = subdomain * SUBDOMAIN_HEIGHT + subdomain; y < (subdomain + 1) * SUBDOMAIN_HEIGHT + subdomain; ++y)
                {
                    /* Update inlet nodes */
                    current_border_node = lbm_access::get_node_index(0,y);
                    current_dist_vals = maxwell_boltzmann_distribution(INLET_VELOCITY, INLET_DENSITY);
                    lbm_access::set_distribution_values_of
                    (
                        current_dist_vals,
                        distribution_values,
                        current_border_node + offset + subdomain_offset,
                        access_function
                    );
                    velocities[current_border_node] = macroscopic::flow_velocity(current_dist_vals);
                    densities[current_border_node] = macroscopic::density(current_dist_vals);

                    /* Update outlet nodes */
                    current_border_node = lbm_access::get_node_index(HORIZONTAL_NODES - 1,y);
                    v = macroscopic::flow_velocity(
                        lbm_access::get_distribution_values_of(
                            distribution_values, lbm_access::get_neighbor(current_border_node + offset + subdomain_offset, 3), access_function
                        )
                    );
                    current_dist_vals = maxwell_boltzmann_distribution(v, OUTLET_DENSITY);
                    lbm_access::set_distribution_values_of
                    (
                        current_dist_vals,
                        distribution_values,
                        current_border_node + offset + subdomain_offset,
                        access_function
                    );

                    velocities[current_border_node] = v;
                    densities[current_border_node] = OUTLET_DENSITY;
                }
            }
        );

        /* Restore correctness of upper and lower outlet node (not guaranteed with shift algorithm)*/
        unsigned int update_node = 0;
        std::vector<double> current_distributions = maxwell_boltzmann_distribution(OUTLET_VELOCITY, OUTLET_DENSITY);
        unsigned int x =  HORIZONTAL_NODES - 1;

        int y = 0;
        update_node = lbm_access::get_node_index(x,y);
        lbm_access::set_distribution_values_of(current_distributions, distribution_values, update_node + offset, access_function);
        
        y = VERTICAL_NODES - 1;
        update_node = lbm_access::get_node_index(x,y) + (BUFFER_COUNT) * (SHIFT_OFFSET);
        lbm_access::set_distribution_values_of(current_distributions, distribution_values, update_node + offset, access_function);
}

/**
 * @brief Performs an outstream step for all border nodes in the directions where they border non-inout ghost nodes.
 *        The distribution values will be stored in the ghost nodes in inverted order such that
 *        after this method is executed, the border nodes can be treated like regular nodes when performing an instream.
 * 
 * @param bsi a border_swap_information generated by retrieve_border_swap_info
 * @param distribution_values a vector containing the distribution values of all nodes
 * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
 *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param read_offset offset for shift algorithm, leave zero for all other algorithms
 */
void parallel_shift_framework::emplace_bounce_back_values
(
    const border_swap_information &bsi,
    std::vector<double> &distribution_values,
    const access_function access_function,
    const unsigned int read_offset
)
{
    hpx::for_each
    (
        hpx::execution::par, 
        bsi.begin(), 
        bsi.end(), 
        [&](const std::vector<unsigned int>& fluid_node)
        {
            for(auto direction_iterator = fluid_node.begin()+1; direction_iterator < fluid_node.end(); ++direction_iterator) 
            {
                distribution_values[
                    access_function(lbm_access::get_neighbor(fluid_node[0] + read_offset, *direction_iterator), invert_direction(*direction_iterator))] = 
                    distribution_values[access_function(fluid_node[0] + read_offset, *direction_iterator)];
            }
        }
    );
}