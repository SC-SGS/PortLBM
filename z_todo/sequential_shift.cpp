#include "../include/sequential_shift.hpp"

#include <set>
#include <iostream>
#include <stdexcept>

/**
 * @brief Performs a combined collision and streaming step for the specified fluid node.
 * 
 * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
 * @param fluid_nodes         a vector containing the indices of all fluid nodes within the simulation domain.
 * @param bsi                 see documentation of border_swap_information
 * @param access_function     An access function from the namespace sequential_shift::access_functions.
 *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param iteration           the iteration the algorithm is currently processing
 */
sim_data_tuple sequential_shift::stream_and_collide
(
    std::vector<double> &distribution_values, 
    const std::vector<unsigned int> &fluid_nodes,
    const border_swap_information &bsi,
    const access_function access_function,
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
        write_offset = SHIFT_OFFSET;

        // Emplace bounce-back values
        bounce_back::emplace_bounce_back_values(bsi, distribution_values, access_function, read_offset);
        
        for(auto node = fluid_nodes.end() - 1; node >= fluid_nodes.begin(); --node)
        {
            sequential_shift::shift_stream(distribution_values, access_function, *node, read_offset, write_offset);
            sequential_shift::shift_collision(*node, distribution_values, access_function, velocities, densities, write_offset);
        }
    }
    else
    {
        read_offset = SHIFT_OFFSET;
        write_offset = 0;

        // Emplace bounce-back values
        bounce_back::emplace_bounce_back_values(bsi, distribution_values, access_function, read_offset);
 
        for(auto node = fluid_nodes.begin(); node < fluid_nodes.end(); ++node)
        {
            sequential_shift::shift_stream(distribution_values, access_function, *node, read_offset, write_offset);
            sequential_shift::shift_collision(*node, distribution_values, access_function, velocities, densities, write_offset);
        }
    }

    /* Update ghost nodes */
    sequential_shift::update_velocity_input_density_output(distribution_values, velocities, densities, access_function, write_offset);

    sim_data_tuple result{velocities, densities};
    return result;
}

/**
 * @brief Updates the ghost nodes that represent inlet and outlet edges.
 *        When updating, a velocity border condition will be considered for the input
 *        and a density border condition for the output.
 *        The corresponding values are constants defined in "defines.hpp".
 * 
 * @param distribution_values the updated distribution values will be written to this vector
 * @param velocities a vector containing the velocities of all nodes
 * @param densities a vector containing the densities of all nodes
 * @param access_function An access function from the namespace sequential_shift::access_functions.
 *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param offset the write offset of the current iteration
 */
void sequential_shift::update_velocity_input_density_output
(
    std::vector<double> &distribution_values, 
    std::vector<velocity> &velocities,
    std::vector<double> &densities, 
    const access_function access_function,
    const unsigned int offset
)
{

    std::vector<double> current_dist_vals(DIRECTION_COUNT, 0);
    int current_border_node = 0;
    velocity v = INLET_VELOCITY;
    double density = 0;

    for(auto y = 0; y < VERTICAL_NODES; ++y)
    {
        /* Update inlet nodes */
        current_border_node = lbm_access::get_node_index(0,y);
        current_dist_vals = maxwell_boltzmann_distribution(INLET_VELOCITY, INLET_DENSITY);
        lbm_access::set_distribution_values_of
        (
            current_dist_vals,
            distribution_values,
            current_border_node + offset,
            access_function
        );
        velocities[current_border_node] = macroscopic::flow_velocity(current_dist_vals);
        densities[current_border_node] = macroscopic::density(current_dist_vals);

        /* Update outlet nodes */
        current_border_node = lbm_access::get_node_index(HORIZONTAL_NODES - 1,y);
        v = macroscopic::flow_velocity(
            lbm_access::get_distribution_values_of(
                distribution_values, lbm_access::get_neighbor(current_border_node + offset, 3), access_function
            )
        );
        current_dist_vals = maxwell_boltzmann_distribution(v, OUTLET_DENSITY);
        lbm_access::set_distribution_values_of
        (
            current_dist_vals,
            distribution_values,
            current_border_node + offset,
            access_function
        );

        velocities[current_border_node] = v;
        densities[current_border_node] = OUTLET_DENSITY;
    }

    /* Restore correctness of upper and lower outlet node (not guaranteed with shift algorithm)*/
    unsigned int update_node = 0;
    std::vector<double> current_distributions = maxwell_boltzmann_distribution(OUTLET_VELOCITY, OUTLET_DENSITY);
    unsigned int x =  HORIZONTAL_NODES - 1;

    int y = 0;
    update_node = lbm_access::get_node_index(x,y);
    lbm_access::set_distribution_values_of(current_distributions, distribution_values, update_node + offset, access_function);
    
    y = VERTICAL_NODES - 1;
    update_node = lbm_access::get_node_index(x,y);
    lbm_access::set_distribution_values_of(current_distributions, distribution_values, update_node + offset, access_function);
}

/**
 * @brief Performs the parallel shift algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes         a vector containing the indices of all fluid nodes within the simulation domain.
 * @param bsi                 see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
 * @param access_function     An access function from the namespace sequential_shift::access_functions.
 *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param iterations          this many iterations will be performed
 */
void sequential_shift::run
(  
    std::vector<unsigned int> &fluid_nodes,       
    std::vector<double> &values, 
    border_swap_information &bsi,
    access_function access_function,
    unsigned int iterations
)
{
    std::vector<sim_data_tuple>result(
        iterations,
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    for(auto time = 0; time < iterations; ++time)
    {
        result[time] = sequential_shift::stream_and_collide(values, fluid_nodes, bsi, access_function, time); 
    }

    if(RESULTS_TO_CSV)
    {
        sim_data_to_csv(result, "results.csv");
    }
}

/**
 * @brief Performs the parallel shift algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes         a vector containing the indices of all fluid nodes within the simulation domain.
 * @param bsi                 see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
 * @param access_function     An access function from the namespace sequential_shift::access_functions.
 *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 * @param iterations          this many iterations will be performed
 */
void sequential_shift::run_debug
(  
    std::vector<unsigned int> &fluid_nodes,       
    std::vector<double> &values, 
    border_swap_information &bsi,
    access_function access_function,
    unsigned int iterations
)
{
    to_console::print_run_greeting("sequential shift algorithm", iterations);

    std::vector<sim_data_tuple>result(
        iterations,
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    for(auto time = 0; time < iterations; ++time)
    {
        std::cout << "\033[33mIteration " << time << ":\033[0m" << std::endl;

        result[time] = sequential_shift::stream_and_collide(values, fluid_nodes, bsi, access_function, time);

        std::cout << "\tFinished iteration " << time << std::endl;
    }

    if(RESULTS_TO_CSV)
    {
        sim_data_to_csv(result, "results.csv");
    }

    to_console::print_simulation_results(result);
    std::cout << "All done, exiting simulation. " << std::endl;
}

/**
 * @brief Create an example domain for testing purposes. The domain is a rectangle with
 *        dimensions specified in the defines file where the outermost nodes are ghost nodes.
 *        The upper and lower ghost nodes are solid whereas the leftmost and rightmost columns are fluid
 *        nodes that mark the inlet and outlet respectively.
 *        Notice that all data will be written to the parameters which are assumed to be empty initially.
 * 
 * @param distribution_values a vector containing all distribution values.
 * @param nodes a vector containing all node indices, including those of solid nodes and ghost nodes.
 * @param fluid_nodes a vector containing the indices of all fluid nodes.
 * @param phase_information a vector containing the phase information of all nodes where true means solid.
 * @param access_function An access function from the namespace sequential_shift::access_functions.
 *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
 */
void sequential_shift::setup_example_domain
(
    std::vector<double> &distribution_values,
    std::vector<unsigned int> &nodes,
    std::vector<unsigned int> &fluid_nodes,
    std::vector<bool> &phase_information,
    access_function access_function
)
{
    distribution_values.assign((TOTAL_NODE_COUNT + SHIFT_OFFSET) * DIRECTION_COUNT, 0); 
    std::vector<double> regular_values = maxwell_boltzmann_distribution(VELOCITY_VECTORS.at(4), 1);
    std::vector<double> inlet_values = maxwell_boltzmann_distribution(INLET_VELOCITY, INLET_DENSITY);
    std::vector<double> outlet_values = maxwell_boltzmann_distribution(OUTLET_VELOCITY, OUTLET_DENSITY);
    int node = 0;

    /* Set all nodes for direct access */
    for(auto i = 0; i < TOTAL_NODE_COUNT; ++i)
    {
        nodes.push_back(i);
    }

    /* Set up distribution values */
    for(auto y = 0; y < VERTICAL_NODES; ++y)
    {
        for(auto x = 0; x < HORIZONTAL_NODES; ++x)
        {
            node = lbm_access::get_node_index(x,y);
            if(x == 0)
            {
                lbm_access::set_distribution_values_of(inlet_values, distribution_values, node, access_function);
            }
            else if(x == HORIZONTAL_NODES - 1)
            {
                lbm_access::set_distribution_values_of(outlet_values, distribution_values, node, access_function);
            }
            else
            {
                lbm_access::set_distribution_values_of(regular_values, distribution_values, node, access_function);
            }
        }
    }

    /* Phase information vector */
    phase_information.assign(TOTAL_NODE_COUNT, false);
    for(auto x = 0; x < HORIZONTAL_NODES; ++x)
    {
        phase_information[lbm_access::get_node_index(x,0)] = true;
        phase_information[lbm_access::get_node_index(x,VERTICAL_NODES - 1)] = true;
    }

    /* Set up vector containing fluid nodes within the simulation domain. */
    for(auto y = 1; y < VERTICAL_NODES - 1; ++y)
    {
        for(auto x = 1; x < HORIZONTAL_NODES - 1; ++x)
        {
            fluid_nodes.push_back(lbm_access::get_node_index(x,y));
        }
    }
}
