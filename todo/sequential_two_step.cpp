#include "../include/sequential_two_step.hpp"

#include <iostream>

/**
 * @brief Performs the streaming step for all fluid nodes within the simulation domain.
 *        Notice that each node is streaming outward.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param distribution_values a vector containing all distribution values
 * @param access_function the access to node values will be performed according to this access function.
 */
void sequential_two_step::perform_stream
(
    const std::vector<unsigned int> &fluid_nodes, 
    std::vector<double> &distribution_values, 
    const access_function access_function
)
{
    // All directions that require left-to-right and/or bottom-to-top node iteration order
    for(auto it = fluid_nodes.begin(); it < fluid_nodes.end(); ++it)
    {
        distribution_values[access_function(lbm_access::get_neighbor(*it, 0), 0)] = distribution_values[access_function(*it, 0)];
        distribution_values[access_function(lbm_access::get_neighbor(*it, 1), 1)] = distribution_values[access_function(*it, 1)];
        distribution_values[access_function(lbm_access::get_neighbor(*it, 2), 2)] = distribution_values[access_function(*it, 2)];
        distribution_values[access_function(lbm_access::get_neighbor(*it, 3), 3)] = distribution_values[access_function(*it, 3)];
    }

    // All directions that require right-to-left and/or top-to-bottom node iteration order
    for(auto it = fluid_nodes.end() - 1; it >= fluid_nodes.begin(); --it)
    {
        distribution_values[access_function(lbm_access::get_neighbor(*it, 5), 5)] = distribution_values[access_function(*it, 5)];
        distribution_values[access_function(lbm_access::get_neighbor(*it, 6), 6)] = distribution_values[access_function(*it, 6)];
        distribution_values[access_function(lbm_access::get_neighbor(*it, 7), 7)] = distribution_values[access_function(*it, 7)];
        distribution_values[access_function(lbm_access::get_neighbor(*it, 8), 8)] = distribution_values[access_function(*it, 8)];
    }
}

/**
 * @brief Performs the streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param bsi see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values
 * @param access_function the access to node values will be performed according to this access function.
 * @return sim_data_tuple see documentation of sim_data_tuple
 */
sim_data_tuple sequential_two_step::stream_and_collide
(
    const std::vector<unsigned int> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &distribution_values,    
    const access_function access_function
)
{
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    /* Streaming */
    sequential_two_step::perform_stream(fluid_nodes, distribution_values, access_function);

    /* Perform bounce-back using ghost nodes */
    bounce_back::perform_boundary_update(bsi, distribution_values, access_function);

    /* Perform inflow and outflow using ghost nodes */
    boundary_conditions::ghost_stream_inout(distribution_values, access_function);

    /* Perform collision for all fluid nodes */
    for(const auto fluid_node : fluid_nodes)
    {
        collision::perform_collision(
            fluid_node, 
            distribution_values, 
            access_function, 
            velocities,
            densities);
    }

    /* Update ghost nodes */
    boundary_conditions::update_velocity_input_density_output(distribution_values, velocities, densities, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs the streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 *        This variant will print several debug comments to the console.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param bsi see documentation of border_swap_information
 * @param distribution_values a vector containing all distribution values
 * @param access_function the access to node values will be performed according to this access function.
 * @return sim_data_tuple see documentation of sim_data_tuple
 */
sim_data_tuple sequential_two_step::stream_and_collide_debug
(
    const std::vector<unsigned int> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &distribution_values,    
    const access_function access_function
)
{
    std::cout << "Distribution values before stream and collide: " << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    /* Streaming */
    sequential_two_step::perform_stream(fluid_nodes, distribution_values, access_function);
    std::cout << "\t Distribution values after streaming:" << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    /* Perform bounce-back using ghost nodes */
    bounce_back::perform_boundary_update(bsi, distribution_values, access_function);
    std::cout << "\t Distribution values after bounce-back update:" << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    /* Perform inflow and outflow using ghost nodes */
    std::cout << "Performing ghost stream inout " << std::endl;
    boundary_conditions::ghost_stream_inout(distribution_values, access_function);
    std::cout << "\t Distribution values after inflow and outflow via ghost nodes:" << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    /* Perform collision for all fluid nodes */
    for(const auto fluid_node : fluid_nodes)
    {
        collision::perform_collision(
            fluid_node, 
            distribution_values, 
            access_function, 
            velocities,
            densities);
    }
    std::cout << "\t Distribution values after collision:" << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    /* Update ghost nodes */
    boundary_conditions::update_velocity_input_density_output(distribution_values, velocities, densities, access_function);
    std::cout << "Distribution values after ghost node update: " << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs the sequential two-step algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param distribution_values the vector containing the distribution values of all nodes
 * @param bsi see documentation of border_swap_information
 * @param access_function the access function according to which the values are to be accessed
 * @param iterations this many iterations will be performed
 */
void sequential_two_step::run
(  
    std::vector<unsigned int> &fluid_nodes,       
    std::vector<double> &distribution_values, 
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
        result[time] = sequential_two_step::stream_and_collide
        (
            fluid_nodes, 
            bsi, 
            distribution_values, 
            access_function
        );
    }

    if(RESULTS_TO_CSV)
    {
        sim_data_to_csv(result, "results.csv");
    }
}

/**
 * @brief Performs the sequential two-step algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param distribution_values the vector containing the distribution values of all nodes
 * @param bsi see documentation of border_swap_information
 * @param access_function the access function according to which the values are to be accessed
 * @param iterations this many iterations will be performed
 */
void sequential_two_step::run_debug
(  
    std::vector<unsigned int> &fluid_nodes,       
    std::vector<double> &distribution_values, 
    border_swap_information &bsi,
    access_function access_function,
    unsigned int iterations
)
{
    to_console::print_run_greeting("sequential two-step algorithm", iterations);

    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    for(auto time = 0; time < iterations; ++time)
    {
        std::cout << "\033[33mIteration " << time << ":\033[0m";
        result[time] = sequential_two_step::stream_and_collide_debug
        (
            fluid_nodes, 
            bsi, 
            distribution_values, 
            access_function
        );
        std::cout << "\tFinished iteration " << time << std::endl;
    }

    if(RESULTS_TO_CSV)
    {
        sim_data_to_csv(result, "results.csv");
    }

    to_console::print_simulation_results(result);
    std::cout << "All done, exiting simulation. " << std::endl;
}
