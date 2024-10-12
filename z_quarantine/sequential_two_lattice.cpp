#include "../include/sequential_two_lattice.hpp"

#include <iostream>
#include "../include/file_interaction.hpp"

/**
 * @brief Performs the combined streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 * 
 * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain.
 * @param bsi see documentation of border_swap_information
 * @param source a vector containing the distribution values of the previous time step
 * @param destination the distribution values will be written to this vector after performing both steps.
 * @param access_function the function used to access the distribution values
 * @return see documentation of sim_data_tuple
 */
sim_data_tuple sequential_two_lattice::stream_and_collide
(
    const std::vector<unsigned int> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &source, 
    std::vector<double> &destination,    
    const access_function access_function
)
{
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    /* Boundary node treatment */
    bounce_back::emplace_bounce_back_values(bsi, source, access_function);

    /* Combined stream and collision step */
    for(const auto fluid_node : fluid_nodes)
    {
        sequential_two_lattice::tl_stream(
            source, 
            destination, 
            access_function, 
            fluid_node);
        
        collision::perform_collision(
            fluid_node, 
            destination, 
            access_function, 
            velocities,
            densities);
    }

    boundary_conditions::update_velocity_input_density_output(destination, velocities, densities, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs the combined streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 *        This variant of the combined streaming and collision step will print several debug comments to the console.
 * 
 * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain.
 * @param bsi see documentation of border_swap_information
 * @param source a vector containing the distribution values of the previous time step
 * @param destination the distribution values will be written to this vector after performing both steps.
 * @param access_function the function used to access the distribution values
 * @return see documentation of sim_data_tuple
 */
sim_data_tuple sequential_two_lattice::stream_and_collide_debug
(
    const std::vector<unsigned int> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &source, 
    std::vector<double> &destination,    
    const access_function access_function
)
{
    std::cout << "\t SOURCE before stream and collide: " << std::endl;
    to_console::print_distribution_values(source, access_function);

    std::cout << "DESTINATION as received by perform_tl_stream_and_collide: " << std::endl;
    to_console::print_distribution_values(destination, access_function);

    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);

    std::cout << "\t TL stream and collide: initializations and declarations performed." << std::endl;

    /* Boundary node treatment */
    bounce_back::emplace_bounce_back_values(bsi, source, access_function);
    std::cout << "SOURCE after emplace bounce-back values: " << std::endl;
    to_console::print_distribution_values(source, access_function);

    /* Streaming step */
    for(const auto fluid_node : fluid_nodes)
    {
        sequential_two_lattice::tl_stream(
            source, 
            destination, 
            access_function, 
            fluid_node);
    }
    std::cout << "DESTINATION after streaming: " << std::endl;
    to_console::print_distribution_values(destination, access_function);

    /* Collision step */
    for(const auto fluid_node : fluid_nodes)
    {   
        collision::perform_collision(
            fluid_node, 
            destination, 
            access_function, 
            velocities,
            densities);
    }
    std::cout << "\t DESTINATION after collision: " << std::endl;
    to_console::print_distribution_values(destination, access_function);

    boundary_conditions::update_velocity_input_density_output(destination, velocities, densities, access_function);
    std::cout << "Updated inlet and outlet ghost nodes." <<std::endl;
    to_console::print_distribution_values(destination, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs the sequential two-lattice algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param boundary_nodes see documentation of border_swap_information
 * @param distribution_values_0 source for even time steps and destination for odd time steps
 * @param distribution_values_1 source for odd time steps and destination for even time steps
 * @param access_function the access function according to which distribution values are to be accessed
 * @param iterations this many iterations will be performed
 */
void sequential_two_lattice::run
(  
    const std::vector<unsigned int> &fluid_nodes,       
    const border_swap_information &boundary_nodes,
    std::vector<double> &distribution_values_0, 
    std::vector<double> &distribution_values_1,   
    const access_function access_function,
    const unsigned int iterations
)
{
    std::vector<double> temp;
    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    for(auto time = 0; time < iterations; ++time)
    {
        result[time] = sequential_two_lattice::stream_and_collide
        (
            fluid_nodes, 
            boundary_nodes, 
            distribution_values_0, 
            distribution_values_1, 
            access_function
        );     

        temp = std::move(distribution_values_0);
        distribution_values_0 = std::move(distribution_values_1);
        distribution_values_1 = std::move(temp);
    }

    if(RESULTS_TO_CSV)
    {
        sim_data_to_csv(result, "results.csv");
    }
}

/**
 * @brief Performs the sequential two-lattice algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param boundary_nodes see documentation of border_swap_information
 * @param distribution_values_0 source for even time steps and destination for odd time steps
 * @param distribution_values_1 source for odd time steps and destination for even time steps
 * @param access_function the access function according to which distribution values are to be accessed
 * @param iterations this many iterations will be performed
 */
void sequential_two_lattice::run_debug
(  
    const std::vector<unsigned int> &fluid_nodes,       
    const border_swap_information &boundary_nodes,
    std::vector<double> &distribution_values_0, 
    std::vector<double> &distribution_values_1,   
    const access_function access_function,
    const unsigned int iterations
)
{
    to_console::print_run_greeting("sequential two-lattice algorithm", iterations);

    std::vector<double> temp;
    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    for(auto time = 0; time < iterations; ++time)
    {
        std::cout << "\033[33mIteration " << time << ":\033[0m";
        result[time] = sequential_two_lattice::stream_and_collide_debug
        (
            fluid_nodes, 
            boundary_nodes, 
            distribution_values_0, 
            distribution_values_1, 
            access_function
        );     
        std::cout << "\tFinished iteration " << time << std::endl;

        temp = std::move(distribution_values_0);
        distribution_values_0 = std::move(distribution_values_1);
        distribution_values_1 = std::move(temp);
    }

    if(RESULTS_TO_CSV)
    {
        sim_data_to_csv(result, "results.csv");
    }

    to_console::print_simulation_results(result);
    std::cout << "All done, exiting simulation. " << std::endl;
}