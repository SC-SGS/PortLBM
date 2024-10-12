#include "../include/sequential_swap.hpp"

#include <iostream>

/**
 * @brief This vector contains all directions in which "active" streaming happens in the shape
 *        of a swap of values.
 */
const std::vector<unsigned int> sequential_swap::ACTIVE_STREAMING_DIRECTIONS = {5,6,7,8};

/**
 * @brief Retrieves an improved version of the border swap information data structure.
 *        This method does not consider inlet and outlet ghost nodes when performing bounce-back
 *        as the inserted values will be overwritten by inflow and outflow values anyways.
 * 
 * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain
 * @param phase_information a vector containing the phase information for every vector (true means solid)
 * @return border_swap_information see documentation of border_swap_information
 */
border_swap_information sequential_swap::retrieve_swap_info
(
    const std::vector<unsigned int> &fluid_nodes, 
    const std::vector<bool> &phase_information
)
{
    border_swap_information result;
    std::vector<unsigned int> current_adjacencies;
    std::vector<unsigned int> swap_adjacencies;
    std::tuple<unsigned int, unsigned int> coords;
    for(const auto node : fluid_nodes)
    {
        current_adjacencies = {};
        swap_adjacencies = {};
        // Determine all nodes with non-inout ghost neighbors
        for(const auto direction : STREAMING_DIRECTIONS)
        {
            unsigned int current_neighbor = lbm_access::get_neighbor(node, direction);
            if(is_non_inout_ghost_node(current_neighbor, phase_information))
            {
                current_adjacencies.push_back(direction);
            }
        }
        // Determine intersection with active streaming direction mask
        std::set_intersection(
            current_adjacencies.begin(), 
            current_adjacencies.end(), 
            ACTIVE_STREAMING_DIRECTIONS.begin(), 
            ACTIVE_STREAMING_DIRECTIONS.end(), 
            std::back_inserter(swap_adjacencies)
            );

        // Determine node coordinates
        coords = lbm_access::get_node_coordinates(node);
        
        if(std::get<0>(coords) == 1) // Inlet node
        {
            swap_adjacencies.push_back(0);
            swap_adjacencies.push_back(3);
        }
        else if(std::get<0>(coords) == (HORIZONTAL_NODES - 2)) // Outlet node
        {
            swap_adjacencies.push_back(2);
        }
        current_adjacencies = {node};
        std::sort(swap_adjacencies.begin(), swap_adjacencies.end());
        current_adjacencies.insert(current_adjacencies.end(), swap_adjacencies.begin(), swap_adjacencies.end());
        
        if(current_adjacencies.size() > 1) result.push_back(current_adjacencies);
    }
    return result;
}

/**
 * @brief Performs the combined streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 *        This variant of the combined streaming and collision step will print several debug comments to the console.
 */
sim_data_tuple sequential_swap::stream_and_collide
(
    const border_swap_information &bsi,
    const std::vector<unsigned int> &fluid_nodes,
    std::vector<double> &distribution_values,    
    const access_function access_function
)
{
    std::vector<velocity> velocities(TOTAL_NODE_COUNT, velocity{0,0});
    std::vector<double> densities(TOTAL_NODE_COUNT, -1);
    std::vector<double> current_distributions(DIRECTION_COUNT, 0);

    // Border node initialization
    for(const auto node : bsi)
    {
        for(auto it = node.begin() + 1; it < node.end(); ++it)
        {
            sequential_swap::perform_swap_step(distribution_values, node[0], access_function, *it);
        }
    }

    for(const auto node : fluid_nodes)
    {
        sequential_swap::perform_swap_step(distribution_values, node, access_function, ACTIVE_STREAMING_DIRECTIONS);
        sequential_swap::restore_order(distribution_values, node, access_function);
        collision::perform_collision(node, distribution_values, access_function, velocities, densities);
    }

    /* Update ghost nodes */
    boundary_conditions::update_velocity_input_density_output(distribution_values, velocities, densities, access_function);
    sequential_swap::restore_inout_correctness(distribution_values, access_function);

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs the combined streaming and collision step for all fluid nodes within the simulation domain.
 *        The border conditions are enforced through ghost nodes.
 *        This variant of the combined streaming and collision step will print several debug comments to the console.
 */
sim_data_tuple sequential_swap::stream_and_collide_debug
(
    const border_swap_information &bsi,
    const std::vector<unsigned int> &fluid_nodes,
    std::vector<double> &distribution_values,    
    const access_function access_function
)
{
    std::cout << "Distribution values before stream and collide: " << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
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
    to_console::print_distribution_values(distribution_values, access_function);

    // Swapping step
    for(const auto node : fluid_nodes)
    {
        sequential_swap::perform_swap_step(distribution_values, node, access_function, ACTIVE_STREAMING_DIRECTIONS);
    }
    std::cout << "Distribution values after swap for every node: " << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);

    // Restore precious order in here
    for(const auto node : fluid_nodes)
    {
        sequential_swap::restore_order(distribution_values, node, access_function);
    }
    std::cout << "Distribution values after ORDER has been restored for every node: " << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);


    // Collision
    for(const auto& node : fluid_nodes)
    {
        collision::perform_collision(node, distribution_values, access_function, velocities, densities);
    }
    std::cout << "Distribution values after collision: " << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);

    /* Update ghost nodes */
    boundary_conditions::update_velocity_input_density_output(distribution_values, velocities, densities, access_function);
    sequential_swap::restore_inout_correctness(distribution_values, access_function);

    std::cout << "Distribution values after ghost node update: " << std::endl;
    to_console::print_distribution_values(distribution_values, access_function);
    std::cout << std::endl;

    sim_data_tuple result{velocities, densities};

    return result;
}

/**
 * @brief Performs the sequential swap algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param bsi see documentation of border_swap_information
 * @param values the vector containing the distribution values of all nodes
 * @param access_function the access function according to which the values are to be accessed
 * @param iterations this many iterations will be performed
 */
void sequential_swap::run
(  
    const std::vector<unsigned int> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &values, 
    const access_function access_function,
    const unsigned int iterations
)
{ 
    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    for(auto time = 0; time < iterations; ++time)
    {   
        result[time] = sequential_swap::stream_and_collide(bsi, fluid_nodes, values, access_function);     
    }
    if(RESULTS_TO_CSV)
    {
        sim_data_to_csv(result, "results.csv");
    }
}

/**
 * @brief Performs the sequential swap algorithm for the specified number of iterations.
 * 
 * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
 * @param bsi see documentation of border_swap_information
 * @param values the vector containing the distribution values of all nodes
 * @param access_function the access function according to which the values are to be accessed
 * @param iterations this many iterations will be performed
 */
void sequential_swap::run_debug
(  
    const std::vector<unsigned int> &fluid_nodes,
    const border_swap_information &bsi,
    std::vector<double> &values, 
    const access_function access_function,
    const unsigned int iterations
)
{
    to_console::print_run_greeting("sequential swap algorithm", iterations); 
       
    std::vector<sim_data_tuple>result(
        iterations, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

    for(auto time = 0; time < iterations; ++time)
    {
        std::cout << "\033[33mIteration " << time << ":\033[0m";    
        
        result[time] = sequential_swap::stream_and_collide_debug(bsi, fluid_nodes, values, access_function);

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
 * @brief Restores the correctness of the outmost inlet and outlet nodes.
 *        Their distribution values are overwritten during the swap step.
 * 
 * @param distribution_values a vector containing all distribution distribution_values
 * @param access_function the access to node values will be performed according to this access function
 */
void sequential_swap::restore_inout_correctness
(
    std::vector<double> &distribution_values,    
    const access_function access_function
)
{
    std::vector<double> inout_update_values(DIRECTION_COUNT, 0);
    // Inlets
    inout_update_values = maxwell_boltzmann_distribution(INLET_VELOCITY, INLET_DENSITY);
    lbm_access::set_distribution_values_of(inout_update_values, distribution_values, lbm_access::get_node_index(0,0), access_function);
    lbm_access::set_distribution_values_of(inout_update_values, distribution_values, lbm_access::get_node_index(0, VERTICAL_NODES - 1), access_function);

    // Outlets
    inout_update_values = maxwell_boltzmann_distribution(OUTLET_VELOCITY, OUTLET_DENSITY);
    lbm_access::set_distribution_values_of(inout_update_values, distribution_values, lbm_access::get_node_index(HORIZONTAL_NODES - 1, 0), access_function);
    lbm_access::set_distribution_values_of(inout_update_values, distribution_values, lbm_access::get_node_index(HORIZONTAL_NODES - 1, VERTICAL_NODES - 1), access_function);
}