/**
 * @file        console_output.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the definitions of various functions for console outputs.
 * 
 * @version     1.0
 * 
 * @date        2024-10-10
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/console/console_output.hpp"

void lbm::console::print_phase_vector
(
    const std::vector<uint8_t> &vector,
    const unsigned int horizontal_nodes
)
{
    unsigned int vertical_nodes = vector.size() / horizontal_nodes;

    for(auto y = vertical_nodes; y-- > 0; )
    {
        for(auto x = 0; x < horizontal_nodes; ++x)
        {
            if(vector[lbm::access::get_node_index(x, y, horizontal_nodes)]) std::cout << "\033[32m#\033[0m";
            else std::cout << "\033[34m~\033[0m"; 
            std::cout << " ";
        }
        std::cout << "\n";
    }
} 

void lbm::console::print_velocities
(
    const lbm::Properties &properties,
    const std::vector<double> &x_velocities, 
    const std::vector<double> &y_velocities,
    const unsigned int time_step
)
{
    unsigned int index = 0;

    for(auto y = properties.vertical_nodes - 1; y-- > 1; )
    {
        for(auto x = 1; x < properties.horizontal_nodes - 1; ++x)
        {
            index = lbm::access::results::get_result_index_no_ghosts(
                            lbm::access::get_node_index(x, y, properties.horizontal_nodes), properties.horizontal_nodes,
                            properties.domain_node_count, time_step);

            std::cout << "(" << x_velocities[index] << ", " << y_velocities[index] << ")";
            std::cout << "\t  \033[0m";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
} 

void lbm::console::print_densities
(
    const lbm::Properties &properties,
    const std::vector<double> &densities,
    const unsigned int time_step
)
{
    unsigned int node_index = 0;

    for(auto y = properties.vertical_nodes - 1; y-- > 1; )
    {
        for(auto x = 1; x < properties.horizontal_nodes - 1; ++x)
        {
            if(x == 0 && y == 0) std::cout << "\033[31m";
            else if(x == (properties.horizontal_nodes - 1) && y == (properties.vertical_nodes -1)) std::cout << "\033[34m";

            std::cout << densities[
                lbm::access::results::get_result_index_no_ghosts(
                    lbm::access::get_node_index(x, y, properties.horizontal_nodes), properties.horizontal_nodes,
                    properties.domain_node_count, time_step
                    )
                ]; 
            std::cout << "\t\033[0m";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
} 

void lbm::console::print_simulation_results
(
    const lbm::Properties &properties,
    const lbm::SimulationResults &simulation_results
)
{
    std::cout << "Velocity values: \n\n";

    for(auto i = 0; i < properties.time_steps; ++i)
    {
        std::cout << "t = " << i << "\n";
        std::cout << "-------------------------------------------------------------------------------- \n";
        lbm::console::print_velocities(properties, *simulation_results.x_velocities, *simulation_results.y_velocities, i);
        std::cout << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Density values: \n\n";
    
    for(auto i = 0; i < properties.time_steps; ++i)
    {
        std::cout << "t = " << i << "\n";
        std::cout << "-------------------------------------------------------------------------------- \n";
        lbm::console::print_densities(properties, *simulation_results.densities, i);
        std::cout << "\n";
    }
    std::cout << "\n";
}