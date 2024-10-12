#ifndef SEQUENTIAL_TWO_STEP_HPP
#define SEQUENTIAL_TWO_STEP_HPP

#include "access.hpp"
#include "boundaries.hpp"
#include "collision.hpp"
#include "defines.hpp"
#include "file_interaction.hpp"
#include "macroscopic.hpp"

#include <set>
#include <vector>

/**
 * @brief This namespace contains all methods for the sequential two-step algorithm.
 */
namespace sequential_two_step
{
    /**
     * @brief Performs the streaming step for all fluid nodes within the simulation domain.
     *        Notice that each node is streaming outward.
     * 
     * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
     * @param distribution_values a vector containing all distribution distribution_values
     * @param access_function the access to node values will be performed according to this access function.
     */
    void perform_stream
    (
        const std::vector<unsigned int> &fluid_nodes, 
        std::vector<double> &distribution_values, 
        const access_function access_function
    );

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
    sim_data_tuple stream_and_collide
    (
        const std::vector<unsigned int> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &distribution_values,    
        const access_function access_function
    );

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
    sim_data_tuple stream_and_collide_debug
    (
        const std::vector<unsigned int> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &distribution_values,    
        const access_function access_function
    );

    /**
     * @brief Performs the sequential two-step algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
     * @param distribution_values the vector containing the distribution values of all nodes
     * @param bsi see documentation of border_swap_information
     * @param access_function the access function according to which the values are to be accessed
     * @param iterations this many iterations will be performed
     */
    void run
    (  
        std::vector<unsigned int> &fluid_nodes,       
        std::vector<double> &distribution_values, 
        border_swap_information &bsi,
        access_function access_function,
        unsigned int iterations
    );

    /**
     * @brief Performs the sequential two-step algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
     * @param distribution_values the vector containing the distribution values of all nodes
     * @param bsi see documentation of border_swap_information
     * @param access_function the access function according to which the values are to be accessed
     * @param iterations this many iterations will be performed
     */
    void run_debug
    (  
        std::vector<unsigned int> &fluid_nodes,       
        std::vector<double> &distribution_values, 
        border_swap_information &bsi,
        access_function access_function,
        unsigned int iterations
    );
}

#endif