#ifndef PARALLEL_SWAP_FRAMEWORK_HPP
#define PARALLEL_SWAP_FRAMEWORK_HPP

#include "access.hpp"
#include "boundaries.hpp"
#include "collision.hpp"
#include "defines.hpp"
#include "file_interaction.hpp"
#include "parallel_framework.hpp"
#include "sequential_swap.hpp"
#include "utils.hpp"

#include <vector>

/**
 * @brief This namespace contains all methods for the framework of the parallel swap algorithm.
 *        Notice that the framework itself is the same for all algorithms but the respective executions need adaptions.
 */
namespace parallel_swap_framework
{
    /**
     * @brief Performs the parallel swap algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes a vector containing the first and last element of an iterator over all fluid nodes within each subdomain
     * @param distribution_values the vector containing the distribution values of all nodes
     * @param bsi see documentation of border_swap_information
     * @param access_function the access function according to which the values are to be accessed
     * @param iterations this many iterations will be performed
     */
    void run
    (  
        const std::vector<start_end_it_tuple> &fluid_nodes,       
        std::vector<double> &distribution_values, 
        const border_swap_information &bsi,
        const access_function access_function,
        const unsigned int iterations
    );

    /**
     * @brief Performs the parallel swap algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes a vector containing the first and last element of an iterator over all fluid nodes within each subdomain
     * @param distribution_values the vector containing the distribution values of all nodes
     * @param bsi see documentation of border_swap_information
     * @param access_function the access function according to which the values are to be accessed
     * @param iterations this many iterations will be performed
     */
    void run_debug
    (  
        const std::vector<start_end_it_tuple> &fluid_nodes,       
        std::vector<double> &distribution_values, 
        const border_swap_information &bsi,
        const access_function access_function,
        const unsigned int iterations
    );

    /**
     * @brief Performs the streaming and collision step for all fluid nodes within the simulation domain.
     *        The border conditions are enforced through ghost nodes.
     * 
     * @param fluid_nodes a vector containing the first and last element of an iterator over all fluid nodes within each subdomain
     * @param bsi see documentation of border_swap_information
     * @param distribution_values a vector containing all distribution values
     * @param access_function the access to node values will be performed according to this access function
     * @param y_values a tuple containing the y values of all regular layers (0) and all buffer layers (1)
     * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
     * @return sim_data_tuple see documentation of sim_data_tuple
     */
    sim_data_tuple stream_and_collide
    (
        const std::vector<start_end_it_tuple> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &distribution_values,    
        const access_function access_function,
        const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
    );

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
    sim_data_tuple stream_and_collide_debug
    (
        const std::vector<start_end_it_tuple> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &distribution_values,    
        const access_function access_function,
        const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
    );

    /**
     * @brief Performs an update for the buffer with the specified boundaries.
     *        It prepares the subdomain-wise streaming and performs the swap step for the uppermost row of each subdomain.
     * 
     * @param buffer_bounds a tuple containing the indices of the first and the last node of the buffer
     * @param distribution_values a vector containing all distribution values
     * @param access_function the access to node values will be performed according to this access function
     */
    void swap_buffer_update
    (
        const std::tuple<unsigned int, unsigned int> &buffer_bounds,
        std::vector<double> &distribution_values,
        const access_function access_function
    );
}

#endif