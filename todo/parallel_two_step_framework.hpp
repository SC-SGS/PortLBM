#ifndef PARALLEL_TWO_STEP_FRAMEWORK_HPP
#define PARALLEL_TWO_STEP_FRAMEWORK_HPP

#include <vector>
#include <set>
#include "defines.hpp"
#include "access.hpp"
#include "collision.hpp"
#include "file_interaction.hpp"
#include "utils.hpp"
#include "boundaries.hpp"
#include "parallel_framework.hpp"

/**
 * @brief This namespace contains all methods for the framework of the parallel two-step algorithm.
 *        Notice that the framework itself is the same for all algorithms but the respective executions need adaptions.
 */
namespace parallel_two_step_framework
{
    /**
     * @brief Performs the parallel two-step algorithm for the specified number of iterations.
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
     * @brief Performs the parallel two-step algorithm for the specified number of iterations.
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
     * @brief Performs the streaming step for all fluid nodes within the specified bounds.
     * 
     * @param fluid_nodes a tuple of the first and last element of an iterator over all fluid nodes within the respective subdomain
     * @param distribution_values a vector containing all distribution distribution_values
     * @param access_function the access to node values will be performed according to this access function
     */
    void perform_stream
    (
        const start_end_it_tuple fluid_node_bounds, 
        std::vector<double> &distribution_values, 
        const access_function access_function
    );

    /**
     * @brief Realizes inflow and outflow by an inward stream of each border node.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param access_function the access function used to access the distribution values
     * @param y_values a tuple containing the y values of all regular layers (0) and all buffer layers (1)
     */
    void ghost_stream_inout
    (
        std::vector<double> &distribution_values, 
        const access_function access_function,
        const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values
    );
    
    /**
     * @brief Performs a halfway bounce-back streaming update for all fluid nodes within the simulation domain.
     * 
     * @param bsi see documentation of border_swap_information
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param access_function the access function used to access the distribution values
     */
    void perform_boundary_update
    (
        const border_swap_information &bsi,
        std::vector<double> &distribution_values, 
        const access_function access_function
    );
}

#endif