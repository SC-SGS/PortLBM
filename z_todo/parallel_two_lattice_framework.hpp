#ifndef PARALLEL_TWO_LATTICE_FRAMEWORK_HPP
#define PARALLEL_TWO_LATTICE_FRAMEWORK_HPP

#include "access.hpp"
#include "boundaries.hpp"
#include "collision.hpp"
#include "defines.hpp"
#include "file_interaction.hpp"
#include "utils.hpp"

#include "parallel_framework.hpp"
#include "sequential_two_lattice.hpp"


/**
 * @brief This namespace contains all methods for the framework of the parallel two-lattice algorithm.
 *        Notice that the framework itself is the same for all algorithms but the respective executions need adaptions.
 */
namespace parallel_two_lattice_framework
{
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
    void run
    (  
        const std::vector<start_end_it_tuple> &fluid_nodes,       
        const border_swap_information &boundary_nodes,
        std::vector<double> &distribution_values_0, 
        std::vector<double> &distribution_values_1,   
        const access_function access_function,
        const unsigned int iterations
    );

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
    void run_debug
    (  
        const std::vector<start_end_it_tuple> &fluid_nodes,       
        const border_swap_information &boundary_nodes,
        std::vector<double> &distribution_values_0, 
        std::vector<double> &distribution_values_1,   
        const access_function access_function,
        const unsigned int iterations
    );

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
    sim_data_tuple stream_and_collide
    (
        const std::vector<start_end_it_tuple> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &source, 
        std::vector<double> &destination,    
        const access_function access_function,
        const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
    );

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
    sim_data_tuple stream_and_collide_debug
    (
        const std::vector<start_end_it_tuple> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &source, 
        std::vector<double> &destination,    
        const access_function access_function,
        const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
    );
}

#endif