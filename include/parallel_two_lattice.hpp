#ifndef PARALLEL_TWO_LATTICE_HPP
#define PARALLEL_TWO_LATTICE_HPP

#include "access.hpp"
#include "boundaries.hpp"
#include "collision.hpp"
#include "defines.hpp"
#include "file_interaction.hpp"
#include "utils.hpp"

#include "sequential_two_lattice.hpp"
#include "parallel_framework.hpp"

#include <iostream>
#include <vector>

/**
 * @brief This namespace contains all methods for the non-framework-version of the parallel two-lattice algorithm.
 *        In this version of the algorithm, HPX takes care of the domain decomposition and scheduling. 
 */
namespace parallel_two_lattice
{
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
    sim_data_tuple stream_and_collide
    (
        const std::vector<unsigned int> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &source,
        std::vector<double> &destination,
        const access_function access_function
    );

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
    sim_data_tuple stream_and_collide_debug
    (
        const std::vector<unsigned int> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &source, 
        std::vector<double> &destination,    
        const access_function access_function
    );

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
    void run
    (  
        const std::vector<unsigned int> &fluid_nodes,       
        const border_swap_information &boundary_nodes,
        std::vector<double> &distribution_values_0, 
        std::vector<double> &distribution_values_1,   
        const access_function access_function,
        const unsigned int iterations
    );

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
    void run_debug
    (  
        const std::vector<unsigned int> &fluid_nodes,       
        const border_swap_information &boundary_nodes,
        std::vector<double> &distribution_values_0, 
        std::vector<double> &distribution_values_1,   
        const access_function access_function,
        const unsigned int iterations
    );

    /**
     * @brief Updates the ghost nodes that represent inlet and outlet edges.
     *        When updating, a velocity border condition will be considered for the input
     *        and a density border condition for the output.
     *        The inlet velocity is constant throughout all inlet nodes whereas the outlet nodes
     *        all have the specified density.
     *        The corresponding values are constants defined in "../include/"defines.hpp".
     * 
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param velocities a vector containing the velocities of all nodes
     * @param densities a vector containing the densities of all nodes
     * @param access_function the access function used to access the distribution values
     */
    void update_velocity_input_density_output
    (
        std::vector<double> &distribution_values,
        std::vector<velocity> &velocities,
        std::vector<double> &densities, 
        const access_function access_function
    );
}

#endif