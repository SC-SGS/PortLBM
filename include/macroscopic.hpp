#ifndef MACROSCOPIC_HPP
#define MACROSCOPIC_HPP

#include "access.hpp"
#include "defines.hpp"

#include <list>
#include <numeric>

/**
 * @brief This namespace includes functions for calculating macroscopic observables of non-boundary nodes 
 *        for the D2Q9I model.
 */
namespace macroscopic
{
    /**
     * @brief Calculates the density of a fluid node.
     * 
     * @param distribution_functions a vector containing all distribution functions of the respective fluid node.
     */
    inline double density(const std::vector<double> &distribution_functions)
    {
        return std::accumulate(distribution_functions.begin(), distribution_functions.end(), 0.0);
    }

    /**
     * @brief Calculates the flow velocity of a fluid node.
     * 
     * @param distribution_functions a vector containing all distribution functions of the respective fluid node.
     * @return velocity a two-dimensional array representing the flow velocity
     */
    velocity flow_velocity(const std::vector<double> &distribution_functions);

    /**
     * @brief Returns a simulation data tuple, i.e. a tuple containing all velocities (0) and all density values (1)
     *        for all fluid nodes using the specified distribution values and access function.
     * 
     * @param fluid_nodes a vector containing the indices of all fluid vectors
     * @param all_distributions a vector containing all distribution values
     * @param access_function This function is used to access the distribution values.
     * @return a tuple containing the simulation data
     */
    sim_data_tuple get_sim_data_tuple
    (
        const std::vector<unsigned int> &fluid_nodes,
        const std::vector<double> &all_distributions, 
        const access_function access_function
    );
}

#endif