#ifndef COLLISION_HPP
#define COLLISION_HPP

#include "access.hpp"
#include "defines.hpp"
#include "macroscopic.hpp"
#include "utils.hpp"

/**
 * @brief This namespace contains all functions necessary for the collision step.
 */
namespace collision
{
    /**
     * @brief Performs the collision step for a node with the specified distribution values, velocity and density.
     * 
     * @param values a vector containing the distribution values of this node.
     * @param u the flow velocity at this node
     * @param density the density at this node
     * @return a vector containing the updated distribution values.
     */
    std::vector<double> collide_bgk
    (
        const std::vector<double> &values, 
        const velocity &u, 
        double density
    );
    
    /**
     * @brief Performs the collision step for all fluid nodes.
     *        This function is intended to be used in the two-step algorithm as the streaming and collision steps cannot be fused there.
     * 
     * @param fluid_nodes a vector containing the node indices of all fluid nodes
     * @param values a vector containing the distribution values of all nodes
     * @param all_velocities a vector containing the velocity values of all fluid nodes
     * @param all_densities a vector containing the density values of all fluid nodes
     * @param access this function is used to access the distribution values
     */
    void collide_all_bgk
    (
        const std::vector<unsigned int> &fluid_nodes,
        std::vector<double> &values, 
        const std::vector<velocity> &all_velocities, 
        const std::vector<double> &all_densities,
        const access_function access
    );

    /**
     * @brief Performs the collision step for the specified fluid node.
     * 
     * @param node the index of the node for which the collision step will be performed
     * @param distribution_values a vector containing all distribution distribution_values
     * @param access_function the access to node values will be performed according to this access function
     * @param velocities a vector containing the velocity values of all nodes
     * @param densities a vector containing the density values of all nodes
     */
    void perform_collision
    (
        const unsigned int node,
        std::vector<double> &distribution_values, 
        const access_function &access_function, 
        std::vector<velocity> &velocities, 
        std::vector<double> &densities
    );
}
#endif