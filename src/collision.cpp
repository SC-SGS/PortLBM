#include "../include/collision.hpp"

#include <iostream>

/**
 * @brief Performs the collision step for a node with the specified distribution values, velocity and density.
 * 
 * @param values a vector containing the distribution values of this node.
 * @param u the flow velocity at this node
 * @param density the density at this node
 * @return a vector containing the updated distribution values.
 */
std::vector<double> collision::collide_bgk
(
    const std::vector<double> &values, 
    const velocity &u, 
    double density
)
{
    double sum = 0;
    std::vector<double> result = maxwell_boltzmann_distribution(u, density);
    for(auto i = 0; i < DIRECTION_COUNT; ++i)
    {
        result[i] = -(1/RELAXATION_TIME) * (values[i] - result[i]) + values[i];
    }
    return result;
}

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
void collision::collide_all_bgk
(
    const std::vector<unsigned int> &fluid_nodes,
    std::vector<double> &values, 
    const std::vector<velocity> &all_velocities, 
    const std::vector<double> &all_densities,
    const access_function access
)
{
    for (const auto fluid_node : fluid_nodes)
    {
        std::vector<double> current_dist_values = lbm_access::get_distribution_values_of(values, fluid_node, access);
        std::vector<double> new_distributions = collision::collide_bgk(current_dist_values, all_velocities[fluid_node], all_densities[fluid_node]);
        lbm_access::set_distribution_values_of(new_distributions, values, fluid_node, access);
    }
}

/**
 * @brief Performs the collision step for the specified fluid node.
 * 
 * @param node the index of the node for which the collision step will be performed
 * @param distribution_values a vector containing all distribution distribution_values
 * @param access_function the access to node values will be performed according to this access function
 * @param velocities a vector containing the velocity values of all nodes
 * @param densities a vector containing the density values of all nodes
 */
void collision::perform_collision
(
    const unsigned int node,
    std::vector<double> &distribution_values, 
    const access_function &access_function, 
    std::vector<velocity> &velocities, 
    std::vector<double> &densities
)
{
    std::vector<double> current_distributions(DIRECTION_COUNT, 0);
    velocity current_velocity = {0,0};
    double current_density = 0;

    current_distributions = 
        lbm_access::get_distribution_values_of(distribution_values, node, access_function);
    current_velocity = macroscopic::flow_velocity(current_distributions);    
    velocities[node] = current_velocity;
    current_density = macroscopic::density(current_distributions);
    densities[node] = current_density;
    current_distributions = collision::collide_bgk(current_distributions, current_velocity, current_density);
    lbm_access::set_distribution_values_of(current_distributions, distribution_values, node, access_function);
} 