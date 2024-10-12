#ifndef SEQUENTIAL_SHIFT_HPP
#define SEQUENTIAL_SHIFT_HPP

#include "access.hpp"
#include "boundaries.hpp"
#include "collision.hpp"
#include "defines.hpp"
#include "file_interaction.hpp"
#include "utils.hpp"

#include <vector>
#include <set>
#include <iostream>

/**
 * @brief This namespace contains all methods for the sequential shift algorithm.
 */
namespace sequential_shift
{
    /**
     * @brief Performs the streaming step for the fluid node with the specified index.
     * 
     * @param distribution_values the updated distribution values will be written to this vector
     * @param access_function An access function from the namespace sequential_shift::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param fluid_node the index of the fluid node
     * @param read_offset the read offset of the current iteration
     * @param write_offset the write offset of the current iteration
     */
    inline void shift_stream
    (
        std::vector<double> &distribution_values, 
        const access_function &access_function, 
        const unsigned int fluid_node,
        const unsigned int read_offset,
        const unsigned int write_offset
    )
    {
        for (const auto direction : ALL_DIRECTIONS)
        {
            distribution_values[access_function(fluid_node + write_offset, direction)] =
                distribution_values[
                    access_function(
                        lbm_access::get_neighbor(fluid_node + read_offset, invert_direction(direction)), 
                        direction)];
        }
    }

    /**
     * @brief Performs the collision step for the fluid node with the specified index.
     * 
     * @param node the index of the fluid node
     * @param distribution_values the updated distribution values will be written to this vector
     * @param access_function An access function from the namespace sequential_shift::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param velocities a vector containing the velocities of all nodes
     * @param densities a vector containing the densities of all nodes
     * @param write_offset the write offset of the current iteration
     */
    inline void shift_collision
    (
        const unsigned int node,
        std::vector<double> &distribution_values, 
        const access_function &access_function, 
        std::vector<velocity> &velocities, 
        std::vector<double> &densities,
        unsigned int write_offset
    )
    {
        std::vector<double> current_distributions(DIRECTION_COUNT, 0);
        current_distributions = lbm_access::get_distribution_values_of(distribution_values, node + write_offset, access_function);
        velocities[node] = macroscopic::flow_velocity(current_distributions);
        densities[node] = macroscopic::density(current_distributions);
        current_distributions = collision::collide_bgk(current_distributions, velocities[node], densities[node]);
        lbm_access::set_distribution_values_of(current_distributions, distribution_values, node + write_offset, access_function);
    }

    /**
     * @brief Performs a combined collision and streaming step for the specified fluid node.
     * 
     * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
     * @param fluid_nodes         a vector containing the indices of all fluid nodes within the simulation domain.
     * @param bsi                 see documentation of border_swap_information
     * @param access_function     An access function from the namespace sequential_shift::access_functions.
     *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param iteration           the iteration the algorithm is currently processing
     */
    sim_data_tuple stream_and_collide
    (
        std::vector<double> &distribution_values, 
        const std::vector<unsigned int> &fluid_nodes,
        const border_swap_information &bsi,
        const access_function access_function,
        const unsigned int iteration
    );

    /**
     * @brief Performs the parallel shift algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes         a vector containing the indices of all fluid nodes within the simulation domain.
     * @param bsi                 see documentation of border_swap_information
     * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
     * @param access_function     An access function from the namespace sequential_shift::access_functions.
     *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param iterations          this many iterations will be performed
     */
    void run
    (  
        std::vector<unsigned int> &fluid_nodes,       
        std::vector<double> &values, 
        border_swap_information &bsi,
        access_function access_function,
        unsigned int iterations
    );

    /**
     * @brief Performs the parallel shift algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes         a vector containing the indices of all fluid nodes within the simulation domain.
     * @param bsi                 see documentation of border_swap_information
     * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
     * @param access_function     An access function from the namespace sequential_shift::access_functions.
     *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param iterations          this many iterations will be performed
     */
    void run_debug
    (  
        std::vector<unsigned int> &fluid_nodes,       
        std::vector<double> &values, 
        border_swap_information &bsi,
        access_function access_function,
        unsigned int iterations
    );

    /**
     * @brief Updates the ghost nodes that represent inlet and outlet edges.
     *        When updating, a velocity border condition will be considered for the input
     *        and a density border condition for the output.
     *        The corresponding values are constants defined in "defines.hpp".
     * 
     * @param distribution_values the updated distribution values will be written to this vector
     * @param velocities a vector containing the velocities of all nodes
     * @param densities a vector containing the densities of all nodes
     * @param access_function An access function from the namespace sequential_shift::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param offset the write offset of the current iteration
     */
    void update_velocity_input_density_output
    (
        std::vector<double> &distribution_values, 
        std::vector<velocity> &velocities,
        std::vector<double> &densities, 
        const access_function access_function,
        const unsigned int offset
    );

    /**
     * @brief Create an example domain for testing purposes. The domain is a rectangle with
     *        dimensions specified in the defines file where the outermost nodes are ghost nodes.
     *        The upper and lower ghost nodes are solid whereas the leftmost and rightmost columns are fluid
     *        nodes that mark the inlet and outlet respectively.
     *        Notice that all data will be written to the parameters which are assumed to be empty initially.
     * 
     * @param distribution_values a vector containing all distribution values.
     * @param nodes a vector containing all node indices, including those of solid nodes and ghost nodes.
     * @param fluid_nodes a vector containing the indices of all fluid nodes.
     * @param phase_information a vector containing the phase information of all nodes where true means solid.
     * @param access_function An access function from the namespace sequential_shift::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     */
    void setup_example_domain
    (
        std::vector<double> &distribution_values,
        std::vector<unsigned int> &nodes,
        std::vector<unsigned int> &fluid_nodes,
        std::vector<bool> &phase_information,
        access_function access_function
    );

    /**
     * @brief This namespace contains all access functions in a version compatible with the shift algorithm.
     *        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     */
    namespace access_functions
    {
        /**
         * @brief Returns the vector index of the collision layout
         * 
         * @param node the node in the simulation domain
         * @param direction the direction of the velocity vector
         * @return the index of the vector storing the distribution values 
         */
        inline unsigned int collision(unsigned int node, unsigned int direction)
        {
            return DIRECTION_COUNT * node + direction;        
        }

        /**
         * @brief Returns the vector index of the stream layout
         * 
         * @param node the node in the simulation domain
         * @param direction the direction of the velocity vector
         * @return the index of the vector storing the distribution values  
         */
        inline unsigned int stream(unsigned int node, unsigned int direction)
        {
            return (TOTAL_NODE_COUNT + SHIFT_OFFSET) * direction + node;
        }

        /**
         * @brief Returns the vector index of the bundle layout
         * 
         * @param node the node in the simulation domain
         * @param direction the direction of the velocity vector
         * @return the index of the vector storing the distribution values  
         */
        inline unsigned int bundle(unsigned int node, unsigned int direction)
        {
            return 3 * (direction / 3) * (TOTAL_NODE_COUNT + SHIFT_OFFSET) + (direction % 3) + 3 * node; 
        }
    }
}

#endif