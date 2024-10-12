#ifndef TWO_SEQUENTIAL_SWAP_HPP
#define TWO_SEQUENTIAL_SWAP_HPP

#include "access.hpp"
#include "boundaries.hpp"
#include "collision.hpp"
#include "defines.hpp"
#include "file_interaction.hpp"
#include "macroscopic.hpp"
#include "utils.hpp"

#include <map>
#include <set>
#include <vector>

/**
 * @brief This namespace contains all methods for the sequential swap algorithm.
 */
namespace sequential_swap
{
    /**
     * @brief This vector contains all directions in which "active" streaming happens in the shape
     *        of a swap of values.
     */
    extern const std::vector<unsigned int> ACTIVE_STREAMING_DIRECTIONS;

    /**
     * @brief Retrieves an adapted version of the border swap information data structure.
     *        This method does not consider inlet and outlet ghost nodes when performing bounce-back
     *        as the inserted values will be overwritten by inflow and outflow values anyways.
     * 
     * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain
     * @param phase_information a vector containing the phase information for every vector (true means solid)
     * @return border_swap_information see documentation of border_swap_information
     */
    border_swap_information retrieve_swap_info
    (
        const std::vector<unsigned int> &fluid_nodes, 
        const std::vector<bool> &phase_information
    );

    /**
     * @brief Performs the sequential swap algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
     * @param bsi see documentation of border_swap_information
     * @param values the vector containing the distribution values of all nodes
     * @param access_function the access function according to which the values are to be accessed
     * @param iterations this many iterations will be performed
     */
    void run
    (  
        const std::vector<unsigned int> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &values, 
        const access_function access_function,
        const unsigned int iterations
    );

    /**
     * @brief Performs the sequential swap algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes A vector containing the indices of all fluid nodes in the domain
     * @param bsi see documentation of border_swap_information
     * @param values the vector containing the distribution values of all nodes
     * @param access_function the access function according to which the values are to be accessed
     * @param iterations this many iterations will be performed
     */
    void run_debug
    (  
        const std::vector<unsigned int> &fluid_nodes,
        const border_swap_information &bsi,
        std::vector<double> &values, 
        const access_function access_function,
        const unsigned int iterations
    );

    /**
     * @brief Performs the combined streaming and collision step for all fluid nodes within the simulation domain.
     *        The border conditions are enforced through ghost nodes.
     * 
     * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain.
     * @param bsi see documentation of border_swap_information
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param access_function the function used to access the distribution values
     * @return see documentation of sim_data_tuple
     */
    sim_data_tuple stream_and_collide
    (
        const border_swap_information &bsi,
        const std::vector<unsigned int> &fluid_nodes,
        std::vector<double> &distribution_values,    
        const access_function access_function
    );

    /**
     * @brief Performs the combined streaming and collision step for all fluid nodes within the simulation domain.
     *        The border conditions are enforced through ghost nodes.
     *        This variant of the combined streaming and collision step will print several debug comments to the console.
     * 
     * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain.
     * @param bsi see documentation of border_swap_information
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param access_function the function used to access the distribution values
     * @return see documentation of sim_data_tuple
     */
    sim_data_tuple stream_and_collide_debug
    (
        const border_swap_information &bsi,
        const std::vector<unsigned int> &fluid_nodes,
        std::vector<double> &distribution_values,    
        const access_function access_function
    );

    /**
     * @brief Restores the correctness of the outmost inlet and outlet nodes.
     *        Their distribution values are overwritten during the swap step.
     * 
     * @param distribution_values a vector containing all distribution distribution_values
     * @param access_function the access to node values will be performed according to this access function
     */
    void restore_inout_correctness
    (
        std::vector<double> &distribution_values,    
        const access_function access_function
    );

    /**
     * @brief Restores the original order after streaming is completed for a node.
     *        Notice that this method automatically achieves the halfway bounce-back boundary condition.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param node_index the index of the node whose values are to be reordered
     * @param access_function the function used to access the distribution values
     */
    inline void restore_order
    (
        std::vector<double> &distribution_values,
        const unsigned int node_index,
        const access_function access_function
    )
    {
        for(const auto dir : {0,1,2,3})
        {
            vec_utils::swap(
                distribution_values, 
                access_function(node_index, dir), 
                access_function(node_index, invert_direction(dir))
            );
        }
    }

    /**
     * @brief Performs the swap step that is equvalent to the "active" streaming step for the specified node.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param node_index the index of the node for which the streaming step is to be performed
     * @param access_function the function used to access the distribution values
     * @param swap_directions a vector containing all directions in which swap will take place
     */
    inline void perform_swap_step
    (
        std::vector<double> &distribution_values,
        const unsigned int node_index,
        const access_function access_function,
        const std::vector<unsigned int> &swap_directions
    )
    {
        for(const auto dir : swap_directions)
        {
            vec_utils::swap(
                distribution_values, 
                access_function(node_index, dir), 
                access_function(lbm_access::get_neighbor(node_index, dir), invert_direction(dir))
            );
        }
    }

    /**
     * @brief Performs the swap step in a single direction for the specified node.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param node_index the index of the node for which the streaming step is to be performed
     * @param access_function the function used to access the distribution values
     * @param direction the direction in which swap will take place
     */
    inline void perform_swap_step
    (
        std::vector<double> &distribution_values,
        const unsigned int node_index,
        const access_function access_function,
        const unsigned int direction
    )
    {
        vec_utils::swap(
            distribution_values, 
            access_function(node_index, direction), 
            access_function(lbm_access::get_neighbor(node_index, direction), invert_direction(direction))
        );
    }
}

#endif