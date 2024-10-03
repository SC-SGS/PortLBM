/**
 * @file        access.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Updated version of the lattice Boltzmann access functions first introduced in my SimTech project work:
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann
 *              The major change is that accessor objects are used to emulate different access patterns since SYCL
 *              is not compatible with function pointers.
 *              The data layouts were proposed by Mattila et al.
 * 
 * @version     2.0
 * 
 * @date        2024-09-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "../include/access.hpp"
#include <list>

lbm_access::LBMAccessorObject::LBMAccessorObject
(
    unsigned int horizontal_nodes
) 
: 
horizontal_nodes(horizontal_nodes)
{};


lbm_access::LBMCollisionAccessor::LBMCollisionAccessor
(
    unsigned int horizontal_nodes
)
:
lbm_access::LBMAccessorObject::LBMAccessorObject(horizontal_nodes)
{};


lbm_access::LBMStreamAccessor::LBMStreamAccessor
(
    unsigned int horizontal_nodes,
    unsigned int buffered_node_count
) 
: 
lbm_access::LBMAccessorObject::LBMAccessorObject(horizontal_nodes),
buffered_node_count(buffered_node_count) 
{};


lbm_access::LBMBundleAccessor::LBMBundleAccessor
(
    unsigned int horizontal_nodes,
    unsigned int buffered_node_count
) 
: 
lbm_access::LBMAccessorObject::LBMAccessorObject(horizontal_nodes),
buffered_node_count(buffered_node_count) 
{};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                  //
//  Careful! All of the following content is deprecated and will be removed in the future.                                          //
//                                                                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief This function returns the distribution values of the node with the specified index using the specified access pattern.
 * 
 * @param source the distribution values will be read from this vector
 * @param node_index this is the index of the node in the domain
 * @param access this access function will be used
 * @return A vector containing the distribution values
 */
std::vector<double> lbm_access::get_distribution_values_of
(
    const std::vector<double> &source, 
    int node_index, 
    access_function access
)
{
    std::vector<double> dist_vals(9,0);
    for(auto direction = 0; direction < DIRECTION_COUNT; ++direction)
    {
        dist_vals[direction] = source[access(node_index, direction)];
    }
    return dist_vals;
}

/**
 * @brief This function sets all distribution values of the node with the specified index to the specified values 
 *        using the specified access pattern.
 * 
 * @param dist_vals a vector containing the values to which the distribution values shall be set
 * @param destination the distribution values will be written to this vector
 * @param node_index this is the index of the node in the domain
 * @param access this access function will be used
 */
void lbm_access::set_distribution_values_of
(
    const std::vector<double> &dist_vals, 
    std::vector<double> &destination, 
    int node_index, 
    access_function access
)
{
    for(auto direction = 0; direction < DIRECTION_COUNT; ++direction)
    {
        destination[access(node_index, direction)] = dist_vals[direction];
    }
}



