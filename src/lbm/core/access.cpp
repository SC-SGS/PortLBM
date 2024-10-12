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
 * @version     2.1
 * 
 * @date        2024-10-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/core/access.hpp"

lbm::access::LBMAccessorObject::LBMAccessorObject(const unsigned int horizontal_nodes) 
: horizontal_nodes(horizontal_nodes){};


lbm::access::LBMCollisionAccessor::LBMCollisionAccessor(const unsigned int horizontal_nodes)
: lbm::access::LBMAccessorObject::LBMAccessorObject(horizontal_nodes){};


lbm::access::LBMStreamAccessor::LBMStreamAccessor
(
    const unsigned int horizontal_nodes,
    const unsigned int buffered_node_count
) 
: 
lbm::access::LBMAccessorObject::LBMAccessorObject(horizontal_nodes),
buffered_node_count(buffered_node_count) 
{};


lbm::access::LBMBundleAccessor::LBMBundleAccessor
(
    const unsigned int horizontal_nodes,
    const unsigned int buffered_node_count
) 
: 
lbm::access::LBMAccessorObject::LBMAccessorObject(horizontal_nodes),
buffered_node_count(buffered_node_count) 
{};
