/**
 * @file        access.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       This header file contains the definitions of classes modelling data layouts for storing distribution
 *              values proposed by Mattila et al, as well as functions for retrieving the indices of nodes and
 *              simulation results for domains with or without decomposition and buffering.
 *
 * @version     2.6
 *
 * @date        March 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#ifndef LBM_ACCESS_HPP
#define LBM_ACCESS_HPP

// LBM core functionality
#include "constants.hpp"

// Standard library
#include <array>
#include <concepts>
#include <string_view>
#include <tuple>
#include <vector>

namespace lbm
{

namespace core
{

/**
 * @brief   This namespace contains accessor classes for different data layouts and general functions mapping
 *          input data to node indices.
 */
namespace access
{

// ACCESS FUNCTIONS ///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Models the access of distribution values according to the collision data layout.
 */
struct CollisionAccessor
{
    constexpr static std::string_view layout_string = "collision";

    /**
     * @brief   Retrieves the index of the distribution value within the according vector
     *          following the collision layout.
     *
     * @param[in]   node                        index of the node
     * @param[in]   direction                   index of the direction
     * @param[in]   total_buffered_node_count   blind parameter in this case, kept for interface
     *                                          compatibility
     *
     * @return  the index of the corresponding distribution value
     */
    inline static unsigned int
    at(const unsigned int node, const unsigned int direction, const unsigned int total_buffered_node_count)
    {
        return 9 * node + direction;
    }
};

/**
 * @brief Models the access of distribution values according to the stream data layout.
 */
struct StreamAccessor
{
    constexpr static std::string_view layout_string = "stream";

    /**
     * @brief   Retrieves the index of the distribution value within the according vector
     *          following the stream layout.
     *
     * @param[in]   node                          index of the node
     * @param[in]   direction                     index of the direction
     * @param[in]   total_buffered_node_count     total amount of nodes in the lattice including buffers
     *                                            and ghosts
     *
     * @return the index of the corresponding distribution value
     */
    inline static unsigned int
    at(const unsigned int node, const unsigned int direction, const unsigned int total_buffered_node_count)
    {
        return total_buffered_node_count * direction + node;
    }
};

/**
 * @brief Models the access of distribution values according to the bundle data layout.
 */
struct BundleAccessor
{
    constexpr static std::string_view layout_string = "bundle";

    /**
     * @brief   Retrieves the index of the distribution value within the according vector
     *          following the stream layout.
     *
     * @param[in]   node                          index of the node
     * @param[in]   direction                     index of the direction
     * @param[in]   total_buffered_node_count     total amount of nodes in the lattice including buffers
     *                                            and ghosts
     *
     * @return the index of the corresponding distribution value
     */
    inline static unsigned int
    at(const unsigned int node, const unsigned int direction, const unsigned int total_buffered_node_count)
    {
        return 3 * (direction / 3) * total_buffered_node_count + (direction % 3) + 3 * node;
    }
};

/**
 * @brief   This concept is used to restrict the template parameter of methods relying on an accessor
 *          function. Any methods or classes operating on distribution values must use a data layout. The
 *          intended use is to specify an `AccessorConcept` as a template parameter, that is, any of the
 *          following classes:
 *
 *          - `lbm::core::access::CollisionAccessor`
 *
 *          - `lbm::core::access::StreamAccessor`
 *
 *          - `lbm::core::access::BundleAccessor`
 *
 * @tparam  T    this class is tested for membership in the set of accessors during compile time
 */
template <class T>
concept AccessorConcept =
    std::same_as<T, CollisionAccessor> || std::same_as<T, StreamAccessor> || std::same_as<T, BundleAccessor>;

// NODE INDEX FUNCTIONS ///////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Returns the index of the neighbor that is reached when moving in the specified direction.
 *
 * @param[in]   node_index        the index of the current node
 * @param[in]   direction         the direction of movement
 * @param[in]   horizontal_nodes  the amount of horizontal nodes in the lattice
 *
 * @return  the node index of the neighbor
 */
inline unsigned int
get_neighbor(const unsigned int node_index, const unsigned int direction, const unsigned int horizontal_nodes)
{
    int y_offset = direction / 3 - 1;
    // -1 for {0,1,2}, 0 for {3,4,5}, 1 for {6,7,8}

    int x_offset = direction % 3 - 1;
    // -1 for {0,3,6}, 0 for {1,4,7}, 1 for {2,5,8}

    return node_index + y_offset * horizontal_nodes + x_offset;
}

/**
 * @brief   Returns the linear index the desired node has within the array that stores it. The origin lies
 *          at the lower left corner and enumeration is row-major.
 *
 * @param[in]   x                 x coordinate
 * @param[in]   y                 y coordinate
 * @param[in]   horizontal_nodes  the amount of horizontal nodes in the lattice
 *
 * @return the index of the desired note
 */
inline unsigned int get_node_index(const unsigned int x, const unsigned int y, const unsigned int horizontal_nodes)
{
    return x + y * horizontal_nodes;
}

// RESULT INDEX FUNCTIONS /////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Retrieves the linear index of the result belonging to the node with the specified global
 *          coordinates assuming the specified extent of the non-extended domain. Careful: This function
 *          only operates correctly if the specified global coordinates belong to a node that is located
 *          within the actual unexpanded simulation domain! Ensuring this is the responsibility of the
 *          caller.
 *
 * @param[in]   global_x                    the global x coordinate of the node in question
 * @param[in]   global_y                    the global y coordinate of the node in question
 * @param[in]   horizontal_nodes_domain     the amount of horizontal nodes in the unexpanded domain
 * @param[in]   domain_node_count           the total amount of nodes in the unexpanded domain
 * @param[in]   time_step                   the time step at which the results are read
 *
 * @return  the linearized index of the according results
 */
inline unsigned int get_result_index(const unsigned int global_x,
                                     const unsigned int global_y,
                                     const unsigned int horizontal_nodes_domain,
                                     const unsigned int domain_node_count,
                                     const unsigned int time_step)
{
    return (global_x - 1) + (global_y - 1) * (horizontal_nodes_domain - 2) + time_step * domain_node_count;
}

/**
 * @brief   Retrieves the linear index of the result belonging to the node with the specified global
 *          coordinates assuming the specified extent of the non-extended domain. Careful: This function
 *          only operates correctly if the specified global coordinates belong to a node that is located
 *          within the actual unexpanded simulation domain! Ensuring this is the responsibility of the
 *          caller.
 *
 * @param[in]   global_x                    the global x coordinate of the node in question
 * @param[in]   global_y                    the global y coordinate of the node in question
 * @param[in]   horizontal_nodes_domain     the amount of horizontal nodes in the unexpanded domain
 *
 * @return  the linearized index of the according results
 */
inline unsigned int
get_result_index(const unsigned int global_x, const unsigned int global_y, const unsigned int horizontal_nodes_domain)
{
    return (global_x - 1) + (global_y - 1) * (horizontal_nodes_domain - 2);
}

/**
 * @brief   Retrieves the linear index of the result belonging to the node with the specified linear index.
 *          Careful: This function only operates correctly if the specified linear index belongs to a node
 *          that is located within the actual unexpanded simulation domain! Ensuring this is the
 *          responsibility of the caller.
 *
 * @param[in]   node_index                  linear index of the node in question
 * @param[in]   horizontal_nodes_domain     the amount of horizontal nodes in the unexpanded domain
 *
 * @return  the linearized index of the according results
 */
inline unsigned int get_result_index(const unsigned int node_index, const unsigned int horizontal_nodes)
{
    return (((node_index - horizontal_nodes) / horizontal_nodes) * (horizontal_nodes - 2)
            + (node_index - 1) % horizontal_nodes);
}

// DECOMPOSED DOMAINS /////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   This namespace contains functions for accessing nodes and results within decomposed domains,
 *          both buffered and non-buffered.
 */
namespace decomposed
{

/**
 * @brief   This struct contains a static method to retrieve the linearized index of a node within a
 *          non-buffered domain. It is compatible with both expanded and unexpanded domains.
 */
struct NonBufferedNodeAccess
{
    /**
     * @brief   Retrieves the linearized index of a node with the specified global coordinates. The
     *          domain is organized into subdomains of the specified size, and horizontally has the
     *          specified amount of total nodes after a potential extension.
     *
     * @param[in]   global_x                    the global x coordinate of the node in question
     * @param[in]   global_y                    the global y coordinate of the node in question
     * @param[in]   subdomain_vertical_nodes    the amount of vertical nodes per subdomain
     * @param[in]   subdomain_horizontal_nodes  the amount of horizontal nodes per subdomain
     * @param[in]   extended_horizontal_nodes   the total amount of nodes in horizontal direction
     *                                          considering a possible domain extension
     *
     * @return  the linearized index of the node in question.
     */
    inline static unsigned int
    get_index(const unsigned int global_x,
              const unsigned int global_y,
              const unsigned int subdomain_vertical_nodes,
              const unsigned int subdomain_horizontal_nodes,
              const unsigned int extended_horizontal_nodes)
    {
        return global_x + global_y * extended_horizontal_nodes;
    }
};

/**
 * @brief   This struct contains a static method to retrieve the linearized index of a node within a
 *          buffered domain. Such domains are always extended to match the work group size specified by
 *          the device but they are also extended by a buffer grid.
 */
struct BufferedNodeAccess
{
    /**
     * @brief   Retrieves the linearized index of a node with the specified global coordinates. The
     *          domain is organized into subdomains of the specified size, and horizontally has the
     *          specified amount of total nodes after extension.
     *
     * @param[in]   global_x                    the global x coordinate of the node in question
     * @param[in]   global_y                    the global y coordinate of the node in question
     * @param[in]   subdomain_vertical_nodes    the amount of vertical nodes per subdomain
     * @param[in]   subdomain_horizontal_nodes  the amount of horizontal nodes per subdomain
     * @param[in]   extended_horizontal_nodes   the total amount of nodes in horizontal direction
     *                                          considering domain extension
     *
     * @return  the linearized index of the node in question.
     */
    inline static unsigned int
    get_index(const unsigned int x,
              const unsigned int y,
              const unsigned int subdomain_vertical_nodes,
              const unsigned int subdomain_horizontal_nodes,
              const unsigned int extended_horizontal_nodes)
    {
        return (x + ((x - 1) / subdomain_horizontal_nodes))
               + (y + ((y - 1) / subdomain_vertical_nodes)) * extended_horizontal_nodes;
    }
};

/**
 * @brief   This concept allows to specify a node accessor class for decomposed domains through a
 *          template parameter.
 *
 * @tparam  T   this class is tested for membership in the set of accessors during compile time
 */
template <class T>
concept NodeAccessor = std::same_as<T, NonBufferedNodeAccess> || std::same_as<T, BufferedNodeAccess>;

}  // namespace decomposed

}  // namespace access

}  // namespace core

}  // namespace lbm

#endif  // ! LBM_ACCESS_HPP
