/**
 * @file        access.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the definitions of classes modelling data layouts for storing distribution 
 *              values proposed by Mattila et al, as well as functions for retrieving the index of nodes and simulation
 *              results for domains with or without decomposition and buffering.
 * 
 * @version     2.4
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_ACCESS_HPP
#define LBM_ACCESS_HPP

// LBM core functionality
#include "constants.hpp"

// Standard library
#include <string_view>
#include <tuple>
#include <vector>
#include <array>
#include <concepts>

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
            /**
             * @brief   Models the access of distribution values according to the collision data layout.
             */
            struct CollisionAccessor
            {
                static constexpr std::string_view layout_string = "collision";

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
                static inline
                unsigned int at
                (
                    const unsigned int node, 
                    const unsigned int direction, 
                    const unsigned int total_buffered_node_count
                )
                {
                    return 9 * node + direction;
                }
            };

            /**
             * @brief Models the access of distribution values according to the stream data layout.
             */
            struct StreamAccessor
            {
                static constexpr std::string_view layout_string = "stream";

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
                static inline
                unsigned int at
                (
                    const unsigned int node, 
                    const unsigned int direction, 
                    const unsigned int total_buffered_node_count
                ) 
                {
                    return total_buffered_node_count * direction + node;
                }
            };

            /**
             * @brief Models the access of distribution values according to the bundle data layout.
             */
            struct BundleAccessor
            {
                static constexpr std::string_view layout_string = "bundle";

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
                static inline
                unsigned int at
                (
                    const unsigned int node, 
                    const unsigned int direction, 
                    const unsigned int total_buffered_node_count
                ) 
                {
                    return 3 * (direction / 3) * total_buffered_node_count + (direction % 3) + 3 * node; 
                }
            };

            /**
             * @brief   This concept is used to restrict the template parameter of methods relying on an accessor 
             *          function. During compile time, the compatibility of the specified class is checked.
             *          Any methods or classes operating on distribution values must use a data layout and an according
             *          access pattern. The intended use is to specify both as a template parameter accepting any 
             *          `AccessorConcept`, that is, any of the following classes:
             * 
             *          - `lbm::core::access::CollisionAccessor`
             * 
             *          - `lbm::core::access::StreamAccessor`
             * 
             *          - `lbm::core::access::BundleAccessor`
             * 
             * @tparam T this class is tested for membership in the set of accessors during compile time
             */
            template <class T>
            concept AccessorConcept = 
                std::same_as<T, CollisionAccessor> || 
                std::same_as<T, StreamAccessor> || 
                std::same_as<T, BundleAccessor>;
            
            /**
            * @brief    Retrieves the coordinates of the node with the specified node index.
            * 
            * @param[in]    node_index         the index of the node in question
            * @param[in]    horizontal_nodes   the amount of horizontal nodes in the lattice
            * 
            * @return a two-dimensional array containing the x and y coordinate of the specified node.
            */
            inline 
            std::array<unsigned int, 2> get_node_coordinates
            (
                const unsigned int node_index,
                const unsigned int horizontal_nodes
            )
            {
                return std::array{node_index % horizontal_nodes, node_index / horizontal_nodes};
            }

            /**
             * @brief   Returns the index of the neighbor that is reached when moving in the specified direction.
             * 
             * @param[in]   node_index        the index of the current node
             * @param[in]   direction         the direction of movement
             * @param[in]   horizontal_nodes  the amount of horizontal nodes in the lattice
             * 
             * @return the node index of the neighbor
             */
            inline 
            unsigned int get_neighbor
            (
                const unsigned int node_index, 
                const unsigned int direction,
                const unsigned int horizontal_nodes
            )
            {
                int y_offset = direction / 3 - 1;                           
                // -1 for {0,1,2}, 0 for {3,4,5}, 1 for {6,7,8}

                int x_offset = direction % 3 - 1;                           
                // -1 for {0,3,6}, 0 for {1,4,7}, 1 for {2,5,8}

                return node_index + y_offset * horizontal_nodes + x_offset;
            }   

            /**
             * @brief   Returns the index the desired node has within the array that stores it. The origin lies at the
             *          lower left corner and enumeration is row-major.
             * 
             * @param[in]   x                 x coordinate
             * @param[in]   y                 y coordinate
             * @param[in]   horizontal_nodes  the amount of horizontal nodes in the lattice
             * 
             * @return the index of the desired note
             */
            inline unsigned int get_node_index
            (
                const unsigned int x, 
                const unsigned int y,
                const unsigned int horizontal_nodes
            )
            {
                return x + y * horizontal_nodes;
            }

            /**
             * @brief   This namespace contains functions for the access of simulation results.
             */
            namespace results
            {
                /**
                 * @brief   Determines the index of any result array that a macroscopic observable of the node with the
                 *          specified index is mapped to at the specified time if ghost nodes are ignored.
                 *          That is, this function is used if no values are stored for the outer "halo".
                 * 
                 * @param[in]   node_index          index of the node in question
                 * @param[in]   horizontal_nodes    the total amount of horizontal nodes within the domain including
                 *                                  ghost nodes
                 * @param[in]   domain_node_count   the total amount of nodes belonging to the actual simulation domain,
                 *                                  i.e. excluding the halo
                 * @param[in]   time_step           the time step to which the value belongs
                 * 
                 * @return the index of the respective value in any vector within the SimulationResults structure     
                 */
                inline unsigned int get_result_index
                (
                    const unsigned int node_index,
                    const unsigned int horizontal_nodes,
                    const unsigned int domain_node_count,
                    const unsigned int time_step
                )
                {
                    return (((node_index - horizontal_nodes) / horizontal_nodes) * (horizontal_nodes - 2) 
                        + (node_index - 1) % horizontal_nodes) + time_step * domain_node_count;
                }

                /**
                 * @brief   Determines the index of any result array that a macroscopic observable of the node with the
                 *          specified index if ghost nodes are ignored. That is, this function is used if no values are 
                 *          stored for the outer "halo".
                 * 
                 * @param[in]   node_index          index of the node in question
                 * @param[in]   horizontal_nodes    the total amount of horizontal nodes within the domain including
                 *                                  ghost nodes
                 * 
                 * @return the index of the respective value in any vector within the SimulationResults structure     
                 */
                inline unsigned int get_result_index
                (
                    const unsigned int node_index,
                    const unsigned int horizontal_nodes
                )
                {
                    return (((node_index - horizontal_nodes) / horizontal_nodes) * (horizontal_nodes - 2) 
                        + (node_index - 1) % horizontal_nodes);
                }
                
            } // ! namespace results

            /**
             * @brief   This namespace contains functions for accessing nodes and results within decomposed domains,
             *          both buffered and non-buffered.
             */
            namespace decomposed
            {

                /**
                 * @brief   Retrieves the linear index of the result belonging to the node with the specified global
                 *          coordinates assuming the specified extent of the non-extended domain.
                 * 
                 * @param[in]   global_x                    the global x coordinate of the node in question
                 * @param[in]   global_y                    the global y coordinate of the node in question
                 * @param[in]   horizontal_nodes_domain     the amount of horizontal nodes in the unexpanded domain
                 * 
                 * @return  the linearized index of the according results
                 */
                inline unsigned int get_results_index
                (
                    const unsigned int global_x,
                    const unsigned int global_y,
                    const unsigned int horizontal_nodes_domain
                )
                {
                    return (global_x - 1) + (global_y - 1) * (horizontal_nodes_domain - 2);
                }

                /**
                 * @brief   This struct contains a static method to retrieve the linearized index of a node within a
                 *          non-buffered domain. It does not matter whether or not the domain was expanded or not.
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
                    static inline
                    unsigned int get_index
                    (
                        const unsigned int global_x,
                        const unsigned int global_y,
                        const unsigned int subdomain_vertical_nodes,
                        const unsigned int subdomain_horizontal_nodes,
                        const unsigned int extended_horizontal_nodes,
                        const unsigned int unextended_horizontal_nodes
                    )
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
                    static inline
                    unsigned int get_index
                    (
                        const unsigned int x,
                        const unsigned int y,
                        const unsigned int subdomain_vertical_nodes,
                        const unsigned int subdomain_horizontal_nodes,
                        const unsigned int extended_horizontal_nodes,
                        const unsigned int unextended_horizontal_nodes
                    )
                    {
                        return (x + ((x - 1) / subdomain_horizontal_nodes)) + 
                            (y + ((y - 1) / subdomain_vertical_nodes)) * extended_horizontal_nodes;
                    }
                };

                /**
                 * @brief   This concept allows to specify a node accessor class dor decomposed domains through a
                 *          template parameter.
                 * 
                 * @tparam  T   this class is tested for membership in the set of accessors during compile time
                 */
                template <class T>
                concept NodeAccessor = std::same_as<T, NonBufferedNodeAccess> || std::same_as<T, BufferedNodeAccess>;

            } // ! namespace decomposed

        } // ! namespace access

    } // ! namespace core

} // ! namespace lbm

#endif // ! ACCESS_HPP
