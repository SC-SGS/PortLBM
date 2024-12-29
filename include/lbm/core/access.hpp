/**
 * @file        access.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Updated version of the lattice Boltzmann access functions first introduced in my SimTech project work:
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann.
 *              The major change is that accessor objects are used to achieve different access patterns since SYCL
 *              is not compatible with function pointers.
 *              The data layouts were proposed by Mattila et al.
 * 
 * @version     2.2
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef ACCESS_HPP
#define ACCESS_HPP

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
         * @brief This namespace contains accessor classes for different data layouts and general functions
         *        mapping input data to node indices.
         */
        namespace access
        {
            /**
             * @brief This abstract non-instanciable accessor class defines the root of an accessor hierarchy.
             *        It is used as the base type of valid accessor objects.
             *        These accessor objects are used to implement substituable access functions.
             */
            class LBMAccessorObject
            {
                protected:

                    /**
                     * @brief This constructor is only available to child classes to set up the amount of
                     *        horizontal nodes. It is not available otherwise, and no instances of this type
                     *        can be constructed.
                     * 
                     * @param[in] horizontal_nodes the amount of horizontal nodes in the lattice
                     */
                    explicit LBMAccessorObject(const unsigned int horizontal_nodes);

                public:

                    unsigned int horizontal_nodes;

                    constexpr static std::string_view layout_string = "generic LBM accessor";

                    /**
                     * @brief Retrieves the index of the distribution value within the according vector
                     *        following the data layout of this accessor object.
                     * 
                     * @param[in] node      index of the node 
                     * @param[in] direction index of the direction
                     * 
                     * @return the index of the corresponding distribution value
                     */
                    virtual unsigned int operator()(const unsigned int node, const unsigned int direction) const = 0;
            };

            /**
             * @brief Instances of this class model the access of distribution values according to the collision data layout.
             */
            class LBMCollisionAccessor : public LBMAccessorObject
            {
                public:

                    /**
                     * @brief Constructs a new collision accessor object.
                     * 
                     * @param[in] horizontal_nodes the amount of horizontal nodes in the lattice
                     */
                    explicit LBMCollisionAccessor(const unsigned int horizontal_nodes);

                    constexpr static std::string_view layout_string = "collision";

                    /**
                     * @brief Retrieves the index of the distribution value within the according vector
                     *        following the collision layout.
                     * 
                     * @param[in] node      index of the node 
                     * @param[in] direction index of the direction
                     * 
                     * @return the index of the corresponding distribution value
                     */
                    inline unsigned int operator()(const unsigned int node, const unsigned int direction) const override 
                    {
                        return 9 * node + direction;  
                    }
            };

            /**
             * @brief Instances of this class model the access of distribution values according to the stream data layout.
             */
            class LBMStreamAccessor : public LBMAccessorObject
            {
                unsigned int buffered_node_count;

                public:

                    /**
                     * @brief Constructs a new stream accessor object.
                     * 
                     * @param[in] horizontal_nodes      the amount of horizontal nodes in the lattice
                     * @param[in] buffered_node_count   the total amount of nodes in the lattice including buffers
                     */
                    explicit LBMStreamAccessor(const unsigned int horizontal_nodes, const unsigned int buffered_node_count); 

                    constexpr static std::string_view layout_string = "stream";

                    /**
                     * @brief Retrieves the index of the distribution value within the according vector
                     *        following the stream layout.
                     * 
                     * @param[in] node      index of the node 
                     * @param[in] direction index of the direction
                     * 
                     * @return the index of the corresponding distribution value
                     */
                    inline unsigned int operator()(const unsigned int node, const unsigned int direction) const override 
                    {
                        return buffered_node_count * direction + node;
                    }
            };

            /**
             * @brief Instances of this class model the access of distribution values according to the bundle data layout.
             */
            class LBMBundleAccessor : public LBMAccessorObject
            {
                unsigned int buffered_node_count;

                public:

                    /**
                     * @brief Constructs a new bundle accessor object.
                     * 
                     * @param[in] horizontal_nodes      the amount of horizontal nodes in the lattice
                     * @param[in] buffered_node_count   the total amount of nodes in the lattice including buffers
                     */
                    explicit LBMBundleAccessor(const unsigned int horizontal_nodes, const unsigned int buffered_node_count); 

                    constexpr static std::string_view layout_string = "bundle";

                    /**
                     * @brief Retrieves the index of the distribution value within the according vector
                     *        following the bundle layout.
                     * 
                     * @param[in] node      index of the node 
                     * @param[in] direction index of the direction
                     * 
                     * @return the index of the corresponding distribution value
                     */
                    inline unsigned int operator()(const unsigned int node, const unsigned int direction) const override 
                    {
                        return 3 * (direction / 3) * buffered_node_count + (direction % 3) + 3 * node; 
                    }
            };

            namespace experimental
            {
                /**
                 * @brief Models the access of distribution values according to the collision data layout.
                 */
                struct CollisionAccessor
                {
                    static constexpr std::string_view layout_string = "collision";

                    /**
                     * @brief Retrieves the index of the distribution value within the according vector
                     *        following the collision layout.
                     * 
                     * @param[in] node      index of the node 
                     * @param[in] direction index of the direction
                     * 
                     * @return the index of the corresponding distribution value
                     */
                    static inline
                    unsigned int at(const unsigned int node, const unsigned int direction, const unsigned int total_buffered_node_count)
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
                     * @brief Retrieves the index of the distribution value within the according vector
                     *        following the stream layout.
                     * 
                     * @param[in] node                          index of the node 
                     * @param[in] direction                     index of the direction
                     * @param[in] total_buffered_node_count     total amount of nodes in the lattice including buffers and ghosts
                     * 
                     * @return the index of the corresponding distribution value
                     */
                    static inline
                    unsigned int at(const unsigned int node, const unsigned int direction, const unsigned int total_buffered_node_count) 
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
                     * @brief Retrieves the index of the distribution value within the according vector
                     *        following the stream layout.
                     * 
                     * @param[in] node                          index of the node 
                     * @param[in] direction                     index of the direction
                     * @param[in] total_buffered_node_count     total amount of nodes in the lattice including buffers and ghosts
                     * 
                     * @return the index of the corresponding distribution value
                     */
                    static inline
                    unsigned int at(const unsigned int node, const unsigned int direction, const unsigned int total_buffered_node_count) 
                    {
                        return 3 * (direction / 3) * total_buffered_node_count + (direction % 3) + 3 * node; 
                    }
                };

                template <class T>
                concept AccessorConcept = std::same_as<T, CollisionAccessor> || std::same_as<T, StreamAccessor> || std::same_as<T, BundleAccessor>;

            } // ! namespace experimental
            
            /**
            * @brief Retrieves the coordinates of the node with the specified node index.
            * 
            * @param[in] node_index         the index of the node in question
            * @param[in] horizontal_nodes   the amount of horizontal nodes in the lattice
            * 
            * @return a tuple containing the x and y coordinate of the specified node.
            */
            inline std::array<unsigned int, 2> get_node_coordinates
            (
                const unsigned int node_index,
                const unsigned int horizontal_nodes
            )
            {
                return std::array{node_index % horizontal_nodes, node_index / horizontal_nodes};
            }

            /**
             * @brief Returns the index of the neighbor that is reached when moving in the specified direction.
             * 
             * @param[in] node_index        the index of the current node
             * @param[in] direction         the direction of movement
             * @param[in] horizontal_nodes  the amount of horizontal nodes in the lattice
             * 
             * @return the node index of the neighbor
             */
            inline unsigned int get_neighbor
            (
                const unsigned int node_index, 
                const unsigned int direction,
                const unsigned int horizontal_nodes
            )
            {
                int y_offset = direction / 3 - 1;                           // -1 for {0,1,2}, 0 for {3,4,5}, 1 for {6,7,8}
                int x_offset = direction % 3 - 1;                           // -1 for {0,3,6}, 0 for {1,4,7}, 1 for {2,5,8}
                return node_index + y_offset * horizontal_nodes + x_offset;
            }   

            /**
             * @brief Returns the index the desired node has within the array that stores it. 
             *        The origin lies at the lower left corner and enumeration is row-major.
             * 
             * @param[in] x                 x coordinate
             * @param[in] y                 y coordinate
             * @param[in] horizontal_nodes  the amount of horizontal nodes in the lattice
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
             * @brief This function returns the distribution values of the node with the specified index 
             *        using the specified accessor object.
             * 
             * @param[in] source        the distribution values will be read from this vector
             * @param[in] node_index    this is the index of the node in the domain
             * 
             * @return a vector containing the distribution values
             */
            template<experimental::AccessorConcept A> inline 
            std::vector<double> get_distribution_values_of
            (
                const std::vector<double> &source, 
                const unsigned int node_index,
                const unsigned int total_node_count
            )
            {
                std::vector<double> dist_vals(9,0);
                for(auto direction = 0; direction < 9; ++direction)
                {
                    dist_vals[direction] = source[A::at(node_index, direction, total_node_count)];
                }
                return dist_vals;
            }

            /**
             * @brief This function sets all distribution values of the node with the specified index 
             *        to the specified values according to the access pattern specified by the accessor object.
             * 
             * @param[in] dist_vals     a vector containing the values to which the distribution values shall be set
             * @param[in] node_index    this is the index of the node in the domain
             * @param[in] destination   the distribution values will be written to this vector
             */
            template<experimental::AccessorConcept A> inline
            void set_distribution_values_of
            (
                const std::vector<double> &dist_vals, 
                const unsigned int node_index, 
                const unsigned int total_node_count,
                std::vector<double> &destination
            )
            {
                for(auto direction = 0; direction < 9; ++direction)
                {
                    destination[A::at(node_index, direction, total_node_count)] = dist_vals[direction];
                }
            }

            /**
             * @brief This namespace contains functions for the access of simulation results within the
             *        SimulationResults structure.
             */
            namespace results
            {
                /**
                 * @brief Determines the index of the entry that the specified node index is mapped to
                 *        in the specified time step.
                 * 
                 * @param[in] node_index        index of the node in question
                 * @param[in] total_node_count  total amount of nodes in the lattice including buffer and ghost nodes
                 * @param[in] time_step         the time step to which the value belongs
                 * 
                 * @return the index of the respective value in any vector within the SimulationResults structure     
                 */
                inline unsigned int get_result_index
                (
                    const unsigned int node_index,
                    const unsigned int total_node_count,
                    const unsigned int time_step
                )
                {
                    return node_index + time_step * total_node_count;
                }

                /**
                 * @brief Determines the index of any result array that a macroscopic observable of the node with the
                 *        specified index is mapped to at the specified time if ghost nodes are ignored.
                 *        That is, this function is used if no values are stored for the outer "halo".
                 * 
                 * @param[in] node_index        index of the node in question
                 * @param[in] horizontal_nodes  the total amount of horizontal nodes within the domain including ghost nodes
                 * @param[in] domain_node_count the total amount of nodes belonging to the actual simulation domain, i.e. excluding the halo
                 * @param[in] time_step         the time step to which the value belongs
                 * 
                 * @return the index of the respective value in any vector within the SimulationResults structure     
                 */
                inline unsigned int get_result_index_no_ghosts
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
                 * @brief Determines the index of any result array that a macroscopic observable of the node with the
                 *        specified index if ghost nodes are ignored.
                 *        That is, this function is used if no values are stored for the outer "halo".
                 * 
                 * @param[in] node_index        index of the node in question
                 * @param[in] horizontal_nodes  the total amount of horizontal nodes within the domain including ghost nodes
                 * @param[in] domain_node_count the total amount of nodes belonging to the actual simulation domain, i.e. excluding the halo
                 * 
                 * @return the index of the respective value in any vector within the SimulationResults structure     
                 */
                inline unsigned int get_result_index_no_ghosts
                (
                    const unsigned int node_index,
                    const unsigned int horizontal_nodes
                )
                {
                    return (((node_index - horizontal_nodes) / horizontal_nodes) * (horizontal_nodes - 2) 
                        + (node_index - 1) % horizontal_nodes);
                }
                
            } // ! namespace results

        } // ! namespace access

    } // ! namespace core

} // ! namespace lbm

#endif // ! ACCESS_HPP
