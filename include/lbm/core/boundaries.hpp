/**
 * @file        boundaries.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of functionality related to the
 *              treatment of boundary conditions.
 * 
 * @version     1.2
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include "macroscopic.hpp"
#include "simulation.hpp"
#include "access.hpp"
#include "../console/console_output.hpp"

namespace lbm
{

    namespace core
    {

        /**
         * @brief Returns whether the node in question borders the ghost nodes outside the actual domain.
         * 
         * @param[in] node_index        the index of the node in question
         * @param[in] vertical_nodes    the amount of vertical nodes including ghost nodes and buffers
         * @param[in] horizontal_nodes  the amount of horizontal nodes including ghost nodes and buffers
         * 
         * @return  whether or not a node is an edge node
         */
        inline bool is_edge_node
        (
            const unsigned int node_index, 
            const unsigned int vertical_nodes, 
            const unsigned int horizontal_nodes
        )
        {
            std::array<unsigned int, 2> coordinates = access::get_node_coordinates(node_index, horizontal_nodes);
            unsigned int x = coordinates[0];
            unsigned int y = coordinates[1];
            return ((x == 1) || (x == (horizontal_nodes - 2))) && ((y == 1) || (y == (vertical_nodes - 2)));
        }

        /**
         * @brief Returns whether the node with the specified index is a ghost node.
         *        This is the case when at least one of the node coordinates is zero or the maximum extent 
         *        in the according dimension, or if the node is solid.
         * 
         * @param[in] node_index        index of the node in question
         * @param[in] properties        the properties structure containing the domain extents
         * @param[in] phase_information whether the node is solid (true) or fluid (false)
         * 
         * @return whether or not the node is a ghost node
         */
        inline bool is_ghost_node
        (
            const unsigned int node_index, 
            const Properties &properties,
            const uint8_t phase_information
        )
        {
            if(phase_information)
            {
                return true;
            }
            else
            {
                std::array<unsigned int, 2> coordinates = access::get_node_coordinates(node_index, properties.horizontal_nodes);
                unsigned int x = coordinates[0];
                unsigned int y = coordinates[1];
                return ((x == 0) || (x == (properties.horizontal_nodes - 1))) || ((y == 0) || (y == (properties.vertical_nodes - 1)));
            }
        }

        inline bool is_ghost_node
        (
            const unsigned int node_index, 
            const unsigned int vertical_nodes,
            const unsigned int horizontal_nodes,
            const uint8_t phase_information
        )
        {
            if(phase_information)
            {
                return true;
            }
            else
            {
                std::array<unsigned int, 2> coordinates = access::get_node_coordinates(node_index, horizontal_nodes);
                unsigned int x = coordinates[0];
                unsigned int y = coordinates[1];
                return ((x == 0) || (x == (horizontal_nodes - 1))) || ((y == 0) || (y == (vertical_nodes - 1)));
            }
        }

        /**
         * @deprecated This method was used to determine the bounce-back directions excluding inlet and outlet nodes in the CPU implementation.
         *             Since the bounce-back semantics are slightly different for the GPU implementation (bounce-back is processed per solid node), 
         *             this method will not be needed in the future.
         * 
         * @brief This convenience method behaves like `is_ghost_node(const unsigned int, const Properties&, const bool)`
         *        with the exception that inlet and outlet nodes are not considered.
         * 
         * @param[in] node_index        index of the node in question
         * @param[in] properties        the properties structure containing the domain extents
         * @param[in] phase_information whether the node is solid (true) or fluid (false)
         * 
         * @return true if the node is a non-inlet and non-outlet ghost node, and false otherwise
         */
        inline bool is_non_inout_ghost_node
        (
            unsigned int node_index, 
            const Properties &properties,
            const uint8_t phase_information
        )
        {
                std::array<unsigned int, 2> coordinates = access::get_node_coordinates(node_index, properties.horizontal_nodes);
                unsigned int x = coordinates[0];
                unsigned int y = coordinates[1];
            return ((x != 0) && (x != (properties.horizontal_nodes - 1))) && ((y == 0) || (y == (properties.vertical_nodes - 1)) || phase_information);
        }

        /**
         * @brief This namespace contains function representations of boundary conditions used in the lattice-Boltzmann model.
         *        Notice that parallel versions exist within "parallel_framework.hpp".
         */
        namespace bounce_back
        {

            /**
             * @brief Performs an outstream step for all border nodes in the directions where they border non-inout ghost nodes.
             *        The distribution values will be stored in the ghost nodes in inverted order such that
             *        after this method is executed, the border nodes can be treated like regular nodes when performing an instream.
             * 
             * @tparam T an lbm accessor object, that is, any object whose class inherits from `access::core::LBMAccessorObject`
             * 
             * @param[in]       bsi                 the border swap information
             * @param[in, out]  distribution_values a vector containing the distribution values of all nodes
             * @param[in]       lbm_accessor        the accessor object according to which distribution values are accessed
             */
            /*
            template <access::experimental::LBMAccessor T> 
            void emplace_bounce_back_values
            (
                const lbm::border_swap_information &bsi,
                std::vector<double> &distribution_values
            )
            {                
                for(auto bsi_iterator = bsi.begin(); bsi_iterator < bsi.end(); ++bsi_iterator)
                {
                    for(auto direction_iterator = (*bsi_iterator).begin()+1; direction_iterator < (*bsi_iterator).end(); ++direction_iterator) 
                    {
                        distribution_values[
                            T(
                                access::get_neighbor((*bsi_iterator)[0], *direction_iterator, horizontal_nodes), 
                                invert_direction(*direction_iterator))] 
                                    = distribution_values[T((*bsi_iterator)[0], *direction_iterator)];
                    }
                }
            }
            */

            /**
             * @brief Retrieves the border swap information data structure.
             *        This method does not consider inlet and outlet ghost nodes when performing bounce-back
             *        as the inserted values will be overwritten by inflow and outflow values anyways.
             * 
             * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::core::access::LBMAccessorObject`
             * 
             * @param[in]       properties      the properties structure containing the domain extents   
             * @param[in, out]  simulation_data a structure containing, among others, the phase information
             * 
             * @return an lbm::border_swap_information, i.e. a vector of vectors with the index of a fluid node and bounce-back directions
             */
            /*
            template <access::experimental::LBMAccessor T> 
            lbm::border_swap_information retrieve_border_swap_info
            (
                const Properties &properties,
                Data &simulation_data
            )
            {
                std::vector<unsigned int> current_adjacencies;
                lbm::border_swap_information result;

                for(unsigned int node = 0; node < simulation_data.end_node_index_buffered; ++node)
                {
                    if(!is_ghost_node(node, properties, (*simulation_data.phase_information)[node]))
                    {
                        current_adjacencies = {node};
                        for(const auto direction : constants::streaming_directions)
                        {
                            unsigned int current_neighbor = access::get_neighbor(node, direction, properties.horizontal_nodes);
                            if(is_non_inout_ghost_node(current_neighbor, properties, (*simulation_data.phase_information)[node]))
                            {
                                std::cout << "\tFound non inout ghost node: " << current_neighbor << "\n";
                                current_adjacencies.push_back(direction);
                            }
                        }
                        if(current_adjacencies.size() > 1) result.push_back(current_adjacencies);
                    }
                }
                for(const auto& current : result)
                    lbm::console::print_vector(current, current.size());
                return result;
            }
            */

        } // ! namespace bounce_back

        /**
         * @brief This namespace contains all functions that are required for enforcing boundary conditions.
         *        It uses the ghost nodes for this purpose.
         */
        namespace boundary_conditions
        {

            /**
             * @brief Updates the ghost nodes that represent inlet and outlet edges.
             *        When updating, a velocity border condition will be considered for the input
             *        and a density border condition for the output.
             *        The inlet velocity is constant throughout all inlet nodes whereas the outlet nodes
             *        all have the specified density.
             *        The corresponding values are lbm::constants defined in `constants.hpp`.
             * 
             * @tparam T an lbm accessor object, that is, any object whose class inherits from `access::LBMAccessorObject`
             * 
             * @param[in]       properties          the properties structure containing the inlet and outlet density and velocity 
             * @param[in]       lbm_accessor        the accessor object according to which distribution values are accessed       
             * @param[in, out]  distribution_values a vector containing the distribution values of all nodes
             */
            template <access::experimental::LBMAccessor T>
            void update_velocity_input_density_output
            (
                const Properties &properties,
                std::vector<double> &distribution_values
            )
            {
                std::vector<double> current_dist_vals(9, 0);
                unsigned int current_border_node = 0;
                std::array<double, 2> v = {properties.inlet_velocity_x, properties.inlet_velocity_y};

                for(auto y = 2; y < properties.vertical_nodes - 2; ++y)
                {
                    // Update inlets
                    current_border_node = access::get_node_index(1,y,properties.horizontal_nodes);
                    current_dist_vals = maxwell_boltzmann_distribution(properties.inlet_velocity_x, properties.inlet_velocity_y, properties.inlet_density);

                    access::set_distribution_values_of
                    (
                        current_dist_vals,
                        current_border_node,
                        lbm_accessor,
                        distribution_values
                    );

                    // Update outlets
                    current_border_node = access::get_node_index(properties.horizontal_nodes - 2, y, properties.horizontal_nodes);

                    v = macroscopic::flow_velocity(
                        access::get_distribution_values_of(
                            distribution_values, access::get_neighbor(current_border_node, 3, properties.horizontal_nodes), lbm_accessor));

                    current_dist_vals = maxwell_boltzmann_distribution(v[0], v[1], properties.outlet_density);

                    access::set_distribution_values_of
                    (
                        current_dist_vals,
                        current_border_node,
                        lbm_accessor,
                        distribution_values
                    );
                }
            }

            /**
             * @brief Updates the distribution values of the inlets and outlets.
             *        This implementation is inspired by that of Zou and He but was adapted to suit the half-way bounce-back scheme.
             * 
             * @tparam T an lbm accessor object, that is, any object whose class inherits from `access::LBMAccessorObject`
             * 
             * @param[in]       properties          the properties structure containing the inlet and outlet density and velocity   
             * @param[in, out]  distribution_values a vector containing the distribution values of all nodes
             */
            template <access::experimental::LBMAccessor> void boundary_update
            (
                const Properties &properties,
                std::vector<double> &distribution_values
            )
            {
                std::vector<double> f(9, 0);
                unsigned int current_border_node = 0;
                double x_velocity = 0.0;
                unsigned int iteration_node_offset = 0;

                std::cout << std::setprecision(6) << std::fixed;

                // Update inlets
                for(auto y = 2; y < properties.vertical_nodes - 2; ++y)
                {
                    current_border_node = access::get_node_index(1,y,properties.horizontal_nodes);
                    f[4] = distribution_values[lbm_accessor(current_border_node, 4)];

                    for(auto dir : {0,1,3,6,7})
                    {
                        f[dir] = distribution_values[lbm_accessor(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)];
                    }

                    x_velocity = 1 - (1 / properties.inlet_density) * (f[1] + f[4] + f[7] + 2 * (f[0] + f[3] + f[6]));
                    std::cout << "Velocity: " << x_velocity << "\n";
                    f[2] = f[6] - 0.5 * (f[1] - f[7]) + 1.0/6 * properties.inlet_density * x_velocity;
                    f[5] = f[3] + (2.0 / 3) * properties.inlet_density * x_velocity;
                    f[8] = f[0] + 0.5 * (f[1] - f[7]) + 1.0/6 * properties.inlet_density * x_velocity;

                    for(auto dir : {2,5,8})
                    {
                        distribution_values[lbm_accessor(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)] = f[dir];
                    }

                    std::cout << "Distribution values of node " << current_border_node << "\n";
                    for(int i = 0; i < 9; ++i)
                    {
                        std::cout << "f_" << i << " = " << f[i] << "\n";
                    }
                    std::cout << "\n";

                }

                // Update outlets
                for(auto y = 2; y < properties.vertical_nodes - 2; ++y)
                {
                    current_border_node = access::get_node_index(properties.horizontal_nodes - 2,y,properties.horizontal_nodes);
                    f[4] = distribution_values[lbm_accessor(current_border_node, 4)];

                    for(auto dir : {1,2,5,7,8})
                    {
                        f[dir] = distribution_values[lbm_accessor(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)];
                    }

                    x_velocity = 1 - (1 / properties.outlet_density) * (f[1] + f[4] + f[7] + 2 * (f[2] + f[5] + f[8]));
                    std::cout << "Velocity: " << x_velocity << "\n";
                    f[0] = f[8] - 0.5 * (f[1] - f[7]) + 1.0/6 * properties.outlet_density * x_velocity;
                    f[3] = f[5] + (2.0 / 3) * properties.outlet_density * x_velocity;
                    f[6] = f[2] + 0.5 * (f[1] - f[7]) + 1.0/6 * properties.outlet_density * x_velocity;

                    for(auto dir : {0,3,6})
                    {
                        distribution_values[lbm_accessor(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)] = f[dir];
                    }

                    std::cout << "Distribution values of node " << current_border_node << "\n";
                    for(int i = 0; i < 9; ++i)
                    {
                        std::cout << "f_" << i << " = " << f[i] << "\n";
                    }
                    std::cout << "\n";

                    std::cout << std::setprecision(3) << std::fixed;
                }
            }

        } // ! namespace boundary_conditions

    } // ! namespace core

} // ! namespace lbm

#endif // ! BOUNDARIES_HPP