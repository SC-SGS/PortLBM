// PARTIALLY DEPRECATED

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

// Console output
#include "../console/console_output.hpp"

// Dependencies on other LBM core features
#include "access.hpp"
#include "macroscopic.hpp"
#include "simulation.hpp"

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
            return ((coordinates[0] == 1) || (coordinates[0] == (horizontal_nodes - 2))) && ((coordinates[1] == 1) || (coordinates[1] == (vertical_nodes - 2)));
        }

        /**
         * @brief Returns whether the node with the specified index is a ghost node.
         *        This is the case when at least one of the node coordinates is zero or the maximum extent 
         *        in the according dimension, or if the node is solid.
         * 
         * @param[in] node_index        index of the node in question
         * @param[in] vertical_nodes    the amount of vertical nodes including ghost nodes and buffers
         * @param[in] horizontal_nodes  the amount of horizontal nodes including ghost nodes and buffers
         * @param[in] phase_information whether the node is solid (`true`) or fluid (`false`)
         * 
         * @return whether or not the node is a ghost node
         */
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
                return ((coordinates[0] == 0) || (coordinates[0] == (horizontal_nodes - 1))) || ((coordinates[1] == 0) || (coordinates[1] == (vertical_nodes - 1)));
            }
        }

        /**
         * @brief This namespace contains the sequential CPU-based versions of all functions that are required for enforcing boundary conditions.
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
             * @tparam A any `core::access::experimental::AccessorConcept` from access.hpp
             * 
             * @param[in]       properties          the properties structure containing the inlet and outlet density and velocity    
             * @param[in, out]  distribution_values a vector containing the distribution values of all nodes
             */
            template <access::experimental::AccessorConcept A>
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

                    access::set_distribution_values_of<A>
                    (
                        current_dist_vals,
                        current_border_node,
                        distribution_values
                    );

                    // Update outlets
                    current_border_node = access::get_node_index(properties.horizontal_nodes - 2, y, properties.horizontal_nodes);

                    v = macroscopic::flow_velocity(
                            access::get_distribution_values_of<A>(
                                distribution_values, access::get_neighbor(current_border_node, 3, properties.horizontal_nodes)));

                    current_dist_vals = maxwell_boltzmann_distribution(v[0], v[1], properties.outlet_density);

                    access::set_distribution_values_of<A>
                    (
                        current_dist_vals,
                        current_border_node,
                        distribution_values
                    );
                }
            }

            /**
             * @brief   Updates the distribution values of the inlets and outlets.
             *          This implementation is inspired by that of Zou and He but was adapted to suit the half-way bounce-back scheme.
             * 
             * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp
             * 
             * @param[in]       properties          the properties structure containing the inlet and outlet density and velocity   
             * @param[in, out]  distribution_values a vector containing the distribution values of all nodes
             */
            template <access::experimental::AccessorConcept A> void boundary_update
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
                    f[4] = distribution_values[A::at(current_border_node, 4)];

                    for(auto dir : {0,1,3,6,7})
                    {
                        f[dir] = distribution_values[A::at(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)];
                    }

                    x_velocity = 1 - (1 / properties.inlet_density) * (f[1] + f[4] + f[7] + 2 * (f[0] + f[3] + f[6]));
                    std::cout << "Velocity: " << x_velocity << "\n";
                    f[2] = f[6] - 0.5 * (f[1] - f[7]) + 1.0/6 * properties.inlet_density * x_velocity;
                    f[5] = f[3] + (2.0 / 3) * properties.inlet_density * x_velocity;
                    f[8] = f[0] + 0.5 * (f[1] - f[7]) + 1.0/6 * properties.inlet_density * x_velocity;

                    for(auto dir : {2,5,8})
                    {
                        distribution_values[A::at(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)] = f[dir];
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
                    f[4] = distribution_values[A::at(current_border_node, 4)];

                    for(auto dir : {1,2,5,7,8})
                    {
                        f[dir] = distribution_values[A::at(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)];
                    }

                    x_velocity = 1 - (1 / properties.outlet_density) * (f[1] + f[4] + f[7] + 2 * (f[2] + f[5] + f[8]));
                    std::cout << "Velocity: " << x_velocity << "\n";
                    f[0] = f[8] - 0.5 * (f[1] - f[7]) + 1.0/6 * properties.outlet_density * x_velocity;
                    f[3] = f[5] + (2.0 / 3) * properties.outlet_density * x_velocity;
                    f[6] = f[2] + 0.5 * (f[1] - f[7]) + 1.0/6 * properties.outlet_density * x_velocity;

                    for(auto dir : {0,3,6})
                    {
                        distribution_values[A::at(access::get_neighbor(current_border_node, invert_direction(dir), properties.horizontal_nodes), dir)] = f[dir];
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