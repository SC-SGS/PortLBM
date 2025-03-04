/**
 * @file        optimized_two_lattice_kernels.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of kernels for the two-lattice algorithm
 *              with linear work item evaluation.
 * 
 * @version     1.1
 * 
 * @date        February 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef OPTIMIZED_TWO_LATTICE_KERNELS_HPP
#define OPTIMIZED_TWO_LATTICE_KERNELS_HPP

// Dependencies on other LBM core features
#include "../../../core/access.hpp"
#include "../../../core/constants.hpp"
#include "../../../core/simulation.hpp"

// SYCL
#include <sycl/sycl.hpp>
#include <limits>

namespace lbm
{

    namespace gpu
    {

        namespace two_lattice
        {

            /**
             * @brief   This namespace contains all two-lattice kernels that operate on a linear work item layout.
             *          That is, work item indices are assigned linearly, no work group structuring is introduced,
             *          and no padding is applied since none is necessary.
             */
            namespace optimized
            {

                /**
                 * @brief This namespace contains all kernels for the two-lattice algorithm.
                 */
                namespace kernels
                {

// Performance kernels ////////////////////////////////////////////////////////////////////////////////////////////////

                    /**
                     * @brief   This kernel performs the streaming step, the update of the macroscopic observables and 
                     *          collision step of a two-lattice iteration.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */          
                    template<core::access::AccessorConcept A>
                    class StreamCollideKernel
                    {
                        private:

                        int8_t *phase_information;
                        real_type *source;
                        real_type *destination;

                        real_type *densities;
                        real_type *x_velocities;
                        real_type *y_velocities;
                        real_type *absolute_velocity_values;

                        unsigned int vertical_nodes_expanded;
                        unsigned int horizontal_nodes_expanded;
                        unsigned int horizontal_nodes_domain;
                        real_type relaxation_time_inverse;

                        public:

                        /**
                         * @brief Constructor for a new `StreamCollideKernel` object.
                         *        Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                         * 
                         * @param[in]   simulation  the structure containing all simulation data
                         */
                        StreamCollideKernel(const core::Simulation &simulation):
                        phase_information(simulation.data->phase_information),
                        source(simulation.data->distribution_values_0),
                        destination(simulation.data->distribution_values_1),
                        densities(simulation.results->densities_gpu),
                        x_velocities(simulation.results->x_velocities_gpu),
                        y_velocities(simulation.results->y_velocities_gpu),
                        absolute_velocity_values(simulation.results->absolute_velocities_gpu),
                        vertical_nodes_expanded(simulation.domain->vertical_nodes),
                        horizontal_nodes_expanded(simulation.domain->horizontal_nodes),
                        horizontal_nodes_domain(simulation.properties->horizontal_nodes),
                        relaxation_time_inverse(1 / simulation.properties->relaxation_time)
                        {}

                        /**
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            // Determine global indices
                            auto global_id_x = nd_item.get_global_id(1) + 1 + (nd_item.get_global_id(1) / nd_item.get_local_range(1));
                            auto global_id_y = nd_item.get_global_id(0) + 1 + (nd_item.get_global_id(0) / nd_item.get_local_range(0));
                            auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            // Only do something for fluid nodes
                            if(!phase_information[linear_index])
                            {
                                // Get index of results vector
                                unsigned int iteration_node_offset =
                                lbm::core::access::decomposed::get_results_index(
                                    (global_id_x) - (global_id_x / (nd_item.get_local_range(1) + 1)),
                                    (global_id_y) - (global_id_y / (nd_item.get_local_range(0) + 1)),
                                    horizontal_nodes_domain
                                );

                                // This private array acts as a "buffer" for the distribution values, effectively replacing a second grid 
                                real_type distribution_values[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

                                // Macroscopic observable declarations
                                real_type result = 0;
                                real_type density = 0;
                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 
                                real_type flow_velocity_x = 0;
                                real_type flow_velocity_y = 0;
                                real_type absolute_velocity = 0;

                                // Get incoming distribution values
                                for (const auto& direction : core::constants::all_directions)
                                {
                                    distribution_values[direction] = 
                                        source[
                                            A::at(
                                                lbm::core::access::get_neighbor(linear_index, 8 - direction, horizontal_nodes_expanded), 
                                                direction, 
                                                horizontal_nodes_expanded * vertical_nodes_expanded
                                            )
                                        ];
                                }

                                nd_item.barrier();

                                // Calculate macroscopic observables
                                for (const auto& direction : core::constants::all_directions)
                                {
                                    // Macroscopic observables
                                    density += distribution_values[direction];
                                    velocity_x_component = direction % 3 - 1; 
                                    velocity_y_component = direction / 3 - 1; 
                                    flow_velocity_x += distribution_values[direction] * velocity_x_component;
                                    flow_velocity_y += distribution_values[direction] * velocity_y_component;
                                }

                                absolute_velocity =
                                    sycl::sqrt(flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y);
                                
                                nd_item.barrier();

                                // Streaming and collision
                                for (const auto& direction : core::constants::all_directions)
                                {
                                    velocity_x_component = (direction % 3) - 1; 
                                    velocity_y_component = (direction / 3) - 1; 

                                    result = core::constants::weights[direction] *
                                        (
                                            density + 3 * (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)
                                            + 9.0/2 *
                                            (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)*
                                            (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)
                                            - 3.0/2 * (flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y)
                                        );

                                    result =    -relaxation_time_inverse * (distribution_values[direction] - result) 
                                                + distribution_values[direction];

                                    source[
                                            A::at(
                                                linear_index, 
                                                direction, 
                                                horizontal_nodes_expanded * vertical_nodes_expanded
                                            )
                                        ] = result;
                                }

                                #ifdef WITH_NAN_PROTECTION 

                                if(sycl::isnan(density) || density > std::numeric_limits<float>::max()) 
                                    density = 0;
                                if(sycl::isnan(flow_velocity_x) || flow_velocity_x > std::numeric_limits<float>::max()) 
                                    flow_velocity_x = 0;
                                if(sycl::isnan(flow_velocity_y) || flow_velocity_y > std::numeric_limits<float>::max()) 
                                    flow_velocity_y = 0;
                                if(sycl::isnan(absolute_velocity) || absolute_velocity > std::numeric_limits<float>::max()) 
                                    absolute_velocity = 0;

                                #endif

                                // Update macroscopic observables
                                densities[iteration_node_offset] = density;
                                x_velocities[iteration_node_offset] = flow_velocity_x;
                                y_velocities[iteration_node_offset] = flow_velocity_y;
                                absolute_velocity_values[iteration_node_offset] = absolute_velocity;
                            }
                        }
                    };

                    /**
                     * @brief   Kernel for emplacing the bounce-back values.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class EmplaceBounceBackKernel
                    {
                        private:

                        int8_t *phase_information;
                        real_type *destination;

                        unsigned int horizontal_nodes;
                        unsigned int horizontal_nodes_original;
                        unsigned int total_nodes;

                        unsigned int subdomain_horizontal_nodes;
                        unsigned int subdomain_vertical_nodes;

                        public:

                        EmplaceBounceBackKernel(const core::Simulation &simulation):
                        phase_information(simulation.data->phase_information),
                        destination(simulation.data->distribution_values_0),
                        horizontal_nodes(simulation.domain->horizontal_nodes),
                        horizontal_nodes_original(simulation.properties->horizontal_nodes),
                        total_nodes(simulation.domain->total_node_count),
                        subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes),
                        subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes)
                        {}

                        void operator()(const sycl::nd_item<2> &nd_item) const
                        {
                            auto global_id_x = nd_item.get_global_id(1) + 1;
                            auto global_id_y = nd_item.get_global_id(0) + 1;
                            auto linear_index = 
                                core::access::decomposed::BufferedNodeAccess::get_index(
                                    global_id_x, 
                                    global_id_y, 
                                    subdomain_vertical_nodes, 
                                    subdomain_horizontal_nodes, 
                                    horizontal_nodes,
                                    horizontal_nodes_original
                                );

                            if(phase_information[linear_index] == 1)
                            {
                                for(const auto& dir : core::constants::streaming_directions)
                                {
                                    destination[A::at(linear_index, dir, total_nodes)] =
                                    destination[
                                        A::at(
                                            core::access::get_neighbor(linear_index, dir, horizontal_nodes), 
                                            8 - dir, 
                                            total_nodes
                                        )
                                    ];           
                                }
                            }
                        }
                    };

                } // ! namespace optimized

            } // ! namespace kernels

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! OPTIMIZED_TWO_LATTICE_KERNELS_HPP
