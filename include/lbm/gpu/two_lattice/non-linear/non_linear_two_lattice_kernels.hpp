/**
 * @file        non_linear_two_lattice_kernels.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of kernels for the two-lattice algorithm
 *              with non-linear work item evaluation.
 * 
 * @version     1.3
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef NON_LINEAR_TWO_LATTICE_KERNELS_HPP
#define NON_LINEAR_TWO_LATTICE_KERNELS_HPP

// Dependencies on other LBM core features
#include "../../../core/access.hpp"
#include "../../../core/constants.hpp"
#include "../../../core/simulation.hpp"

// SYCL
#include <sycl/sycl.hpp>

// Standard library
#include <limits>

namespace lbm
{

    namespace gpu
    {

        namespace two_lattice
        {

            /**
             * @brief   This namespace contains all two-lattice kernels that operate on a non-linear work item layout.
             */
            namespace non_linear
            {

                /**
                 * @brief This namespace contains all kernels for the non-linear two-lattice algorithm.
                 */
                namespace kernels
                {

// SEPARATE DEBUG KERNELS /////////////////////////////////////////////////////////////////////////////////////////////

                    /**
                     * @brief   This kernel performs the streaming step of a non-linear two-lattice iteration.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class StreamKernel 
                    {

                        private:

                        int8_t *phase_information;
                        real_type *source;
                        real_type *destination;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;

                        public:

                        /**
                         * @brief   Constructor for a new `StreamKernel` object. Create an instance of this kernel and 
                         *          pass it to `cgh.parallel_for(...)`.
                         * 
                         * @param[in]   simulation  the structure containing all simulation data
                         */
                        explicit StreamKernel(const core::Simulation &simulation): 
                        phase_information(simulation.data->phase_information),
                        source(simulation.data->distribution_values_0),
                        destination(simulation.data->distribution_values_1),
                        vertical_nodes(simulation.domain->vertical_nodes),
                        horizontal_nodes(simulation.domain->horizontal_nodes)
                        {}

                        /**
                         * @brief   This overloaded operator is implicitly called to launch the kernel for various work 
                         *          items.
                         * 
                         * @param[in]   nd_item a work item from a two-dimensional SYCL nd-range
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const
                        {
                            auto global_id_x = nd_item.get_global_id(1) + 1;
                            auto global_id_y = nd_item.get_global_id(0) + 1;
                            auto linear_index = 
                                core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);

                            if(!phase_information[linear_index])
                            {
                                for (int direction = 0; direction < 9; ++direction)
                                {
                                    destination[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                                        source[
                                            A::at(
                                                lbm::core::access::get_neighbor(
                                                    linear_index, 8 - direction, horizontal_nodes
                                                ), 
                                                direction, 
                                                horizontal_nodes * vertical_nodes
                                            )
                                        ];
                                }
                            }
                        }
                    };

                    /**
                     * @brief   This kernel performs the update of the macroscopic observables.
                     * 
                     * @tparam  A   any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class MacroscopicObservablesKernel
                    {
                        private:

                        int8_t *phase_information;
                        real_type *destination;

                        real_type *densities;
                        real_type *x_velocities;
                        real_type *y_velocities;
                        real_type *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes_expanded;
                        unsigned int horizontal_nodes_domain;

                        public:

                        /**
                         * @brief   Constructor for a new `MacroscopicObservablesKernel` object. Create an instance of 
                         *          this kernel and pass it to `cgh.parallel_for(...)`.
                         * 
                         * @param[in]   simulation  the structure containing all simulation data
                         */
                        explicit MacroscopicObservablesKernel(const core::Simulation &simulation): 
                        phase_information(simulation.data->phase_information),
                        destination(simulation.data->distribution_values_1),
                        densities(simulation.results->densities_gpu),
                        x_velocities(simulation.results->x_velocities_gpu),
                        y_velocities(simulation.results->y_velocities_gpu),
                        absolute_velocity_values(simulation.results->absolute_velocities_gpu),
                        vertical_nodes(simulation.domain->vertical_nodes),
                        horizontal_nodes_expanded(simulation.domain->horizontal_nodes),
                        horizontal_nodes_domain(simulation.properties->horizontal_nodes)
                        {}

                        /**
                         * @brief   This overloaded operator is implicitly called to launch the kernel for various work 
                         *          items.
                         * 
                         * @param[in]   nd_item a work item from a two-dimensional SYCL nd-range
                         */ 
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            auto global_id_x = nd_item.get_global_id(1) + 1;
                            auto global_id_y = nd_item.get_global_id(0) + 1;
                            auto linear_index = 
                            core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            if(!phase_information[linear_index])
                            {
                                unsigned int iteration_node_offset =
                                    lbm::core::access::get_result_index(
                                        global_id_x, 
                                        global_id_y, 
                                        horizontal_nodes_domain
                                    );

                                real_type dist_vals[9];
                                real_type density = 0;
                                real_type absolute_velocity = 0;
                                sycl::vec<real_type, 2> flow_velocity{0,0};
                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 

                                for (int direction = 0; direction < 9; ++direction)
                                {
                                    dist_vals[direction] = 
                                        destination[
                                            A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes)
                                        ];
                                }
                                
                                for(const auto& i : core::constants::all_directions)
                                {
                                    density += dist_vals[i];
                                    velocity_x_component = i % 3 - 1; 
                                    velocity_y_component = i / 3 - 1; 
                                    flow_velocity[0] += dist_vals[i] * velocity_x_component;
                                    flow_velocity[1] += dist_vals[i] * velocity_y_component;
                                }

                                absolute_velocity = 
                                    sycl::sqrt(flow_velocity[0] * flow_velocity[0] + flow_velocity[1] * flow_velocity[1]);

                                #ifdef WITH_NAN_PROTECTION 

                                if(sycl::isnan(density) || density > std::numeric_limits<float>::max()) 
                                    density = 0;
                                if(sycl::isnan(flow_velocity[0]) || flow_velocity[0] > std::numeric_limits<float>::max()) 
                                    flow_velocity[0] = 0;
                                if(sycl::isnan(flow_velocity[1]) || flow_velocity[1] > std::numeric_limits<float>::max()) 
                                    flow_velocity[1] = 0;
                                if(sycl::isnan(absolute_velocity) || absolute_velocity > std::numeric_limits<float>::max()) 
                                    absolute_velocity = 0;

                                #endif

                                densities[iteration_node_offset] = density;
                                x_velocities[iteration_node_offset] = flow_velocity[0]; 
                                y_velocities[iteration_node_offset] = flow_velocity[1]; 
                                absolute_velocity_values[iteration_node_offset] = absolute_velocity;
                            }
                        }
                    };

                    /**
                     * @brief   This kernel performs the collision step of a two-lattice iteration.
                     * 
                     * @tparam  A   any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A> 
                    class CollideKernel
                    {

                        private:

                        int8_t *phase_information;
                        real_type *destination;

                        real_type *densities;
                        real_type *x_velocities;
                        real_type *y_velocities;
                        real_type *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes_expanded;
                        unsigned int horizontal_nodes_domain;
                        real_type relaxation_time_inverse;

                        public:

                        /**
                         * @brief   Constructor for a new `CollideKernel` object. Create an instance of this kernel and
                         *          pass it to `cgh.parallel_for(...)`.
                         * 
                         * @param[in]   simulation  the structure containing all simulation data
                         */
                        explicit CollideKernel(const core::Simulation &simulation):
                        phase_information(simulation.data->phase_information),
                        destination(simulation.data->distribution_values_1),
                        densities(simulation.results->densities_gpu),
                        x_velocities(simulation.results->x_velocities_gpu),
                        y_velocities(simulation.results->y_velocities_gpu),
                        vertical_nodes(simulation.domain->vertical_nodes),
                        horizontal_nodes_expanded(simulation.domain->horizontal_nodes),
                        horizontal_nodes_domain(simulation.properties->horizontal_nodes),
                        relaxation_time_inverse(1 / simulation.properties->relaxation_time)
                        {}

                        /**
                         * @brief   This overloaded operator is implicitly called to launch the kernel for various work 
                         *          items.
                         * 
                         * @param[in]   nd_item a work item from a two-dimensional SYCL nd-range
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            auto global_id_x = nd_item.get_global_id(1) + 1;
                            auto global_id_y = nd_item.get_global_id(0) + 1;
                            auto linear_index = 
                            core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            if(!phase_information[linear_index])
                            {
                                unsigned int iteration_node_offset =
                                    lbm::core::access::get_result_index(
                                        global_id_x, global_id_y, horizontal_nodes_domain);

                                real_type& x_velocity = x_velocities[iteration_node_offset];
                                real_type& y_velocity = y_velocities[iteration_node_offset];
                                real_type& density = densities[iteration_node_offset];

                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 

                                real_type value;
                                real_type result;

                                for (int direction = 0; direction < 9; ++direction)
                                {
                                    value = 
                                    destination[
                                        A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes)];

                                    velocity_x_component = (direction % 3) - 1; 
                                    velocity_y_component = (direction / 3) - 1; 

                                    result = core::constants::weights[direction] *
                                        (
                                            density + 3 * 
                                            (velocity_x_component * x_velocity + velocity_y_component * y_velocity)
                                            + 9.0/2 *
                                            (velocity_x_component * x_velocity + velocity_y_component * y_velocity) *
                                            (velocity_x_component * x_velocity + velocity_y_component * y_velocity)
                                            - 3.0/2 * (x_velocity * x_velocity + y_velocity * y_velocity)
                                        );

                                    result = -relaxation_time_inverse * (value - result) + value;
                                    destination[
                                        A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes)
                                    ] = result;
                                }
                            }
                        }
                    };

// PERFORMANCE KERNELS ////////////////////////////////////////////////////////////////////////////////////////////////

                    /**
                     * @brief   This kernel performs the streaming step, the update of the macroscopic observables and 
                     *          collision step of a non-linear two-lattice iteration.
                     * 
                     * @tparam  A   any `core::access::AccessorConcept` from access.hpp 
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
                         * @brief   Constructor for a new `StreamCollideKernel` object. Create an instance of this 
                         *          kernel and pass it to `cgh.parallel_for(...)`.
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
                         * @brief   This overloaded operator is implicitly called to launch the kernel for various work 
                         *          items.
                         * 
                         * @param[in]   nd_item a work item from a two-dimensional SYCL nd-range
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            auto global_id_x = nd_item.get_global_id(1) + 1;
                            auto global_id_y = nd_item.get_global_id(0) + 1;
                            auto linear_index = 
                            core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            if(!phase_information[linear_index])
                            {
                                unsigned int iteration_node_offset =
                                    lbm::core::access::get_result_index(
                                        global_id_x, global_id_y, horizontal_nodes_domain
                                );

                                real_type distribution_values[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                                real_type result = 0;
                                real_type result_buf = 0;
                                real_type density = 0;
                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 
                                real_type flow_velocity_x = 0;
                                real_type flow_velocity_y = 0;
                                real_type absolute_velocity = 0;

                                // Loading distribution values
                                for (int direction = 0; direction < 9; ++direction)
                                {
                                    // Loading distribution values
                                    distribution_values[direction] = 
                                        source[
                                            A::at(
                                                lbm::core::access::get_neighbor(
                                                    linear_index, 8 - direction, horizontal_nodes_expanded), 
                                                direction, 
                                                horizontal_nodes_expanded * vertical_nodes_expanded
                                            )
                                        ];
                                }

                                // Macroscopic observables
                                for (int direction = 0; direction < 9; ++direction)
                                {   
                                    density += distribution_values[direction];
                                    velocity_x_component = direction % 3 - 1; 
                                    velocity_y_component = direction / 3 - 1; 
                                    flow_velocity_x += distribution_values[direction] * velocity_x_component;
                                    flow_velocity_y += distribution_values[direction] * velocity_y_component;
                                }
                                
                                absolute_velocity = 
                                flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y;

                                // Streaming and collision
                                for (int direction = 0; direction < 9; ++direction)
                                {
                                    velocity_x_component = (direction % 3) - 1; 
                                    velocity_y_component = (direction / 3) - 1; 

                                    result_buf = 
                                    velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y;

                                    result = core::constants::weights[direction] *
                                        (
                                            density + 3 * result_buf + 9.0/2 * result_buf * result_buf - 3.0/2 * 
                                            absolute_velocity
                                        );

                                    result = -relaxation_time_inverse * (distribution_values[direction] - result) 
                                        + distribution_values[direction];

                                    destination[
                                            A::at(
                                                linear_index, 
                                                direction, 
                                                horizontal_nodes_expanded * vertical_nodes_expanded
                                            )
                                        ] = result;
                                }

                                absolute_velocity = sycl::sqrt(absolute_velocity);

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
                     * @tparam  A   any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class EmplaceBounceBackKernel
                    {
                        private:

                        int8_t *phase_information;
                        real_type *destination;

                        unsigned int horizontal_nodes;
                        unsigned int total_nodes;

                        public:

                        EmplaceBounceBackKernel(const core::Simulation &simulation):
                        phase_information(simulation.data->phase_information),
                        destination(simulation.data->distribution_values_0),
                        horizontal_nodes(simulation.domain->horizontal_nodes),
                        total_nodes(simulation.domain->total_node_count)
                        {}

                        /**
                         * @brief   This overloaded operator is implicitly called to launch the kernel for various work 
                         *          items.
                         * 
                         * @param[in]   nd_item a work item from a two-dimensional SYCL nd-range
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const
                        {
                            auto global_id_x = nd_item.get_global_id(1) + 1;
                            auto global_id_y = nd_item.get_global_id(0) + 1;
                            auto linear_index = 
                                core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);

                            if(phase_information[linear_index] == 1)
                            {
                                for(int dir = 0; dir < 4; ++dir)
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

                                for(int dir = 5; dir < 9; ++dir)
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

                } // ! namespace non_linear

            } // ! namespace kernels

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! NON_LINEAR_TWO_LATTICE_KERNELS_HPP
