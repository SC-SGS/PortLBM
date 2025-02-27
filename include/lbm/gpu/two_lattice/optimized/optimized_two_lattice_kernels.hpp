/**
 * @file        optimized_two_lattice_kernels.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of kernels for the two-lattice algorithm
 *              with linear work item evaluation.
 * 
 * @version     1.0
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

// Separate debug kernels /////////////////////////////////////////////////////////////////////////////////////////////

                    /**
                     * @brief   This kernel performs the streaming step of a two-lattice iteration.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class StreamKernel 
                    {

                        private:

                        int8_t *phase_information;
                        double *source;
                        double *destination;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;

                        public:

                        /**
                         * @brief   Constructor for a new `StreamKernel` object.
                         *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
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
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const
                        {
                            auto global_id_x = nd_item.get_global_id(1);
                            auto global_id_y = nd_item.get_global_id(0);
                            auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);

                            if(!phase_information[linear_index])
                            {
                                for (const auto& direction : core::constants::all_directions)
                                {
                                    destination[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                                        source[
                                            A::at(
                                                lbm::core::access::get_neighbor(linear_index, 8 - direction, horizontal_nodes), 
                                                direction, 
                                                horizontal_nodes * vertical_nodes
                                            )
                                        ];
                                }
                            }
                        }
                    };

#ifdef WITH_NAN_PROTECTION 

                    /**
                     * @brief   This kernel performs the update of the macroscopic observables.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class MacroscopicObservablesKernel
                    {
                        private:

                        int8_t *phase_information;
                        double *destination;

                        double *densities;
                        double *x_velocities;
                        double *y_velocities;
                        double *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes_expanded;
                        unsigned int horizontal_nodes_domain;

                        public:

                        /**
                         * @brief   Constructor for a new `MacroscopicObservablesKernel` object.
                         *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
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
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            auto global_id_x = nd_item.get_global_id(1);
                            auto global_id_y = nd_item.get_global_id(0);
                            auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            if(!phase_information[linear_index])
                            {
                                unsigned int iteration_node_offset =
                                    lbm::core::access::decomposed::get_results_index(global_id_x, global_id_y, horizontal_nodes_domain);
                                double dist_vals[9];
                                double density = 0;
                                double absolute_velocity = 0;

                                for (const auto& direction : core::constants::all_directions)
                                {
                                    dist_vals[direction] = destination[A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes)];
                                    density += dist_vals[direction];
                                }

                                if(sycl::isnan(density) || density > std::numeric_limits<float>::max()) density = 0;
                                densities[iteration_node_offset] = density;
                                
                                sycl::vec<double,2> flow_velocity{0,0};

                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 
                                
                                for(const auto& i : core::constants::all_directions)
                                {
                                    velocity_x_component = i % 3 - 1; 
                                    velocity_y_component = i / 3 - 1; 
                                    flow_velocity[0] += dist_vals[i] * velocity_x_component;
                                    flow_velocity[1] += dist_vals[i] * velocity_y_component;
                                }

                                if(sycl::isnan(flow_velocity[0]) || flow_velocity[0] > std::numeric_limits<float>::max()) flow_velocity[0] = 0;
                                if(sycl::isnan(flow_velocity[1]) || flow_velocity[1] > std::numeric_limits<float>::max()) flow_velocity[1] = 0;

                                x_velocities[iteration_node_offset] = flow_velocity[0]; 
                                y_velocities[iteration_node_offset] = flow_velocity[1];

                                absolute_velocity = sycl::sqrt(flow_velocity[0] * flow_velocity[0] + flow_velocity[1] * flow_velocity[1]);

                                if(sycl::isnan(absolute_velocity) || absolute_velocity > std::numeric_limits<float>::max()) absolute_velocity = 0;
                                absolute_velocity_values[iteration_node_offset] = absolute_velocity;
                            }
                        }
                    };

#else // ! WITH_NAN_PROTECTION 

                    /**
                     * @brief   This kernel performs the update of the macroscopic observables.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class MacroscopicObservablesKernel
                    {
                        private:

                        int8_t *phase_information;
                        double *destination;

                        double *densities;
                        double *x_velocities;
                        double *y_velocities;
                        double *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes_expanded;
                        unsigned int horizontal_nodes_domain;

                        public:

                        /**
                         * @brief   Constructor for a new `MacroscopicObservablesKernel` object.
                         *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
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
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            auto global_id_x = nd_item.get_global_id(1);
                            auto global_id_y = nd_item.get_global_id(0);
                            auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            if(!phase_information[linear_index])
                            {
                                unsigned int iteration_node_offset =
                                    //lbm::core::access::results::get_result_index(linear_index, horizontal_nodes);
                                    lbm::core::access::decomposed::get_results_index(global_id_x, global_id_y, horizontal_nodes_domain);
                                double dist_vals[9];
                                double density = 0;

                                for (const auto& direction : core::constants::all_directions)
                                {
                                    dist_vals[direction] = destination[A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes)];
                                    density += dist_vals[direction];
                                }
                                densities[iteration_node_offset] = density;
                                
                                sycl::vec<double,2> flow_velocity{0,0};

                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 
                                
                                for(const auto& i : core::constants::all_directions)
                                {
                                    velocity_x_component = i % 3 - 1; 
                                    velocity_y_component = i / 3 - 1; 
                                    flow_velocity[0] += dist_vals[i] * velocity_x_component;
                                    flow_velocity[1] += dist_vals[i] * velocity_y_component;
                                }

                                x_velocities[iteration_node_offset] = flow_velocity[0]; //global_id_x;//global_id_x; //flow_velocity[0];
                                y_velocities[iteration_node_offset] = flow_velocity[1]; //global_id_y; //flow_velocity[1];

                                absolute_velocity_values[iteration_node_offset] = 
                                    sycl::sqrt(flow_velocity[0] * flow_velocity[0] + flow_velocity[1] * flow_velocity[1]);
                            }
                        }
                    };

#endif

                    /**
                     * @brief   This kernel performs the collision step of a two-lattice iteration.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A> 
                    class CollideKernel
                    {

                        private:

                        int8_t *phase_information;
                        double *destination;

                        double *densities;
                        double *x_velocities;
                        double *y_velocities;
                        double *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes_expanded;
                        unsigned int horizontal_nodes_domain;
                        double relaxation_time_inverse;

                        public:

                        /**
                         * @brief Constructor for a new `CollideKernel` object.
                         *        Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
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
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            auto global_id_x = nd_item.get_global_id(1);
                            auto global_id_y = nd_item.get_global_id(0);
                            auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            if(!phase_information[linear_index])
                            {
                                unsigned int iteration_node_offset =
                                    //lbm::core::access::results::get_result_index(linear_index, horizontal_nodes);
                                    lbm::core::access::decomposed::get_results_index(global_id_x, global_id_y, horizontal_nodes_domain);

                                double& x_velocity = x_velocities[iteration_node_offset];
                                double& y_velocity = y_velocities[iteration_node_offset];
                                double& density = densities[iteration_node_offset];

                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 

                                double value;
                                double result;

                                for (const auto& direction : core::constants::all_directions)
                                {
                                    value = destination[A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes)];

                                    velocity_x_component = (direction % 3) - 1; 
                                    velocity_y_component = (direction / 3) - 1; 

                                    result = core::constants::weights[direction] *
                                        (
                                            density + 3 * (velocity_x_component * x_velocity + velocity_y_component * y_velocity)
                                            + 9.0/2 *
                                            (velocity_x_component * x_velocity + velocity_y_component * y_velocity) *
                                            (velocity_x_component * x_velocity + velocity_y_component * y_velocity)
                                            - 3.0/2 * (x_velocity * x_velocity + y_velocity * y_velocity)
                                        );

                                    result = -relaxation_time_inverse * (value - result) + value;
                                    destination[A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes)] = result;
                                }
                            }
                        }
                    };

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
                        double *source;
                        // double *destination;

                        double *densities;
                        double *x_velocities;
                        double *y_velocities;
                        double *absolute_velocity_values;

                        unsigned int vertical_nodes_expanded;
                        unsigned int horizontal_nodes_expanded;
                        unsigned int horizontal_nodes_domain;
                        double relaxation_time_inverse;

                        // unsigned int local_buffer_length;
                        // sycl::local_accessor<double, 1> local;

                        public:

                        /**
                         * @brief Constructor for a new `StreamCollideKernel` object.
                         *        Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                         * 
                         * @param[in]   simulation  the structure containing all simulation data
                         */
                        StreamCollideKernel(const core::Simulation &simulation/*, sycl::handler &cgh, int work_group_size*/):
                        phase_information(simulation.data->phase_information),
                        source(simulation.data->distribution_values_0),
                        densities(simulation.results->densities_gpu),
                        x_velocities(simulation.results->x_velocities_gpu),
                        y_velocities(simulation.results->y_velocities_gpu),
                        absolute_velocity_values(simulation.results->absolute_velocities_gpu),
                        vertical_nodes_expanded(simulation.domain->vertical_nodes),
                        horizontal_nodes_expanded(simulation.domain->horizontal_nodes),
                        horizontal_nodes_domain(simulation.properties->horizontal_nodes),
                        relaxation_time_inverse(1 / simulation.properties->relaxation_time) //,
                        // local_buffer_length(9 * simulation.domain->subdomain_horizontal_nodes * simulation.domain->subdomain_vertical_nodes),
                        // local(sycl::local_accessor<double, 1>(sycl::range<1>(local_buffer_length), cgh))
                        {}

                        /**
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::nd_item<2> &nd_item) const 
                        {
                            // Determine global indices
                            auto global_id_x = nd_item.get_global_id(1);
                            auto global_id_y = nd_item.get_global_id(0);
                            auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes_expanded);

                            // Only do something for fluid nodes
                            if(!phase_information[linear_index])
                            {
                                // Get index of results vector
                                unsigned int iteration_node_offset =
                                    lbm::core::access::decomposed::get_results_index(global_id_x, global_id_y, horizontal_nodes_domain);

                                // This private array acts as a "buffer" for the distribution values, effectively replacing a second grid 
                                double distribution_values[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

                                // Macroscopic observable declarations
                                double result = 0;
                                double density = 0;
                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 
                                double flow_velocity_x = 0;
                                double flow_velocity_y = 0;

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
                                
                                densities[iteration_node_offset] = density;
                                x_velocities[iteration_node_offset] = flow_velocity_x;
                                y_velocities[iteration_node_offset] = flow_velocity_y;

                                absolute_velocity_values[iteration_node_offset] =
                                    sycl::sqrt(flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y);

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

                                    source[A::at(linear_index, direction, horizontal_nodes_expanded * vertical_nodes_expanded)] = result;
                                }
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
                        double *destination;

                        unsigned int horizontal_nodes;
                        unsigned int total_nodes;

                        public:

                        EmplaceBounceBackKernel(const core::Simulation &simulation):
                        phase_information(simulation.data->phase_information),
                        destination(simulation.data->distribution_values_0),
                        horizontal_nodes(simulation.domain->horizontal_nodes),
                        total_nodes(simulation.domain->total_node_count)
                        {}

                        void operator()(const sycl::nd_item<2> &nd_item) const
                        {
                            auto global_id_x = nd_item.get_global_id(1);
                            auto global_id_y = nd_item.get_global_id(0);
                            auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);

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

                    /**
                     * @brief Kernel for updating the inlets and outlets.
                     * 
                     * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                     */
                    template<core::access::AccessorConcept A>
                    class InoutUpdateKernel
                    {
                        private:

                        double *source;
                        double *y_velocities;
                        double outlet_density;
                        double outlet_density_inverse;
                        unsigned int horizontal_nodes;
                        unsigned int horizontal_nodes_original;
                        unsigned int vertical_nodes;

                        public:

                        InoutUpdateKernel(const core::Simulation &simulation):
                        source(simulation.data->distribution_values_0),
                        y_velocities(simulation.results->y_velocities_gpu),
                        outlet_density(simulation.properties->outlet_density),
                        outlet_density_inverse(1 / simulation.properties->outlet_density),
                        vertical_nodes(simulation.domain->vertical_nodes),
                        horizontal_nodes(simulation.domain->horizontal_nodes),
                        horizontal_nodes_original(simulation.properties->horizontal_nodes)
                        {}

                        void operator()(const sycl::item<1> &id) const
                        {
                            unsigned int missing[3];

                            double f_inverse[3];
                            double f_1 = 0;
                            double f_4 = 0;
                            double f_7 = 0;

                            for(int i = 0; i < 3; ++i) { missing[i] = 3 * i; }

                            // Setup actual coordinate, y coordinate: offset by two
                            int y = id + 2;

                            unsigned int current_border_node = 
                                core::access::get_node_index(horizontal_nodes_original - 2, y, horizontal_nodes);

                            f_1 = source[A::at(
                                core::access::get_neighbor(current_border_node, 7, horizontal_nodes), 
                                1, 
                                horizontal_nodes * vertical_nodes
                            )];

                            f_4 = source[A::at(
                                current_border_node, 
                                4, 
                                horizontal_nodes * vertical_nodes
                            )];

                            f_7 = source[A::at(
                                core::access::get_neighbor(current_border_node, 1, horizontal_nodes), 
                                7, 
                                horizontal_nodes * vertical_nodes
                            )];

                            for(int i = 0; i < 3; ++i)
                            {
                                f_inverse[i] = source[
                                    A::at(
                                        core::access::get_neighbor(current_border_node, missing[i], horizontal_nodes), 
                                        8 - missing[i], 
                                        horizontal_nodes * vertical_nodes
                                    )];
                            }

                            double x_velocity = 
                                - 1 + (1 / outlet_density) * (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));
                            
                            double y_velocity =
                                y_velocities[core::access::decomposed::get_results_index(horizontal_nodes_original - 3, y, horizontal_nodes_original)];

                            source[A::at(
                                core::access::get_neighbor(current_border_node, 8 - missing[0], horizontal_nodes), 
                                missing[0], 
                                horizontal_nodes * vertical_nodes
                            )] = f_inverse[0] - 0.5 * (f_1 - f_7) - 1.0/6 * outlet_density * x_velocity - 0.5 * outlet_density * y_velocity;

                            source[A::at(
                                core::access::get_neighbor(current_border_node, 8 - missing[1], horizontal_nodes), 
                                missing[1], 
                                horizontal_nodes * vertical_nodes
                            )] = f_inverse[1] - (2.0/3) * outlet_density * x_velocity;

                            source[A::at(
                                core::access::get_neighbor(current_border_node, 8 - missing[2], horizontal_nodes), 
                                missing[2], 
                                horizontal_nodes * vertical_nodes
                            )] = f_inverse[2] + 0.5 * (f_1 - f_7) - 1.0/6 * outlet_density * x_velocity + 0.5 * outlet_density * y_velocity;
                        }
                    };

                } // ! namespace optimized

            } // ! namespace kernels

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! OPTIMIZED_TWO_LATTICE_KERNELS_HPP
