/**
 * @file        swap_kernels.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of kernels for the swap algorithm.
 * 
 * @version     1.2
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_SWAP_KERNELS_HPP
#define LBM_SWAP_KERNELS_HPP

// Dependencies on other LBM core features
#include "../../core/access.hpp"
#include "../../core/constants.hpp"
#include "../../core/simulation.hpp"

// SYCL
#include <sycl/sycl.hpp>
#include <limits>

namespace lbm
{

    namespace gpu
    {

        namespace swap
        {

            /**
             * @brief This namespace contains all kernels for the swap algorithm.
             */
            namespace kernels
            {

// Performance kernels ////////////////////////////////////////////////////////////////////////////////////////////////

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

                    double *distribution_values;

                    unsigned int horizontal_nodes;
                    unsigned int total_nodes;

                    unsigned int subdomain_horizontal_nodes;
                    unsigned int subdomain_vertical_nodes;

                    public:

                    /**
                     * @brief   Constructor for a new `EmplaceBounceBackKernel` object.
                     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
                     */
                    EmplaceBounceBackKernel(const core::Simulation &simulation):
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    total_nodes(simulation.domain->total_node_count),
                    subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes),
                    subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes)
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param[in]   nd_item a work item from a two-dimensional SYCL nd-range
                     */
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
                                horizontal_nodes
                            );

                        if(phase_information[linear_index] == 1)
                        {
                            for(const auto& dir : core::constants::streaming_directions)
                            {
                                distribution_values[A::at(linear_index, dir, total_nodes)] = 
                                    distribution_values[
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
                 * @brief   This kernel performs the streaming step, the update of the macroscopic observables and 
                 *          collision step of a swap iteration.
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */          
                template<core::access::AccessorConcept A>
                class StreamCollideKernel
                {
                    private:

                    int8_t *phase_information;

                    double *distribution_values;

                    double *densities;
                    double *x_velocities;
                    double *y_velocities;
                    double *absolute_velocity_values;

                    unsigned int vertical_nodes_expanded;
                    unsigned int horizontal_nodes_expanded;
                    unsigned int horizontal_nodes_domain;
                    unsigned int total_nodes;

                    unsigned int subdomain_horizontal_nodes;
                    unsigned int subdomain_vertical_nodes;

                    double relaxation_time_inverse;

                    unsigned int local_buffer_length;
                    sycl::local_accessor<double, 1> local;

                    public:

                    /**
                     * @brief Constructor for a new `StreamCollideKernel` object.
                     *        Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation      the structure containing all simulation data
                     * @param[in]   cgh             the SYCL command group handler is used for setting up the local 
                     *                              memory
                     * @param[in]   work_group_size the work group size used by this kernel
                     */
                    StreamCollideKernel(const core::Simulation &simulation, sycl::handler &cgh, int work_group_size):
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    densities(simulation.results->densities_gpu),
                    x_velocities(simulation.results->x_velocities_gpu),
                    y_velocities(simulation.results->y_velocities_gpu),
                    absolute_velocity_values(simulation.results->absolute_velocities_gpu),
                    vertical_nodes_expanded(simulation.domain->vertical_nodes),
                    horizontal_nodes_expanded(simulation.domain->horizontal_nodes),
                    horizontal_nodes_domain(simulation.properties->horizontal_nodes),
                    total_nodes(simulation.domain->total_node_count),
                    relaxation_time_inverse(1 / simulation.properties->relaxation_time),
                    subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes),
                    subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes),
                    local_buffer_length((subdomain_horizontal_nodes + 4) * (subdomain_vertical_nodes + 1)),
                    local(sycl::local_accessor<double, 1>(sycl::range<1>(4 * local_buffer_length), cgh))
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param[in]   nd_item a work item from a two-dimensional SYCL nd-range
                     */
                    void operator()(const sycl::nd_item<2> &nd_item) const 
                    {
                        unsigned int global_id_x = 
                            nd_item.get_global_id(1) - nd_item.get_global_id(1) / nd_item.get_local_range(1);

                        unsigned int global_id_y = 
                            nd_item.get_global_id(0);

                        unsigned int linear_index = 
                            core::access::get_node_index(global_id_x, global_id_y,  horizontal_nodes_expanded);

                        unsigned int neighbor_index = 0;

                        unsigned int buffer_id_x = nd_item.get_local_id(1) + 1;
                        unsigned int buffer_id_y = nd_item.get_local_id(0);

                        unsigned int own_buffer_index = 
                            core::access::get_node_index(buffer_id_x, buffer_id_y, subdomain_horizontal_nodes + 4);

                        unsigned int buffer_target_index = 0;

                        double density = 0;
                        int velocity_x_component = 0; 
                        int velocity_y_component = 0; 
                        double flow_velocity_x = 0;
                        double flow_velocity_y = 0;
                        double absolute_velocity = 0;

                        double private_distribution_values [9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                        double result = 0;

                        // Send out active values and accept passive values of neighbor
                        for(auto& dir : {5, 6, 7, 8})
                        {
                            buffer_target_index = 
                                core::access::get_neighbor(own_buffer_index, dir, subdomain_horizontal_nodes + 4);

                            // Write values that are sent to neighbor into their shared buffer
                            local[4 * buffer_target_index + (8 - dir)] =
                                distribution_values[A::at(linear_index, dir, total_nodes)];

                            // Determine global index of neighbor
                            neighbor_index = 
                                core::access::get_neighbor(linear_index, dir, horizontal_nodes_expanded);

                            // Store incoming distribution values in private array
                            private_distribution_values[8 - dir] = 
                                distribution_values[A::at(neighbor_index, 8 - dir, total_nodes)];
                        } 

                        nd_item.barrier();

                        //Collect passive values from shared buffer
                        for(auto& dir : {0, 1, 2, 3})
                        {
                            private_distribution_values[8 - dir] = local[4 * own_buffer_index + dir];
                        }
                        
                        //Load stationary value into private buffer
                        private_distribution_values[4] = distribution_values[A::at(linear_index, 4, total_nodes)];

                        unsigned int iteration_node_offset =
                        lbm::core::access::decomposed::get_results_index(
                            (global_id_x) - (global_id_x / (subdomain_horizontal_nodes + 1)),
                            (global_id_y) - (global_id_y / (subdomain_vertical_nodes + 1)),
                            horizontal_nodes_domain
                        );

                        for (const auto& direction : core::constants::all_directions)
                        {
                            density += private_distribution_values[direction];
                            velocity_x_component = direction % 3 - 1; 
                            velocity_y_component = direction / 3 - 1; 
                            flow_velocity_x += private_distribution_values[direction] * velocity_x_component;
                            flow_velocity_y += private_distribution_values[direction] * velocity_y_component;
                        }
                        absolute_velocity = 
                            sycl::sqrt(flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y);

                        nd_item.barrier();

                        if(!phase_information[linear_index])
                        {       
                            // Perform collision and write back values to main memory
                            for (const auto& direction : core::constants::all_directions)
                            {
                                velocity_x_component = (direction % 3) - 1; 
                                velocity_y_component = (direction / 3) - 1; 

                                result = core::constants::weights[direction] *
                                    (
                                        density 
                                        + 3 * 
                                        (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)
                                        + 9.0/2 *
                                        (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y) *
                                        (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)
                                        - 3.0/2 * 
                                        (flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y)
                                    );

                                result =    -relaxation_time_inverse * (private_distribution_values[direction] - result) 
                                            + private_distribution_values[direction];

                                distribution_values[A::at(linear_index, direction, total_nodes)] = result;
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

            } // ! namespace kernels

        } // ! namespace swap

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! LBM_SWAP_KERNELS_HPP
