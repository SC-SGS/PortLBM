/**
 * @file        non_linear_swap_kernels.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of kernels for the swap algorithm
 *              with linear work item evaluation.
 * 
 * @version     1.1
 * 
 * @date        January 2025
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

// Separate debug kernels /////////////////////////////////////////////////////////////////////////////////////////////


// Performance kernels ////////////////////////////////////////////////////////////////////////////////////////////////

                /**
                 * @brief   Copies the values of all incoming 
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class HorizontalCopyToBufferKernel 
                {
                    private:

                    int8_t *phase_information;
                    double *distribution_values;

                    unsigned int vertical_nodes;
                    unsigned int horizontal_nodes;
                    unsigned int vertical_subdomain_size;

                    public:

                    /**
                     * @brief   Constructor for a new `HorizontalCopyToBufferKernel` object.
                     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
                     */
                    explicit HorizontalCopyToBufferKernel(const core::Simulation &simulation): 
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    vertical_nodes(simulation.domain->vertical_nodes),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    vertical_subdomain_size(simulation.domain->subdomain_vertical_nodes)
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                     */
                    void operator()(const sycl::item<2> &item) const
                    {
                        auto global_id_x = item.get_id(1);
                        auto global_id_y = (item.get_id(0) + 1) * (vertical_subdomain_size + 1);
                        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);
                        unsigned int neighbor_top = 
                            lbm::core::access::get_neighbor(linear_index, 7, horizontal_nodes);
                        unsigned int neighbor_bottom = 
                            lbm::core::access::get_neighbor(linear_index, 1, horizontal_nodes);

                        for (const auto& direction : {0, 1, 2})
                        {
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] = 
                            distribution_values[A::at(neighbor_top, direction, horizontal_nodes * vertical_nodes)];
                        }

                        for (const auto& direction : {6, 7, 8})
                        {
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                            distribution_values[A::at(neighbor_bottom, direction, horizontal_nodes * vertical_nodes)];
                        }
                    }
                };

                /**
                 * @brief   Copies the values of all incoming 
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class VerticalCopyToBufferKernel 
                {
                    private:

                    int8_t *phase_information;
                    double *distribution_values;

                    unsigned int vertical_nodes;
                    unsigned int horizontal_nodes;
                    unsigned int horizontal_subdomain_size;

                    public:

                    /**
                     * @brief   Constructor for a new `VerticalCopyToBufferKernel` object.
                     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
                     */
                    explicit VerticalCopyToBufferKernel(const core::Simulation &simulation): 
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    vertical_nodes(simulation.domain->vertical_nodes),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    horizontal_subdomain_size(simulation.domain->subdomain_horizontal_nodes)
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                     */
                    void operator()(const sycl::item<2> &item) const
                    {
                        auto global_id_x = (item.get_id(1) + 1) * (horizontal_subdomain_size + 1);
                        auto global_id_y = item.get_id(0);
                        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);
                        unsigned int neighbor_left = 
                            lbm::core::access::get_neighbor(linear_index, 3, horizontal_nodes);
                        unsigned int neighbor_right = 
                            lbm::core::access::get_neighbor(linear_index, 5, horizontal_nodes);

                        for (const auto& direction : {2, 5, 8})
                        {
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                            distribution_values[A::at(neighbor_left, direction, horizontal_nodes * vertical_nodes)];
                        }

                        for (const auto& direction : {0, 3, 6})
                        {
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                            distribution_values[A::at(neighbor_right, direction, horizontal_nodes * vertical_nodes)];
                        }
                    }
                };

                /**
                 * @brief   Copies the values of all incoming 
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class HorizontalCopyFromBufferKernel 
                {
                    private:

                    int8_t *phase_information;
                    double *distribution_values;

                    unsigned int vertical_nodes;
                    unsigned int horizontal_nodes;
                    unsigned int vertical_subdomain_size;

                    public:

                    /**
                     * @brief   Constructor for a new `HorizontalCopyToBufferKernel` object.
                     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
                     */
                    explicit HorizontalCopyFromBufferKernel(const core::Simulation &simulation): 
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    vertical_nodes(simulation.domain->vertical_nodes),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    vertical_subdomain_size(simulation.domain->subdomain_vertical_nodes)
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                     */
                    void operator()(const sycl::item<2> &item) const
                    {
                        auto global_id_x = item.get_id(1);
                        auto global_id_y = (item.get_id(0) + 1) * (vertical_subdomain_size + 1);
                        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);
                        unsigned int neighbor_top = 
                            lbm::core::access::get_neighbor(linear_index, 7, horizontal_nodes);
                        unsigned int neighbor_bottom = 
                            lbm::core::access::get_neighbor(linear_index, 1, horizontal_nodes);

                        for (const auto& direction : {6, 7, 8})
                        {
                            distribution_values[A::at(neighbor_top, direction, horizontal_nodes * vertical_nodes)] = 
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)];
                        }

                        for (const auto& direction : {0, 1, 2})
                        {
                            distribution_values[A::at(neighbor_bottom, direction, horizontal_nodes * vertical_nodes)] =
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)];
                        }
                    }
                };

                /**
                 * @brief   Copies the values of all incoming 
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class VerticalCopyFromBufferKernel 
                {
                    private:

                    int8_t *phase_information;
                    double *distribution_values;

                    unsigned int vertical_nodes;
                    unsigned int horizontal_nodes;
                    unsigned int horizontal_subdomain_size;

                    public:

                    /**
                     * @brief   Constructor for a new `VerticalCopyToBufferKernel` object.
                     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
                     */
                    explicit VerticalCopyFromBufferKernel(const core::Simulation &simulation): 
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    vertical_nodes(simulation.domain->vertical_nodes),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    horizontal_subdomain_size(simulation.domain->subdomain_horizontal_nodes)
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                     */
                    void operator()(const sycl::item<2> &item) const
                    {
                        auto global_id_x = (item.get_id(1) + 1) * (horizontal_subdomain_size + 1) - 1;
                        auto global_id_y = item.get_id(0);
                        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);
                        unsigned int neighbor_left = 
                            lbm::core::access::get_neighbor(linear_index, 3, horizontal_nodes);
                        unsigned int neighbor_right = 
                            lbm::core::access::get_neighbor(linear_index, 5, horizontal_nodes);

                        for (const auto& direction : {0, 3, 6})
                        {
                            distribution_values[A::at(neighbor_left, direction, horizontal_nodes * vertical_nodes)] = 
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)];
                        }

                        for (const auto& direction : {2, 5, 8})
                        {
                            distribution_values[A::at(neighbor_right, direction, horizontal_nodes * vertical_nodes)] =
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)];
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
                    double *distribution_values;

                    unsigned int horizontal_nodes;
                    unsigned int total_nodes;

                    unsigned int subdomain_horizontal_nodes;
                    unsigned int subdomain_vertical_nodes;

                    public:

                    EmplaceBounceBackKernel(const core::Simulation &simulation):
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
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
                                global_id_x, global_id_y, subdomain_vertical_nodes, subdomain_horizontal_nodes, horizontal_nodes
                            );

                        if(phase_information[linear_index] == 1)
                        {
                            for(const auto& dir : {0, 1, 2, 3, 5, 6, 7, 8})
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
                 * @brief Kernel for updating the inlets.
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class InletUpdateKernel
                {
                    private:

                    double *distribution_values;
                    double inlet_density;
                    double inlet_velocity_x;
                    double inlet_velocity_y;
                    unsigned int horizontal_nodes;
                    unsigned int total_nodes;
                    double distribution [9];
                    unsigned int subdomain_vertical_nodes;
                    unsigned int subdomain_horizontal_nodes;

                    public:

                    InletUpdateKernel(const core::Simulation &simulation, double distribution [9]):
                    distribution_values(simulation.data->distribution_values_0),
                    inlet_density(simulation.properties->inlet_density),
                    inlet_velocity_x(simulation.properties->inlet_velocity_x),
                    inlet_velocity_y(simulation.properties->inlet_velocity_y),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    total_nodes(simulation.domain->total_node_count),
                    subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes),
                    subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes)
                    {
                        memcpy(this->distribution, distribution, 9*sizeof(double));
                    }

                    void operator()(const sycl::item<1> &id) const
                    {
                        unsigned int current_node = 
                            core::access::decomposed::BufferedNodeAccess::get_index(
                                0, 
                                id.get_id(0) + 1, 
                                subdomain_vertical_nodes, 
                                subdomain_horizontal_nodes, 
                                horizontal_nodes
                            );

                        for(int i = 0; i < 9; ++i)
                        {
                            distribution_values[A::at(current_node, i, total_nodes)] = distribution[i];
                        }

                        distribution_values[A::at(current_node, 5, total_nodes)] = id.get_id(0) + 1;
                    }
                };  

                /**
                 * @brief Kernel for updating the inlets and outlets.
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class OutletUpdateKernel
                {
                    private:

                    double *distribution_values;
                    double *y_velocities;
                    double outlet_density;
                    double outlet_density_inverse;
                    unsigned int horizontal_nodes;
                    unsigned int horizontal_nodes_original;
                    unsigned int vertical_nodes;
                    unsigned int subdomain_vertical_nodes;
                    unsigned int subdomain_horizontal_nodes;

                    public:

                    OutletUpdateKernel(const core::Simulation &simulation):
                    distribution_values(simulation.data->distribution_values_0),
                    y_velocities(simulation.results->y_velocities_gpu),
                    outlet_density(simulation.properties->outlet_density),
                    outlet_density_inverse(1 / simulation.properties->outlet_density),
                    vertical_nodes(simulation.domain->vertical_nodes),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    horizontal_nodes_original(simulation.properties->horizontal_nodes),
                    subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes),
                    subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes)
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
                                core::access::decomposed::BufferedNodeAccess::get_index(
                                    horizontal_nodes_original - 2, 
                                    y, 
                                    subdomain_vertical_nodes, 
                                    subdomain_horizontal_nodes, 
                                    horizontal_nodes
                                );

                        f_1 = distribution_values[A::at(
                            core::access::get_neighbor(current_border_node, 7, horizontal_nodes), 
                            1, 
                            horizontal_nodes * vertical_nodes
                        )];

                        f_4 = distribution_values[A::at(
                            current_border_node, 
                            4, 
                            horizontal_nodes * vertical_nodes
                        )];

                        f_7 = distribution_values[A::at(
                            core::access::get_neighbor(current_border_node, 1, horizontal_nodes), 
                            7, 
                            horizontal_nodes * vertical_nodes
                        )];

                        for(int i = 0; i < 3; ++i)
                        {
                            f_inverse[i] = distribution_values[
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

                        distribution_values[A::at(
                            core::access::get_neighbor(current_border_node, 8 - 0, horizontal_nodes), 
                            0, 
                            horizontal_nodes * vertical_nodes
                        )] = f_inverse[0] - 0.5 * (f_1 - f_7) - 1.0/6 * outlet_density * x_velocity - 0.5 * outlet_density * y_velocity;

                        distribution_values[A::at(
                            core::access::get_neighbor(current_border_node, 8 - 3, horizontal_nodes), 
                            3, 
                            horizontal_nodes * vertical_nodes
                        )] = f_inverse[1] - (2.0/3) * outlet_density * x_velocity;

                        distribution_values[A::at(
                            core::access::get_neighbor(current_border_node, 8 - 6, horizontal_nodes),  // current_border_node,// 
                            6, // 2,// 6, 
                            horizontal_nodes * vertical_nodes
                        )] = f_inverse[2] + 0.5 * (f_1 - f_7) - 1.0/6 * outlet_density * x_velocity + 0.5 * outlet_density * y_velocity;
                    }
                };                

                /**
                 * @brief   Copies the values of all incoming 
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class HorizontalBufferStreamKernel 
                {
                    private:

                    int8_t *phase_information;
                    double *distribution_values;

                    unsigned int vertical_nodes;
                    unsigned int horizontal_nodes;
                    unsigned int vertical_subdomain_size;

                    public:

                    /**
                     * @brief   Constructor for a new `HorizontalCopyToBufferKernel` object.
                     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
                     */
                    explicit HorizontalBufferStreamKernel(const core::Simulation &simulation): 
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    vertical_nodes(simulation.domain->vertical_nodes),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    vertical_subdomain_size(simulation.domain->subdomain_vertical_nodes)
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                     */
                    void operator()(const sycl::item<2> &item) const
                    {
                        auto global_id_x = item.get_id(1);
                        auto global_id_y = (item.get_id(0) + 1) * (vertical_subdomain_size + 1) - 1;
                        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);

                        for (const auto& direction : {6, 7, 8})
                        {
                            distribution_values[
                                A::at(
                                    lbm::core::access::get_neighbor(linear_index, direction, horizontal_nodes), 
                                    8 - direction, 
                                    horizontal_nodes * vertical_nodes
                                )
                            ] =
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)];
                        }
                    }
                };

                /**
                 * @brief   Copies the values of all incoming 
                 * 
                 * @tparam  A any `core::access::AccessorConcept` from access.hpp 
                 */
                template<core::access::AccessorConcept A>
                class VerticalBufferStreamKernel
                {
                    private:

                    int8_t *phase_information;
                    double *distribution_values;

                    unsigned int vertical_nodes;
                    unsigned int horizontal_nodes;
                    unsigned int horizontal_subdomain_size;

                    public:

                    /**
                     * @brief   Constructor for a new `VerticalCopyToBufferKernel` object.
                     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
                     */
                    explicit VerticalBufferStreamKernel(const core::Simulation &simulation): 
                    phase_information(simulation.data->phase_information),
                    distribution_values(simulation.data->distribution_values_0),
                    vertical_nodes(simulation.domain->vertical_nodes),
                    horizontal_nodes(simulation.domain->horizontal_nodes),
                    horizontal_subdomain_size(simulation.domain->subdomain_horizontal_nodes)
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                     */
                    void operator()(const sycl::item<2> &item) const
                    {
                        auto global_id_x = (item.get_id(1) + 1) * (horizontal_subdomain_size + 1) - 1;
                        auto global_id_y = item.get_id(0);
                        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);

                        for (const auto& direction : {5, 6, 8})
                        {
                            distribution_values[
                                A::at(
                                    lbm::core::access::get_neighbor(linear_index, direction, horizontal_nodes), 
                                    8 - direction, 
                                    horizontal_nodes * vertical_nodes
                                )
                            ] =
                            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)];
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
                    double *destination;

                    double *densities;
                    double *x_velocities;
                    double *y_velocities;
                    double *absolute_velocity_values;

                    unsigned int vertical_nodes_expanded;
                    unsigned int horizontal_nodes_expanded;
                    unsigned int horizontal_nodes_domain;
                    unsigned int subdomain_horizontal_nodes;
                    unsigned int subdomain_vertical_nodes;
                    unsigned int total_nodes;
                    unsigned int wg_horizontal_with_buffers;
                    double relaxation_time_inverse;
                    unsigned int local_buffer_length;
                    sycl::local_accessor<double, 1> local;

                    public:

                    /**
                     * @brief Constructor for a new `StreamCollideKernel` object.
                     *        Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                     * 
                     * @param[in]   simulation  the structure containing all simulation data
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
                    wg_horizontal_with_buffers(horizontal_nodes_expanded + 1),
                    subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes),
                    subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes),
                    local_buffer_length((subdomain_horizontal_nodes + 4) * (subdomain_vertical_nodes + 1)),
                    local(sycl::local_accessor<double, 1>(sycl::range<1>(4 * local_buffer_length), cgh))
                    {}

                    /**
                     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                     * 
                     * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                     */
                    void operator()(const sycl::nd_item<2> &nd_item) const 
                    {
                        unsigned int global_id_x = nd_item.get_global_id(1) - nd_item.get_global_id(1) / nd_item.get_local_range(1);
                        unsigned int global_id_y = nd_item.get_global_id(0);
                        unsigned int linear_index = core::access::get_node_index(global_id_x, global_id_y,  horizontal_nodes_expanded);
                        unsigned int neighbor_index = 0;

                        unsigned int buffer_id_x = nd_item.get_local_id(1) + 1;
                        unsigned int buffer_id_y = nd_item.get_local_id(0) + 1;
                        unsigned int own_buffer_index = core::access::get_node_index(buffer_id_x, buffer_id_y,  subdomain_horizontal_nodes + 4);
                        unsigned int buffer_target_index = 0;

                        double density = 0;
                        int velocity_x_component = 0; 
                        int velocity_y_component = 0; 
                        double flow_velocity_x = 0;
                        double flow_velocity_y = 0;
                        double absolute_velocity = 0;

                        double private_distribution_values [9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                        double result = 0;

                        // // Load all values into shared ("local") memory
                        // for(auto& dir : core::constants::all_directions)
                        // {
                        //     local[A::at(own_buffer_index, dir, local_buffer_length)] = 
                        //         distribution_values[A::at(linear_index, direction, total_nodes)];
                        // }
                        // nd_item.barrier();

                        // // All values are present in shared memory

                        // Send out active values
                        for(auto& dir : {5, 6, 7, 8})
                        {
                            buffer_target_index = core::access::get_neighbor(own_buffer_index, dir, subdomain_horizontal_nodes + 4);

                            // Write values that are sent to neighbor into their shared buffer
                            local[4 * buffer_target_index + (8 - dir)] =
                                distribution_values[A::at(linear_index, dir, total_nodes)];
                        }  

                        // Accept passive values of neighbor
                        for(auto& dir : {0, 1, 2, 3})
                        {
                            // Determine global index of neighbor
                            neighbor_index = core::access::get_neighbor(linear_index, 8 - dir, horizontal_nodes_expanded);

                            // Store incoming distribution values in private array
                            private_distribution_values[dir] = distribution_values[A::at(neighbor_index, dir, total_nodes)];
                        }  

                        //Collect passive values from shared buffer
                        for(auto& dir : {0, 1, 2, 3})
                        {
                            private_distribution_values[8 - dir] = //local[A::at(own_buffer_index, dir, local_buffer_length)];
                            local[4 * own_buffer_index + dir];
                        }
                        // private_distribution_values[6] = buffer_id_x;
                        // private_distribution_values[7] = buffer_id_y;
                        // private_distribution_values[8] = own_buffer_index;
                        // private_distribution_values[5] = core::access::get_neighbor(own_buffer_index, 5, subdomain_horizontal_nodes + 4);
                        
                        //Load stationary value into private buffer
                        private_distribution_values[4] = distribution_values[A::at(linear_index, 4, total_nodes)];

                        nd_item.barrier();

                        // All values are present in private memory!

                        unsigned int iteration_node_offset =
                        lbm::core::access::decomposed::get_results_index(
                            (global_id_x) - (global_id_x / (subdomain_horizontal_nodes + 1)), //nd_item.get_local_range(1)),
                            (global_id_y) - (global_id_y / (subdomain_vertical_nodes + 1)), //nd_item.get_local_range(0)),
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
                        absolute_velocity = sycl::sqrt(flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y);

                        if(!phase_information[linear_index])
                        {
                            // Update macroscopic observables
                            densities[iteration_node_offset] = iteration_node_offset;  // density;
                            x_velocities[iteration_node_offset] = (global_id_x) - (global_id_x / (subdomain_horizontal_nodes + 1)); // nd_item.get_local_range(1)); //  flow_velocity_x; 
                            y_velocities[iteration_node_offset] = (global_id_y) - (global_id_y / (subdomain_vertical_nodes + 1)); //nd_item.get_local_range(0)); //flow_velocity_y; 
                            absolute_velocity_values[iteration_node_offset] = absolute_velocity;
                                
                            // Perform collision and write back values to main memory
                            for (const auto& direction : core::constants::all_directions)
                            {
                                // velocity_x_component = (direction % 3) - 1; 
                                // velocity_y_component = (direction / 3) - 1; 

                                // result = core::constants::weights[direction] *
                                //     (
                                //         density + 3 * (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)
                                //         + 9.0/2 *
                                //         (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)*
                                //         (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)
                                //         - 3.0/2 * (flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y)
                                //     );

                                // result =    -relaxation_time_inverse * (private_distribution_values[direction] - result) 
                                //             + private_distribution_values[direction];

                                distribution_values[A::at(linear_index, direction, total_nodes)] = private_distribution_values[direction];// result;
                            }
                        }
                    }
                };


            } // ! namespace kernels

        } // ! namespace swap

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! LINEAR_swap_KERNELS_HPP















































                        // auto global_id_x = nd_item.get_global_id(1);
                        // auto global_id_y = nd_item.get_global_id(0);
                        // auto linear_index = core::access::get_node_index(global_id_x, global_id_y,  horizontal_nodes_expanded);

                        // double density = 0;
                        // int velocity_x_component = 0; 
                        // int velocity_y_component = 0; 
                        // double flow_velocity_x = 0;
                        // double flow_velocity_y = 0;

                        // int target_outside_subdomain = 0;

                        // int source_node_x = 0;
                        // int source_node_y = 0;
                        // int source_node_linear_index = 0;

                        // int local_node_x = 0;
                        // int local_node_y = 0;
                        // int buffer_target_index = 0;
                        // int swap_neighbor_index = 0;














                        //     /* Load passive values into local (shared) memory, perform swap and update observables */

                        //     // dir 5 //////////////////////////////////////////////////////////////////////////////////
                        //     target_outside_subdomain = (nd_item.get_local_id(1) + 1) / nd_item.get_local_range(1);

                        //     // Determine location of source node within global lattice
                        //     source_node_x = nd_item.get_global_id(1) - target_outside_subdomain * nd_item.get_local_range(1);
                        //     source_node_y = nd_item.get_global_id(0);
                        //     source_node_linear_index = core::access::decomposed::BufferedNodeAccess::get_index(
                        //         source_node_x, 
                        //         source_node_y, 
                        //         nd_item.get_local_range(0), 
                        //         nd_item.get_local_range(1), 
                        //         horizontal_nodes_expanded
                        //     );
                            
                        //     // Determine index of swap partner's passive buffer within local lattice
                        //     local_node_x = 
                        //         nd_item.get_local_id(1) + 1 - target_outside_subdomain * nd_item.get_local_range(1);
                        //     local_node_y = 
                        //         nd_item.get_local_id(0);
                        //     buffer_target_index = 
                        //         core::access::get_node_index(local_node_x, local_node_y, nd_item.get_local_range(1));

                        //     // Determine global index of swap partner to store incoming values within lattice
                        //     local_node_x = 
                        //         nd_item.get_global_id(1) + 1 - target_outside_subdomain * nd_item.get_local_range(1);
                        //     local_node_y = 
                        //         nd_item.get_global_id(0);
                        //     swap_neighbor_index = 
                        //         core::access::decomposed::BufferedNodeAccess::get_index(
                        //             local_node_x, 
                        //             local_node_y, 
                        //             nd_item.get_local_range(0), 
                        //             nd_item.get_local_range(1), 
                        //             horizontal_nodes_expanded
                        //         );

                        //     // Update target's passive buffer
                        //     local[4 * buffer_target_index + 3] = 
                        //         distribution_values[A::at(source_node_linear_index, 5, total_nodes)];

                        //     // Update this node's private memory for incoming distribution values
                        //     private_distribution_values[3] = 
                        //         distribution_values[A::at(swap_neighbor_index, 3, total_nodes)];

                        //     // dir 6 //////////////////////////////////////////////////////////////////////////////////

                        //     // Determine location of source node within global lattice

                        //     /*  source_node_x =         global base x                                               A
                        //      *                      +   subgroup_horizontal_nodes (if local_id_x = 0)               B
                        //      *                      -   local_id_x (if local_id_y = subgroup_vertical_nodes - 1)    C 
                        //      */
                        //     source_node_x = 
                        //         // A
                        //         nd_item.get_global_id(1)
                        //         // B
                        //         + ((nd_item.get_local_range(1) - nd_item.get_local_id(1)) / 
                        //             nd_item.get_local_range(1)) * nd_item.get_local_range(1)
                        //         // C
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * (nd_item.get_local_id(1));


                        //     /*  source_node_y =         global base y                                   A
                        //      *                      -   subgroup_vertical_nodes                         B
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1)
                        //      */
                        //     source_node_y = 
                        //         // A
                        //         nd_item.get_global_id(0)
                        //         // B
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * nd_item.get_local_range(0);
                            
                        //     source_node_linear_index = core::access::decomposed::BufferedNodeAccess::get_index(
                        //         source_node_x,
                        //         source_node_y, 
                        //         nd_item.get_local_range(0), 
                        //         nd_item.get_local_range(1), 
                        //         horizontal_nodes_expanded
                        //     );

                        //     // Determine index of swap partner's passive buffer within local lattice

                        //     /*  source_node_x =         local base x                                                A
                        //      *                      -   1 (base offset)                                             B 
                        //      *                      +   subgroup_horizontal_nodes (if local_id_x = 0)               C
                        //      *                      -   local_id_x (if local_id_y = subgroup_vertical_nodes - 1)    D 
                        //      */
                        //     local_node_x = 
                        //         // A
                        //         nd_item.get_local_id(1) 
                        //         // B
                        //         - 1
                        //         // C 
                        //         + ((nd_item.get_local_range(1) - nd_item.get_local_id(1)) / 
                        //         nd_item.get_local_range(1)) * nd_item.get_local_range(1)
                        //         // D
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * (nd_item.get_local_id(1));

                        //     /*  source_node_y =         local base y                                    A
                        //      *                      +   1 (base offset)                                 B
                        //      *                      -   subgroup_vertical_nodes                         C
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1)
                        //      */
                        //     local_node_y = 
                        //         // A
                        //         nd_item.get_local_id(0) 
                        //         // B
                        //         + 1 
                        //         // C
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * nd_item.get_local_range(0);
                              
                        //     buffer_target_index = 
                        //         core::access::get_node_index(local_node_x, local_node_y, nd_item.get_local_range(1));

                        //     // Determine global index of swap partner to store incoming values within lattice

                        //     /*  source_node_x =         global base x                                               A
                        //      *                      -   1 (base offset)                                             B
                        //      *                      +   subgroup_horizontal_nodes (if local_id_x = 0)               C
                        //      *                      -   local_id_x (if local_id_y = subgroup_vertical_nodes - 1)    D 
                        //      */
                        //     source_node_x = 
                        //         // A
                        //         nd_item.get_global_id(1)
                        //         // B
                        //         - 1
                        //         // C
                        //         + ((nd_item.get_local_range(1) - nd_item.get_local_id(1)) / 
                        //             nd_item.get_local_range(1)) * nd_item.get_local_range(1)
                        //         // D
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * (nd_item.get_local_id(1));


                        //     /*  source_node_y =         global base y                                   A
                        //      *                      +   1 (base offset)                                 B
                        //      *                      -   subgroup_vertical_nodes                         C
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1)
                        //      */
                        //     source_node_y = 
                        //         // A
                        //         nd_item.get_global_id(0)
                        //         // B
                        //         + 1
                        //         // C
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * nd_item.get_local_range(1);
                            
                        //     // Update target's passive buffer
                        //     local[4 * buffer_target_index + 2] = 
                        //         distribution_values[A::at(source_node_linear_index, 6, total_nodes)];
                            
                        //     // Update this node's private memory for incoming distribution values
                        //     private_distribution_values[2] = 
                        //         distribution_values[A::at(swap_neighbor_index, 2, total_nodes)];

                        //     // dir 7 //////////////////////////////////////////////////////////////////////////////////
                        //     target_outside_subdomain = (nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0);
                            
                        //     // Determine location of source node within global lattice
                        //     source_node_x = nd_item.get_global_id(1);
                        //     source_node_y = nd_item.get_global_id(0) - target_outside_subdomain * nd_item.get_local_range(0);
                        //     source_node_linear_index = core::access::decomposed::BufferedNodeAccess::get_index(
                        //         source_node_x, 
                        //         source_node_y, 
                        //         nd_item.get_local_range(0), 
                        //         nd_item.get_local_range(1), 
                        //         horizontal_nodes_expanded
                        //     );

                        //     // Determine index of swap partner's passive buffer within local lattice
                        //     local_node_x = 
                        //         nd_item.get_local_id(1);
                        //     local_node_y = 
                        //         nd_item.get_local_id(0) + 1 - target_outside_subdomain * nd_item.get_local_range(0);
                        //     buffer_target_index = 
                        //         core::access::get_node_index(local_node_x, local_node_y, nd_item.get_local_range(1));

                        //     // Determine global index of swap partner to store incoming values within lattice
                        //     local_node_x = 
                        //         nd_item.get_global_id(1); 
                        //     local_node_y = 
                        //         nd_item.get_global_id(0) + 1 - target_outside_subdomain * nd_item.get_local_range(0);;
                        //     swap_neighbor_index = 
                        //         core::access::decomposed::BufferedNodeAccess::get_index(
                        //             local_node_x, 
                        //             local_node_y, 
                        //             nd_item.get_local_range(0), 
                        //             nd_item.get_local_range(1), 
                        //             horizontal_nodes_expanded
                        //         );

                        //     // Update target's passive buffer
                        //     local[4 * buffer_target_index + 1] =
                        //         distribution_values[A::at(source_node_linear_index, 7, total_nodes)];

                        //     // Update this node's private memory for incoming distribution values
                        //     private_distribution_values[1] = 
                        //         distribution_values[A::at(swap_neighbor_index, 1, total_nodes)];

                        //     // dir 8 //////////////////////////////////////////////////////////////////////////////////

                        //     // Determine location of source node within global lattice

                        //     /*  source_node_x =         global base x                                               A
                        //      *                      -   subgroup_horizontal_nodes                                   B
                        //      *                          (if local_id_x = subgroup_horizontal_nodes - 1)              
                        //      *                      +   (subgroup_horizontal_nodes - 1 - local_id_x)                C
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1) 
                        //      */
                        //     source_node_x = 
                        //         // A
                        //         nd_item.get_global_id(1)
                        //         // B
                        //         - ((nd_item.get_local_id(1) + 1) / nd_item.get_local_range(1)) 
                        //             * nd_item.get_local_range(1)
                        //         // C
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * (nd_item.get_local_range(1) - 1 - nd_item.get_local_id(1));

                        //     /*  source_node_y =         global base y                                   A
                        //      *                      -   subgroup_vertical_nodes                         B
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1)
                        //      */
                        //     source_node_y = 
                        //         // A
                        //         nd_item.get_global_id(0)
                        //         // B
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * nd_item.get_local_range(0);
                            
                        //     source_node_linear_index = core::access::decomposed::BufferedNodeAccess::get_index(
                        //         source_node_x,
                        //         source_node_y, 
                        //         nd_item.get_local_range(0), 
                        //         nd_item.get_local_range(1), 
                        //         horizontal_nodes_expanded
                        //     );

                        //     // Determine index of swap partner's passive buffer within local lattice

                        //     /*  source_node_x =         local base x                                                A
                        //      *                      +   1 (base offset)                                             B
                        //      *                      -   subgroup_horizontal_nodes                                   C
                        //      *                          (if local_id_x = subgroup_horizontal_nodes - 1)              
                        //      *                      +   (subgroup_horizontal_nodes - 1 - local_id_x)                D
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1) 
                        //      */
                        //     local_node_x = 
                        //         // A
                        //         nd_item.get_local_id(1)
                        //         // B
                        //         + 1
                        //         // C
                        //         - ((nd_item.get_local_id(1) + 1) / nd_item.get_local_range(1)) 
                        //             * nd_item.get_local_range(1)
                        //         // D
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * (nd_item.get_local_range(1) - 1 - nd_item.get_local_id(1));

                        //     /*  source_node_y =         local base y                                    A
                        //      *                      +   1 (base offset)                                 B
                        //      *                      -   subgroup_vertical_nodes                         C
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1)
                        //      */
                        //     local_node_y = 
                        //         // A
                        //         nd_item.get_local_id(0)
                        //         // B
                        //         + 1
                        //         // C
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * nd_item.get_local_range(0);
                              
                        //     buffer_target_index = 
                        //         core::access::get_node_index(local_node_x, local_node_y, nd_item.get_local_range(1));

                        //     // Determine global index of swap partner to store incoming values within lattice

                        //     /*  source_node_x =         global base x                                               A
                        //      *                      +   1 (base offset)                                             B
                        //      *                      -   subgroup_horizontal_nodes                                   C
                        //      *                          (if local_id_x = subgroup_horizontal_nodes - 1)              
                        //      *                      +   (subgroup_horizontal_nodes - 1 - local_id_x)                D
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1) 
                        //      */
                        //     local_node_x = 
                        //         // A
                        //         nd_item.get_global_id(1)
                        //         // B
                        //         + 1
                        //         // C
                        //         - ((nd_item.get_local_id(1) + 1) / nd_item.get_local_range(1)) 
                        //             * nd_item.get_local_range(1)
                        //         // D
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * (nd_item.get_local_range(1) - 1 - nd_item.get_local_id(1));

                        //     /*  source_node_y =         global base y                                   A
                        //      *                      +   1 (base offset)                                 B
                        //      *                      -   subgroup_vertical_nodes                         C
                        //      *                          (if local_id_y = subgroup_vertical_nodes - 1)
                        //      */
                        //     local_node_y = 
                        //         // A
                        //         nd_item.get_global_id(0)
                        //         // B
                        //         + 1
                        //         // C
                        //         - ((nd_item.get_local_id(0) + 1) / nd_item.get_local_range(0)) 
                        //             * nd_item.get_local_range(0);
                            
                        //     // Update target's passive buffer
                        //     local[4 * buffer_target_index] = 
                        //         distribution_values[A::at(source_node_linear_index, 8, total_nodes)];
                            
                        //     // Update this node's private memory for incoming distribution values
                        //     private_distribution_values[0] = 
                        //         distribution_values[A::at(swap_neighbor_index, 0, total_nodes)];

                        //     /* All passive values per subdomain node are present */

                        //     /* Fill up remaining values in private distribution values and resort */

                        //     // Load stationary population into private distribution values
                        //     private_distribution_values[4] = 
                        //         distribution_values[A::at(swap_neighbor_index, 4, total_nodes)];
                            
                        //     // Get buffer index of this work item
                        //     buffer_target_index = 
                        //         core::access::get_node_index(
                        //             nd_item.get_local_id(1), 
                        //             nd_item.get_local_id(0), 
                        //             nd_item.get_local_range(1)
                        //     );

                        //     // Copy to private memory
                        //     for (const auto& direction : {0, 1, 2, 3})
                        //     {
                        //         private_distribution_values[8 - direction] = local[4 * buffer_target_index + direction];
                        //     }

                        //     /* All distribution values of a node are present in private memory at the right spot */