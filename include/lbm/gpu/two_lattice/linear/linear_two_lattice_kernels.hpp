/**
 * @file        linear_two_lattice_kernels.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of kernels for the two-lattice algorithm
 *              with linear work item evaluation.
 * 
 * @version     1.0
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LINEAR_TWO_LATTICE_KERNELS_HPP
#define LINEAR_TWO_LATTICE_KERNELS_HPP

#include "../../../core/access.hpp"
#include "../../../core/constants.hpp"
#include "../../../core/simulation.hpp"

#include <sycl/sycl.hpp>

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
            namespace linear
            {

                /**
                 * @brief This namespace contains all kernels for the two-lattice algorithm.
                 */
                namespace kernels
                {

                    /**
                     * @brief   This kernel performs the streaming step of a two-lattice iteration.
                     * 
                     * @tparam  LBMAccessor an lbm accessor object, that is, any object whose class inherits 
                     *          from `lbm::core::access::LBMAccessorObject`
                     */
                    template <class LBMAccessor> class StreamKernel 
                    {
                        static_assert
                        (
                            std::is_base_of<core::access::LBMAccessorObject, LBMAccessor>::value, 
                            "Template class must have base class lbm::core::access::LBMAccessorObject."
                        );

                        private:

                            sycl::accessor<uint8_t, 1, constants::read> phase_info_accessor;
                            sycl::accessor<double, 1, constants::read> source_accessor;
                            sycl::accessor<double, 1, constants::write> destination_accessor;

                            LBMAccessor lbm_accessor;

                            unsigned int vertical_nodes;
                            unsigned int horizontal_nodes;

                        public:

                            /**
                             * @brief   Constructor for a new `StreamKernel` object.
                             *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                             * 
                             * @param[in]   phase_info_accessor     reference to a SYCL accessor to the phase information vector
                             * @param[in]   source_accessor         reference to a SYCL accessor to the source distribution values vector
                             * @param[in]   destination_accessor    reference to a SYCL accessor to the destination distribution values vector
                             * @param[in]   lbm_accessor            reference to an instance of an LBM accessor object
                             * @param[in]   properties              reference to a struct containing the simulation properties
                             */
                            explicit StreamKernel
                            (
                                const sycl::accessor<uint8_t, 1, constants::read> &phase_info_accessor,
                                const sycl::accessor<double, 1, constants::read> &source_accessor,
                                const sycl::accessor<double, 1, constants::write> &destination_accessor,
                                const LBMAccessor &lbm_accessor,
                                const core::Properties &properties
                            )
                            : 
                            phase_info_accessor(phase_info_accessor), 
                            source_accessor(source_accessor),
                            destination_accessor(destination_accessor), 
                            lbm_accessor(lbm_accessor),
                            vertical_nodes(properties.vertical_nodes),
                            horizontal_nodes(properties.horizontal_nodes)
                            {}

                            /**
                             * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                             * 
                             * @param item the SYCL work item processing this kernel, which is set by the SYCL runtime
                             */
                            void operator()(const sycl::item<1> &item) const
                            {
                                auto id = item.get_linear_id();

                                if(!core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_info_accessor[id]))
                                {
                                    for (const auto& direction : core::constants::all_directions)
                                    {
                                        destination_accessor[lbm_accessor(id, direction)] =
                                            source_accessor[lbm_accessor(lbm::core::access::get_neighbor(id, core::invert_direction(direction), horizontal_nodes), direction)];
                                    }
                                }
                            }
                    };

                    /**
                     * @brief   This kernel performs the update of the macroscopic observables.
                     * 
                     * @tparam  LBMAccessor an lbm accessor object, that is, any object whose class inherits 
                     *          from `lbm::core::access::LBMAccessorObject`
                     */
                    template <class LBMAccessor> class MacroscopicObservablesKernel
                    {
                        static_assert
                        (
                            std::is_base_of<lbm::core::access::LBMAccessorObject, LBMAccessor>::value, 
                            "Template class must have base class lbm::core::access::LBMAccessorObject."
                        );

                        private:

                            sycl::accessor<uint8_t, 1, constants::read> phase_info_accessor;
                            sycl::accessor<double, 1, constants::read> destination_accessor;

                            sycl::accessor<double, 1, constants::write> densities_accessor;
                            sycl::accessor<double, 1, constants::write> x_velocities_accessor;
                            sycl::accessor<double, 1, constants::write> y_velocities_accessor;

                            LBMAccessor lbm_accessor;

                            unsigned int vertical_nodes;
                            unsigned int horizontal_nodes;
                            unsigned int domain_node_count;
                            unsigned int iteration;

                        public:

                            /**
                             * @brief   Constructor for a new `MacroscopicObservablesKernel` object.
                             *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                             * 
                             * @param[in]   phase_info_accessor     reference to a SYCL accessor to the phase information vector
                             * @param[in]   destination_accessor    reference to a SYCL accessor to the destination distribution values vector
                             * @param[in]   densities_accessor      reference to a SYCL accessor to the densities vector
                             * @param[in]   x_velocities_accessor   reference to a SYCL accessor to the x velocities vector
                             * @param[in]   y_velocities_accessor   reference to a SYCL accessor to the y velocities vector
                             * @param[in]   lbm_accessor            reference to an instance of an LBM accessor object
                             * @param[in]   properties              reference to a struct containing the simulation properties
                             * @param[in]   iteration               the iteration in which this kernel is accessed
                             */
                            explicit MacroscopicObservablesKernel
                            (
                                const sycl::accessor<uint8_t, 1, constants::read> phase_info_accessor,
                                const sycl::accessor<double, 1, constants::read> destination_accessor,
                                const sycl::accessor<double, 1, constants::write> densities_accessor,
                                const sycl::accessor<double, 1, constants::write> x_velocities_accessor,
                                const sycl::accessor<double, 1, constants::write> y_velocities_accessor,
                                const LBMAccessor &lbm_accessor,
                                const core::Properties &properties,
                                const unsigned int iteration
                            )
                            : 
                            phase_info_accessor(phase_info_accessor), 
                            destination_accessor(destination_accessor), 
                            densities_accessor(densities_accessor),
                            x_velocities_accessor(x_velocities_accessor),
                            y_velocities_accessor(y_velocities_accessor),
                            lbm_accessor(lbm_accessor),
                            vertical_nodes(properties.vertical_nodes),
                            horizontal_nodes(properties.horizontal_nodes),
                            domain_node_count(properties.domain_node_count),
                            iteration(iteration)
                            {}

                            /**
                             * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                             * 
                             * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                             */
                            void operator()(const sycl::item<1> &item) const
                            {
                                auto id = item.get_linear_id();

                                if(!core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_info_accessor[id]))
                                {
                                    unsigned int iteration_node_offset =
                                        lbm::core::access::results::get_result_index_no_ghosts(id, horizontal_nodes, domain_node_count, iteration);
                                    double dist_vals[9];
                                    double density = 0;

                                    for (const auto& direction : core::constants::all_directions)
                                    {
                                        dist_vals[direction] = destination_accessor[lbm_accessor(id, direction)];
                                        density += dist_vals[direction];
                                    }
                                    densities_accessor[iteration_node_offset] = density;
                                    
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
                                    x_velocities_accessor[iteration_node_offset] = flow_velocity[0];
                                    y_velocities_accessor[iteration_node_offset] = flow_velocity[1];
                                }
                            }
                    };

                    /**
                     * @brief   This kernel performs the collision step of a two-lattice iteration.
                     * 
                     * @tparam  LBMAccessor an lbm accessor object, that is, any object whose class inherits 
                     *          from `lbm::core::access::LBMAccessorObject`
                     */
                    template <class LBMAccessor> class CollideKernel
                    {
                        static_assert
                        (
                            std::is_base_of<lbm::core::access::LBMAccessorObject, LBMAccessor>::value, 
                            "Template class must have base class lbm::core::access::LBMAccessorObject."
                        );

                        private:

                            sycl::accessor<uint8_t, 1, constants::read> phase_info_accessor;

                            sycl::accessor<double, 1, constants::read_write> destination_accessor;

                            sycl::accessor<double, 1, constants::write> densities_accessor;
                            sycl::accessor<double, 1, constants::write> x_velocities_accessor;
                            sycl::accessor<double, 1, constants::write> y_velocities_accessor;

                            LBMAccessor lbm_accessor;

                            unsigned int vertical_nodes;
                            unsigned int horizontal_nodes;
                            unsigned int domain_node_count;
                            unsigned int iteration;
                            double relaxation_time;

                        public:

                            /**
                             * @brief Constructor for a new `CollideKernel` object.
                             *        Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                             * 
                             * @param[in]   phase_info_accessor     reference to a SYCL accessor to the phase information vector
                             * @param[in]   destination_accessor    reference to a SYCL accessor to the destination distribution values vector
                             * @param[in]   densities_accessor      reference to a SYCL accessor to the densities vector
                             * @param[in]   x_velocities_accessor   reference to a SYCL accessor to the x velocities vector
                             * @param[in]   y_velocities_accessor   reference to a SYCL accessor to the y velocities vector
                             * @param[in]   lbm_accessor            reference to an instance of an LBM accessor object
                             * @param[in]   properties              reference to a struct containing the simulation properties
                             * @param[in]   iteration               the iteration in which this kernel is accessed
                             */
                            explicit CollideKernel
                            (
                                const sycl::accessor<uint8_t, 1, constants::read> phase_info_accessor,
                                const sycl::accessor<double, 1, constants::read_write> destination_accessor,
                                const sycl::accessor<double, 1, constants::write> densities_accessor,
                                const sycl::accessor<double, 1, constants::write> x_velocities_accessor,
                                const sycl::accessor<double, 1, constants::write> y_velocities_accessor,
                                const LBMAccessor &lbm_accessor,
                                const core::Properties &properties,
                                const unsigned int iteration
                            )
                            : 
                            phase_info_accessor(phase_info_accessor), 
                            destination_accessor(destination_accessor), 
                            densities_accessor(densities_accessor),
                            x_velocities_accessor(x_velocities_accessor),
                            y_velocities_accessor(y_velocities_accessor),
                            lbm_accessor(lbm_accessor),
                            vertical_nodes(properties.vertical_nodes),
                            horizontal_nodes(properties.horizontal_nodes),
                            domain_node_count(properties.domain_node_count),
                            iteration(iteration),
                            relaxation_time(properties.relaxation_time)
                            {}

                            /**
                             * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                             * 
                             * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                             */
                            void operator()(sycl::item<1> item) const
                            {
                                auto id = item.get_linear_id();

                                if(!lbm::core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_info_accessor[id]))
                                {
                                    unsigned int iteration_node_offset =
                                        lbm::core::access::results::get_result_index_no_ghosts(id, horizontal_nodes, domain_node_count, iteration);

                                    double& x_velocity = x_velocities_accessor[iteration_node_offset];
                                    double& y_velocity = y_velocities_accessor[iteration_node_offset];
                                    double& density = densities_accessor[iteration_node_offset];

                                    int velocity_x_component = 0; 
                                    int velocity_y_component = 0; 

                                    double value;
                                    double result;

                                    for (const auto& direction : core::constants::all_directions)
                                    {
                                        value = destination_accessor[lbm_accessor(id, direction)];

                                        velocity_x_component = (direction % 3) - 1; 
                                        velocity_y_component = (direction / 3) - 1; 

                                        result = core::constants::weights[direction] *
                                            (
                                                density + 3 * (velocity_x_component * x_velocity + velocity_y_component * y_velocity)
                                                + 9.0/2 * std::pow(velocity_x_component * x_velocity + velocity_y_component * y_velocity, 2)
                                                - 3.0/2 * (x_velocity * x_velocity + y_velocity * y_velocity)
                                            );

                                        result = -(1/relaxation_time) * (value - result) + value;
                                        destination_accessor[lbm_accessor(id, direction)] = result;
                                    }
                                }
                            }
                    };

                    /**
                     * @brief   This kernel performs the streaming step, the update of the macroscopic observables and 
                     *          collision step of a two-lattice iteration.
                     * 
                     * @tparam  LBMAccessor an lbm accessor object, that is, any object whose class inherits 
                     *          from `lbm::core::access::LBMAccessorObject`
                     */          
                    template <class LBMAccessor> class StreamCollideKernel
                    {
                        static_assert
                        (
                            std::is_base_of<lbm::core::access::LBMAccessorObject, LBMAccessor>::value, 
                            "Template class must have base class lbm::core::access::LBMAccessorObject."
                        );

                        private:

                            sycl::accessor<uint8_t, 1, constants::read> phase_info_accessor;
                            sycl::accessor<double, 1, constants::read> source_accessor;

                            sycl::accessor<double, 1, constants::read_write> destination_accessor;
                            sycl::accessor<double, 1, constants::read_write> densities_accessor;
                            sycl::accessor<double, 1, constants::read_write> x_velocities_accessor;
                            sycl::accessor<double, 1, constants::read_write> y_velocities_accessor;

                            LBMAccessor lbm_accessor;

                            unsigned int vertical_nodes;
                            unsigned int horizontal_nodes;
                            unsigned int domain_node_count;
                            unsigned int iteration;
                            double relaxation_time;

                        public:

                            /**
                             * @brief Constructor for a new `StreamCollideKernel` object.
                             *        Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
                             * 
                             * @param[in]   phase_info_accessor     reference to a SYCL accessor to the phase information vector
                             * @param[in]   destination_accessor    reference to a SYCL accessor to the destination distribution values vector
                             * @param[in]   densities_accessor      reference to a SYCL accessor to the densities vector
                             * @param[in]   x_velocities_accessor   reference to a SYCL accessor to the x velocities vector
                             * @param[in]   y_velocities_accessor   reference to a SYCL accessor to the y velocities vector
                             * @param[in]   lbm_accessor            reference to an instance of an LBM accessor object
                             * @param[in]   properties              reference to a struct containing the simulation properties
                             * @param[in]   iteration               the iteration in which this kernel is accessed
                             */
                            StreamCollideKernel
                            (
                                const sycl::accessor<uint8_t, 1, constants::read> &phase_info_accessor,
                                const sycl::accessor<double, 1, constants::read> &source_accessor,
                                const sycl::accessor<double, 1, constants::read_write> &destination_accessor,
                                const sycl::accessor<double, 1, constants::read_write> &densities_accessor,
                                const sycl::accessor<double, 1, constants::read_write> &x_velocities_accessor,
                                const sycl::accessor<double, 1, constants::read_write> &y_velocities_accessor,
                                const LBMAccessor &lbm_accessor,
                                const core::Properties &properties,
                                const unsigned int iteration
                            )
                            : 
                            phase_info_accessor(phase_info_accessor), 
                            source_accessor(source_accessor), 
                            destination_accessor(destination_accessor), 
                            densities_accessor(densities_accessor),
                            x_velocities_accessor(x_velocities_accessor),
                            y_velocities_accessor(y_velocities_accessor),
                            lbm_accessor(lbm_accessor),
                            vertical_nodes(properties.vertical_nodes),
                            horizontal_nodes(properties.horizontal_nodes),
                            domain_node_count(properties.domain_node_count),
                            iteration(iteration),
                            relaxation_time(properties.relaxation_time)
                            {}

                            /**
                             * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                             * 
                             * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                             */
                            void operator()(const sycl::item<1> &item) const
                            {
                                auto id = item.get_linear_id();

                                if(!lbm::core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_info_accessor[id]))
                                {
                                    unsigned int iteration_node_offset =
                                        lbm::core::access::results::get_result_index_no_ghosts(id, horizontal_nodes, domain_node_count, iteration);

                                    // Streaming
                                    for (const auto& direction : core::constants::all_directions)
                                    {
                                        destination_accessor[lbm_accessor(id, direction)] =
                                            source_accessor[lbm_accessor(lbm::core::access::get_neighbor(id, core::invert_direction(direction), horizontal_nodes), direction)];
                                    }

                                    // Macroscopic observables
                                    double value = 0;
                                    double result = 0;
                                    double density = 0;
                                    int velocity_x_component = 0; 
                                    int velocity_y_component = 0; 
                                    double flow_velocity_x = 0;
                                    double flow_velocity_y = 0;

                                    for (const auto& direction : core::constants::all_directions)
                                    {
                                        value = destination_accessor[lbm_accessor(id, direction)];
                                        density += value;
                                        velocity_x_component = direction % 3 - 1; 
                                        velocity_y_component = direction / 3 - 1; 
                                        flow_velocity_x += value * velocity_x_component;
                                        flow_velocity_y += value * velocity_y_component;
                                    }
                                    
                                    densities_accessor[iteration_node_offset] = density;
                                    x_velocities_accessor[iteration_node_offset] = flow_velocity_x;
                                    y_velocities_accessor[iteration_node_offset] = flow_velocity_y;

                                    // Collision
                                    for (const auto& direction : core::constants::all_directions)
                                    {
                                        value = destination_accessor[lbm_accessor(id, direction)];

                                        velocity_x_component = (direction % 3) - 1; 
                                        velocity_y_component = (direction / 3) - 1; 

                                        result = core::constants::weights[direction] *
                                            (
                                                density + 3 * (velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y)
                                                + 9.0/2 * std::pow(velocity_x_component * flow_velocity_x + velocity_y_component * flow_velocity_y, 2)
                                                - 3.0/2 * (flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y)
                                            );

                                        result = -(1/relaxation_time) * (value - result) + value;
                                        destination_accessor[lbm_accessor(id, direction)] = result;
                                    }
                                }
                            }
                    };

                } // ! namespace linear

            } // ! namespace kernels

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! TL_KERNELS_HPP