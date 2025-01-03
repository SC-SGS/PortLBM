/**
 * @file        linear_two_lattice_kernels.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of kernels for the two-lattice algorithm
 *              with linear work item evaluation.
 * 
 * @version     1.1
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LINEAR_TWO_LATTICE_KERNELS_HPP
#define LINEAR_TWO_LATTICE_KERNELS_HPP

// Dependencies on other LBM core features
#include "../../../core/access.hpp"
#include "../../../core/constants.hpp"
#include "../../../core/simulation.hpp"

// SYCL
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
                     * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                     */
                    template<core::access::experimental::AccessorConcept A>
                    class StreamKernel 
                    {

                        private:

                        uint8_t *phase_information;
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
                        vertical_nodes(simulation.properties->vertical_nodes),
                        horizontal_nodes(simulation.properties->horizontal_nodes)
                        {}

                        /**
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param id the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::id<1> &id) const
                        {
                            if(!core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_information[id]))
                            {
                                for (const auto& direction : core::constants::all_directions)
                                {
                                    destination[A::at(id, direction, horizontal_nodes * vertical_nodes)] =
                                        source[A::at(lbm::core::access::get_neighbor(id, core::invert_direction(direction), horizontal_nodes), direction, horizontal_nodes * vertical_nodes)];
                                }
                            }
                        }
                    };

                    /**
                     * @brief   This kernel performs the update of the macroscopic observables.
                     * 
                     * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                     */
                    template<core::access::experimental::AccessorConcept A>
                    class MacroscopicObservablesKernel
                    {
                        private:

                        uint8_t *phase_information;
                        double *destination;

                        double *densities;
                        double *x_velocities;
                        double *y_velocities;
                        double *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;

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
                        vertical_nodes(simulation.properties->vertical_nodes),
                        horizontal_nodes(simulation.properties->horizontal_nodes)
                        {}

                        /**
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::id<1> &id) const 
                        {
                            if(!core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_information[id]))
                            {
                                unsigned int iteration_node_offset =
                                    lbm::core::access::results::get_result_index_no_ghosts(id, horizontal_nodes);
                                double dist_vals[9];
                                double density = 0;

                                for (const auto& direction : core::constants::all_directions)
                                {
                                    dist_vals[direction] = destination[A::at(id, direction, horizontal_nodes * vertical_nodes)];
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
                                x_velocities[iteration_node_offset] = flow_velocity[0];
                                y_velocities[iteration_node_offset] = flow_velocity[1];
                                absolute_velocity_values[iteration_node_offset] = sycl::sqrt(flow_velocity[0] * flow_velocity[0] + flow_velocity[1] * flow_velocity[1]);
                            }
                        }
                    };

                    /**
                     * @brief   This kernel performs the collision step of a two-lattice iteration.
                     * 
                     * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                     */
                    template<core::access::experimental::AccessorConcept A> 
                    class CollideKernel
                    {

                        private:

                        uint8_t *phase_information;
                        double *destination;

                        double *densities;
                        double *x_velocities;
                        double *y_velocities;
                        double *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;
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
                        vertical_nodes(simulation.properties->vertical_nodes),
                        horizontal_nodes(simulation.properties->horizontal_nodes),
                        relaxation_time_inverse(1 / simulation.properties->relaxation_time)
                        {}

                        /**
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::id<1> &id) const 
                        {
                            if(!lbm::core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_information[id]))
                            {
                                unsigned int iteration_node_offset =
                                    lbm::core::access::results::get_result_index_no_ghosts(id, horizontal_nodes);

                                double& x_velocity = x_velocities[iteration_node_offset];
                                double& y_velocity = y_velocities[iteration_node_offset];
                                double& density = densities[iteration_node_offset];

                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 

                                double value;
                                double result;

                                for (const auto& direction : core::constants::all_directions)
                                {
                                    value = destination[A::at(id, direction, horizontal_nodes * vertical_nodes)];

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
                                    destination[A::at(id, direction, horizontal_nodes * vertical_nodes)] = result;
                                }
                            }
                        }
                    };

                    /**
                     * @brief   This kernel performs the streaming step, the update of the macroscopic observables and 
                     *          collision step of a two-lattice iteration.
                     * 
                     * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                     */          
                    template<core::access::experimental::AccessorConcept A>
                    class StreamCollideKernel
                    {
                        private:

                        uint8_t *phase_information;
                        double *source;
                        double *destination;

                        double *densities;
                        double *x_velocities;
                        double *y_velocities;
                        double *absolute_velocity_values;

                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;
                        double relaxation_time_inverse;

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
                        vertical_nodes(simulation.properties->vertical_nodes),
                        horizontal_nodes(simulation.properties->horizontal_nodes),
                        relaxation_time_inverse(1 / simulation.properties->relaxation_time)
                        {}

                        /**
                         * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
                         * 
                         * @param[in] item the SYCL work item processing this kernel, which is set by the SYCL runtime
                         */
                        void operator()(const sycl::item<1> &id) const
                        {
                            if(!lbm::core::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_information[id]))
                            {
                                unsigned int iteration_node_offset =
                                    lbm::core::access::results::get_result_index_no_ghosts(id, horizontal_nodes);

                                double distribution_values[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                                double result = 0;
                                double density = 0;
                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 
                                double flow_velocity_x = 0;
                                double flow_velocity_y = 0;

                                // Loading distribution values and macroscopic observables
                                for (const auto& direction : core::constants::all_directions)
                                {
                                    // Loading distribution values
                                    distribution_values[direction] = source[A::at(lbm::core::access::get_neighbor(id, core::invert_direction(direction), horizontal_nodes), 
                                        direction, 
                                        horizontal_nodes * vertical_nodes
                                    )];

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
                                absolute_velocity_values[iteration_node_offset] = sycl::sqrt(flow_velocity_x * flow_velocity_x + flow_velocity_y * flow_velocity_y);

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

                                    result = -relaxation_time_inverse * (distribution_values[direction] - result) + distribution_values[direction];
                                    destination[A::at(id, direction, horizontal_nodes * vertical_nodes)] = result;
                                }
                            }
                        }
                    };

                    /**
                     * @brief This namespace contains GPU-based implementations of various methods concerning boundaries.
                     */
                    namespace boundaries
                    {

                        /**
                         * @brief   Kernel for emplacing the bounce-back values.
                         * 
                         * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                         */
                        template<core::access::experimental::AccessorConcept A>
                        class EmplaceBounceBackKernel
                        {
                            private:

                            uint8_t *phase_information;
                            double *destination;

                            unsigned int horizontal_nodes;
                            unsigned int total_nodes;

                            public:

                            EmplaceBounceBackKernel(const core::Simulation &simulation):
                            phase_information(simulation.data->phase_information),
                            destination(simulation.data->distribution_values_1),
                            horizontal_nodes(simulation.properties->horizontal_nodes),
                            total_nodes(simulation.properties->buffered_node_count)
                            {}

                            void operator()(const sycl::item<1> &id) const
                            {
                                if(phase_information[id])
                                {
                                    for(const auto& dir : core::constants::all_directions)
                                    {
                                        destination[A::at(id, dir, total_nodes)] =
                                        destination[A::at(core::access::get_neighbor(id, dir, horizontal_nodes), core::invert_direction(dir), total_nodes)];
                                        // destination[A::at(core::access::get_neighbor(id, dir, horizontal_nodes), core::invert_direction(dir), total_nodes)];               
                                    }
                                }
                            }
                        };

                        /**
                         * @brief Kernel for updating the inlets and outlets.
                         * 
                         * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                         */
                        template<core::access::experimental::AccessorConcept A>
                        class InoutUpdateKernel
                        {
                            private:

                            double *source;
                            double outlet_density;
                            double outlet_density_inverse;
                            unsigned int horizontal_nodes;
                            unsigned int vertical_nodes;

                            public:

                            InoutUpdateKernel(const core::Simulation &simulation):
                            source(simulation.data->distribution_values_0),
                            outlet_density(simulation.properties->outlet_density),
                            outlet_density_inverse(1 / simulation.properties->outlet_density),
                            horizontal_nodes(simulation.properties->horizontal_nodes),
                            vertical_nodes(simulation.properties->vertical_nodes)
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

                                unsigned int current_border_node = core::access::get_node_index(horizontal_nodes - 2, y, horizontal_nodes);

                                f_1 = source[A::at(core::access::get_neighbor(current_border_node, 7, horizontal_nodes), 1, horizontal_nodes * vertical_nodes)];
                                f_4 = source[A::at(current_border_node, 4, horizontal_nodes * vertical_nodes)];
                                f_7 = source[A::at(core::access::get_neighbor(current_border_node, 1, horizontal_nodes), 7, horizontal_nodes * vertical_nodes)];

                                for(int i = 0; i < 3; ++i)
                                {
                                    f_inverse[i] = source[
                                        A::at(
                                            core::access::get_neighbor(current_border_node, missing[i], horizontal_nodes), 8 - missing[i], horizontal_nodes * vertical_nodes
                                        )];
                                }

                                double x_velocity = 1 - outlet_density_inverse * (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

                                source[A::at(core::access::get_neighbor(current_border_node, 8 - missing[0], horizontal_nodes), missing[0], horizontal_nodes * vertical_nodes)]
                                    = f_inverse[0] - 0.5 * (f_1 - f_7) + 1.0/6 * outlet_density * x_velocity;
                                source[A::at(core::access::get_neighbor(current_border_node, 8 - missing[1], horizontal_nodes), missing[1], horizontal_nodes * vertical_nodes)]
                                    = f_inverse[1] + (2.0/3) * outlet_density * x_velocity;
                                source[A::at(core::access::get_neighbor(current_border_node, 8 - missing[2], horizontal_nodes), missing[2], horizontal_nodes * vertical_nodes)]
                                    = f_inverse[2] + 0.5 * (f_1 - f_7) + 1.0/6 * outlet_density * x_velocity;
                            }
                        };

                        /**
                         * @brief Kernel for updating the inlets and outlets.
                         * 
                         * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                         */
                        template<core::access::experimental::AccessorConcept A>
                        class LegacyInoutUpdateKernel
                        {

                            private:

                            sycl::accessor<double, 1, constants::read_write> distribution_values_accessor;
                            double densities[2];
                            unsigned int horizontal_nodes;
                            unsigned int vertical_nodes;

                            public:

                            LegacyInoutUpdateKernel
                            (
                                const sycl::accessor<double, 1, constants::read_write> &distribution_values_accessor,
                                const core::Properties &properties
                            )
                            : 
                            distribution_values_accessor(distribution_values_accessor),
                            densities(properties.inlet_density, properties.outlet_density),
                            horizontal_nodes(properties.horizontal_nodes),
                            vertical_nodes(properties.vertical_nodes)
                            {}

                            void operator()(const sycl::id<2> &id) const
                            {
                                auto id_y = id[0]; // in {0,1, ... , horizontal_nodes - 5}
                                auto id_x = id[1]; // in {0,1}

                                unsigned int missing[3];

                                double f_inverse[3];
                                double f_1 = 0;
                                double f_4 = 0;
                                double f_7 = 0;

                                double density = densities[id_x];

                                for(int i = 0; i < 3; ++i)
                                {
                                    // x = 0: inlet boundary, missing directions: 2, 5, 8
                                    // x = 1: outlet boundary, missing directions: 0, 3, 6
                                    missing[i] = id_x * 3 * i + (1 - id_x) * (2 + 3 * i);
                                }

                                // Setup actual coordinates
                                id_y = id_y + 2;                                         // y coordinate: offset by two
                                id_x = 1 + id_x * (horizontal_nodes - 3);   // x coordinate: 0 -> 1, 1 -> horizontal_nodes - 2

                                unsigned int current_border_node = core::access::get_node_index(id_x, id_y, horizontal_nodes);

                                f_1 = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 7, horizontal_nodes), 1, horizontal_nodes * vertical_nodes)];
                                f_4 = distribution_values_accessor[A::at(current_border_node, 4, horizontal_nodes * vertical_nodes)];
                                f_7 = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 1, horizontal_nodes), 7, horizontal_nodes * vertical_nodes)];

                                for(int i = 0; i < 3; ++i)
                                {
                                    f_inverse[i] = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, missing[i], horizontal_nodes), 8 - missing[i], horizontal_nodes * vertical_nodes)];
                                }

                                double x_velocity = 1 - (1 / density) * (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

                                distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[0], horizontal_nodes), missing[0], horizontal_nodes * vertical_nodes)]
                                    = f_inverse[0] - 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;
                                distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[1], horizontal_nodes), missing[1], horizontal_nodes * vertical_nodes)]
                                    = f_inverse[1] + (2.0/3) * density * x_velocity;
                                distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[2], horizontal_nodes), missing[2], horizontal_nodes * vertical_nodes)]
                                    = f_inverse[2] + 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;
                            }
                        };


                        /**
                         * @brief   Debug method that enacts a CPU-based computation that is semantically identical to `lbm::gpu::boundaries::linear::InoutUpdateKernel::operator()`.
                         * 
                         * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                         * 
                         * @param densities 
                         * @param properties 
                         * @param distribution_values_accessor 
                         */
                        template<core::access::experimental::AccessorConcept A> 
                        void inout_update_debugger
                        (
                            const double densities[2],
                            const core::Properties &properties,
                            std::vector<double> &distribution_values_accessor
                        )
                        {
                            std::cout << "Number of outer iterations (y): " << properties.vertical_nodes - 4 << "\n";
                            for(int outer = 0; outer < properties.vertical_nodes - 4; ++outer)
                            {
                                int id = outer;

                                unsigned int missing[3];

                                double f_inverse[3];
                                double f_1 = 0;
                                double f_4 = 0;
                                double f_7 = 0;

                                double density = densities[1];
                                std::cout << "Density: " << density << "\n";
                                std::cout << std::setprecision(6) << std::fixed;
                                std::cout << "Missing variables: ";
                                for(int i = 0; i < 3; ++i)
                                {
                                    // x = 1: outlet boundary, missing directions: 0, 3, 6
                                    missing[i] = 3 * i;
                                    std::cout << missing[i] << " ";
                                }
                                std::cout << "\n";

                                // Setup actual coordinates
                                id = id + 2;                                         // y coordinate: offset by two
                                unsigned int current_border_node = core::access::get_node_index(properties.horizontal_nodes - 2, id, properties.horizontal_nodes);
                                std::cout << "Treating node with coordinates (x, y) = (" << properties.horizontal_nodes - 2 << ", " << id << "), with node index " << current_border_node << "\n";
                                
                                unsigned int known_directions[6] = {1, 4, 7, 8 - missing[0], 8 - missing[1], 8 - missing[2]};
                                
                                std::cout << "Known directions: ";
                                for(int i = 0; i < 6; ++i)
                                {
                                    std::cout << known_directions[i] << " ";
                                }
                                std::cout << "\n";

                                f_1 = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 7, properties.horizontal_nodes), 1, properties.buffered_node_count)];
                                f_4 = distribution_values_accessor[A::at(current_border_node, 4, properties.buffered_node_count)];
                                f_7 = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 1, properties.horizontal_nodes), 7, properties.buffered_node_count)];

                                for(int i = 0; i < 3; ++i)
                                {
                                    f_inverse[i] = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, missing[i], properties.horizontal_nodes), 8 - missing[i], properties.buffered_node_count)];
                                    std::cout << "The corresponding neighbor of this node in direction " << 
                                                missing[i] << " is believed to have index " << core::access::get_neighbor(current_border_node, missing[i], properties.horizontal_nodes) << "\n";
                                }

                                std::cout << "\n";

                                std::cout << "Known distribution values: \n";
                                std::cout << "f_1 = " << f_1 << "\n";
                                std::cout << "f_4 = " << f_4 << "\n";
                                std::cout << "f_7 = " << f_7 << "\n";

                                for(int i = 0; i < 3; ++i)
                                {
                                    std::cout << "f_" << 8 - missing[i] << " = " << f_inverse[i] << "\n";
                                }

                                std::cout << "\n";

                                double x_velocity = 1 - (1 / density) * (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

                                std::cout << "Velocity: " << x_velocity << "\n";

                                distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[0], properties.horizontal_nodes), missing[0], properties.buffered_node_count)]
                                    = f_inverse[0] - 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;
                                distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[1], properties.horizontal_nodes), missing[1], properties.buffered_node_count)]
                                    = f_inverse[1] + (2.0/3) * density * x_velocity;
                                distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[2], properties.horizontal_nodes), missing[2], properties.buffered_node_count)]
                                    = f_inverse[2] + 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;

                                std::cout << "Calculated distribution values: \n";

                                for(int i = 0; i < 3; ++i)
                                {
                                    std::cout << "f[" << missing[i] << "] = " 
                                            << distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[i], properties.horizontal_nodes), missing[i], properties.buffered_node_count)] << "\n";
                                }
                                std::cout << "\n";
                                std::cout << std::setprecision(3) << std::fixed;
                            }
                        }

                        /**
                         * @brief   Debug method that enacts a CPU-based computation that is semantically identical to `lbm::gpu::boundaries::linear::InoutUpdateKernel::operator()`.
                         * 
                         * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                         * 
                         * @param densities 
                         * @param properties 
                         * @param distribution_values_accessor 
                         */
                        template<core::access::experimental::AccessorConcept A> 
                        void legacy_inout_update_debugger
                        (
                            const double densities[2],
                            const core::Properties &properties,
                            std::vector<double> &distribution_values_accessor
                        )
                        {
                            std::cout << "Number of outer iterations (y): " << properties.vertical_nodes - 4 << "\n";
                            for(int outer = 0; outer < properties.vertical_nodes - 4; ++outer)
                            {
                                for(int inner = 0; inner < 2; ++inner)
                                {
                                    int id[2] = {outer, inner};

                                    auto id_y = id[0]; // in {0,1, ... , horizontal_nodes - 5}
                                    auto id_x = id[1]; // in {0,1}

                                    unsigned int missing[3];

                                    double f_inverse[3];
                                    double f_1 = 0;
                                    double f_4 = 0;
                                    double f_7 = 0;

                                    double density = densities[id_x];
                                    std::cout << "Density: " << density << "\n";
                                    std::cout << std::setprecision(6) << std::fixed;
                                    std::cout << "Missing variables: ";
                                    for(int i = 0; i < 3; ++i)
                                    {
                                        // x = 0: inlet boundary, missing directions: 2, 5, 8
                                        // x = 1: outlet boundary, missing directions: 0, 3, 6
                                        missing[i] = id_x * 3 * i + (1 - id_x) * (2 + 3 * i);
                                        std::cout << missing[i] << " ";
                                    }
                                    std::cout << "\n";

                                    // Setup actual coordinates
                                    id_y = id_y + 2;                                         // y coordinate: offset by two
                                    id_x = 1 + id_x * (properties.horizontal_nodes - 3);   // x coordinate: 0 -> 1, 1 -> horizontal_nodes - 2
                                    unsigned int current_border_node = core::access::get_node_index(id_x, id_y, properties.horizontal_nodes);
                                    std::cout << "Treating node with coordinates (x, y) = (" << id_x << ", " << id_y << "), with node index " << current_border_node << "\n";
                                    
                                    unsigned int known_directions[6] = {1, 4, 7, 8 - missing[0], 8 - missing[1], 8 - missing[2]};
                                    
                                    std::cout << "Known directions: ";
                                    for(int i = 0; i < 6; ++i)
                                    {
                                        std::cout << known_directions[i] << " ";
                                    }
                                    std::cout << "\n";

                                    f_1 = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 7, properties.horizontal_nodes), 1, properties.buffered_node_count)];
                                    f_4 = distribution_values_accessor[A::at(current_border_node, 4, properties.buffered_node_count)];
                                    f_7 = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 1, properties.horizontal_nodes), 7, properties.buffered_node_count)];

                                    for(int i = 0; i < 3; ++i)
                                    {
                                        f_inverse[i] = distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, missing[i], properties.horizontal_nodes), 8 - missing[i], properties.buffered_node_count)];
                                        std::cout << "The corresponding neighbor of this node in direction " << 
                                                  missing[i] << " is believed to have index " << core::access::get_neighbor(current_border_node, missing[i], properties.horizontal_nodes) << "\n";
                                    }

                                    std::cout << "\n";

                                    std::cout << "Known distribution values: \n";
                                    std::cout << "f_1 = " << f_1 << "\n";
                                    std::cout << "f_4 = " << f_4 << "\n";
                                    std::cout << "f_7 = " << f_7 << "\n";

                                    for(int i = 0; i < 3; ++i)
                                    {
                                        std::cout << "f_" << 8 - missing[i] << " = " << f_inverse[i] << "\n";
                                    }

                                    std::cout << "\n";

                                    double x_velocity = 1 - (1 / density) * (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

                                    std::cout << "Velocity: " << x_velocity << "\n";

                                    distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[0], properties.horizontal_nodes), missing[0], properties.buffered_node_count)]
                                        = f_inverse[0] - 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;
                                    distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[1], properties.horizontal_nodes), missing[1], properties.buffered_node_count)]
                                        = f_inverse[1] + (2.0/3) * density * x_velocity;
                                    distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[2], properties.horizontal_nodes), missing[2], properties.buffered_node_count)]
                                        = f_inverse[2] + 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;

                                    std::cout << "Calculated distribution values: \n";

                                    for(int i = 0; i < 3; ++i)
                                    {
                                        std::cout << "f[" << missing[i] << "] = " 
                                                << distribution_values_accessor[A::at(core::access::get_neighbor(current_border_node, 8 - missing[i], properties.horizontal_nodes), missing[i], properties.buffered_node_count)] << "\n";
                                    }
                                    std::cout << "\n";
                                    std::cout << std::setprecision(3) << std::fixed;
                                }
                            }
                        }

                    } // ! namespace boundaries

                } // ! namespace linear

            } // ! namespace kernels

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! TL_KERNELS_HPP