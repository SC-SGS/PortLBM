/**
 * @file        gpu_two_lattice.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header, all functionality of the GPU-based two lattice algorithm is declared.
 *              At the moment, the name is misleading since the functionality is not yet realized on the GPU.
 *              In this early version, all simulations are still run on the CPU variant as defined in my SimTech project work
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann.
 *              Currently, the objective is to prepare the data structures required for the GPU variant.
 *              Compatiblity to the CPU implementation is achieved by additional measures.
 *              Eventually, all functionality is realized on the GPU, and any deprecated features will be removed.
 * 
 * @version     0.1
 * 
 * @date        2024-10-03
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef GPU_TWO_LATTICE_HPP
#define GPU_TWO_LATTICE_HPP

#include "../../core/access.hpp"
#include "../../core/boundaries.hpp"
#include "../../core/collision.hpp"
#include "../../console/console_output.hpp"
#include "../../core/domain_initialization.hpp"
#include "../../core/simulation.hpp"

#include <sycl/sycl.hpp>

#include <vector>

namespace lbm
{
    /**
     * @brief This namespace contains the GPU implementations of the two-lattice and the swap algorithm.
     */
    namespace gpu
    {
        /**
         * @brief This namespace contains the GPU implementation of the two-lattice algorithm.
         */
        namespace two_lattice
        {

            /** TODO: Implement */
            template <class T> void stream_and_collide
            (
                const lbm::SimulationData<T> &simulation_data
            );

            /**
             * @brief This is the debug and development variant of the stream and collide method on the GPU.
             *        It performs the streaming and the collision step.
             *        Various debug comments are printed in the console.
             * 
             * @tparam T any child class of lbm::access::LBMAccessorObject
             * 
             * @param simulation_data   a structure containing the data on which the algorithm operates
             * @param sim_results       a structure containing the macroscopic simulation results produced by the algorithm
             * @param iteration         the current time step
             */
            template <class T> void stream_and_collide_debug
            (
                const lbm::SimulationData<T> &simulation_data,
                const lbm::SimulationResults &simulation_results,
                const lbm::Properties &properties,
                const unsigned int iteration
            )
            {
                static_assert(
                    std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                    "Template class must have base class lbm::access::LBMAccessorObject.");

                // Stream
                for(unsigned int node = 0; node <= simulation_data.end_node_index_buffered; ++node)
                {   
                    if(!lbm::is_ghost_node(node, properties, (*simulation_data.phase_information)[node]))
                    {
                        for (const auto direction : lbm::constants::all_directions)
                        {
                            (*simulation_data.distribution_values_1)[simulation_data.lbm_accessor->get_index(node, direction)] =
                                (*simulation_data.distribution_values_0)[
                                    simulation_data.lbm_accessor->get_index(
                                        lbm::access::get_neighbor(node, invert_direction(direction), simulation_data.lbm_accessor->horizontal_nodes), 
                                        direction)];
                        }
                    }
                }

                std::cout << "\033[33mDestination lattice after streaming: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";
                lbm::console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);

                // Calculate density and velocity
                for(unsigned int node = 0; node < simulation_data.end_node_index_buffered; ++node)
                {
                    if(!lbm::is_ghost_node(node, properties, (*simulation_data.phase_information)[node]))
                    {
                        unsigned int iteration_node_offset = 
                        lbm::access::results::get_result_index_no_ghosts(node, properties.horizontal_nodes, properties.domain_node_count, iteration);
                        std::vector<double> current_distributions = 
                            lbm::access::get_distribution_values_of<T>(*simulation_data.distribution_values_1, node, *simulation_data.lbm_accessor);
                        
                        // Density
                        (*simulation_results.densities)[iteration_node_offset] = lbm::macroscopic::density(current_distributions);

                        // Velocity
                        lbm::velocity v = lbm::macroscopic::flow_velocity(current_distributions); 
                        (*simulation_results.x_velocities)[iteration_node_offset] = v[0];
                        (*simulation_results.y_velocities)[iteration_node_offset] = v[1];
                    }
                }

                // std::cout << "Velocities: \n"
                //         << "-------------------------------------------------------------------------------\n";
                // lbm::console::print_velocities(properties, *simulation_results.x_velocities, *simulation_results.y_velocities, iteration);

                // std::cout << "Densities: \n"
                //         << "-------------------------------------------------------------------------------\n";
                // lbm::console::print_densities(properties, *simulation_results.densities, iteration);
                        
                // Collide
                for(unsigned int node = 0; node < simulation_data.end_node_index_buffered; ++node)
                {   
                    if(!lbm::is_ghost_node(node, properties, (*simulation_data.phase_information)[node]))
                    {
                        lbm::collision::collide_bgk<T>
                        (
                            simulation_data,
                            simulation_results,
                            properties.relaxation_time, 
                            lbm::access::results::get_result_index_no_ghosts(node, properties.horizontal_nodes, properties.domain_node_count, iteration),
                            node
                        );
                    }
                }

                std::cout << "\033[33mDestination lattice after collision: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";
                lbm::console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);
            }

            template <class T> void emplace_bounce_back
            (
                const Properties &properties,
                const SimulationData<T> &simulation_data
            )
            {
                static_assert(
                        std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                        "Template class must have base class lbm::access::LBMAccessorObject.");
                        
                for(int node = 0; node < properties.buffered_node_count; ++node)
                {
                    if((*simulation_data.phase_information)[node])
                    {
                        for(int dir = 0; dir < 9; ++dir)
                        {
                            (*simulation_data.distribution_values_0)[simulation_data.lbm_accessor->get_index(node, dir)] =
                            (*simulation_data.distribution_values_0)[simulation_data.lbm_accessor->get_index(lbm::access::get_neighbor(node, dir, simulation_data.lbm_accessor->horizontal_nodes), invert_direction(dir))];            
                        }
                    }
                }
            }

            /**
             * @brief Performs the GPU two-lattice algorithm with the specified properties.
             *        The macroscopic observables are stored in the `SimulationData` structure.
             *        This debug variant prints several pieces of information to the console.
             * 
             * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
             * 
             * @param[in]       properties       a structure containing all properties of the simulation
             * @param[in, out]  simulation_data  a structure containing the macroscopic observables
             * @param[in]       bsi              a lbm::border_swap_information required for the bounce-back part
             */
            template <class T> void run_debug
            (  
                const lbm::Properties &properties,
                const lbm::border_swap_information &bsi,
                SimulationData<T> &simulation_data
            )
            {
                static_assert(
                    std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                    "Template class must have base class lbm::access::LBMAccessorObject.");

                std::cout << "\033[33mNow running GPU two-lattice for " << properties.time_steps << " iterations.\033[0m\n\n";
                std::cout.flush();
                
                SimulationResults simulation_results(properties);

                std::vector<lbm::velocity> velocities(properties.non_buffered_node_count, {0,0});
                std::vector<double> densities(properties.non_buffered_node_count, 0);

                for(auto step = 0; step < properties.time_steps; ++step)
                {
                    std::cout << "\033[33m===============================================================================\n"
                            << "Iteration " << step << "\n"
                            << "===============================================================================\033[0m\n\n";

                    // Bounce-back
                    // lbm::bounce_back::emplace_bounce_back_values(bsi, *simulation_data.distribution_values_0, *simulation_data.lbm_accessor);
                    lbm::gpu::two_lattice::emplace_bounce_back(properties, simulation_data);
                    lbm::boundary_conditions::boundary_update(properties, *simulation_data.lbm_accessor, *simulation_data.distribution_values_0);

                    // std::cout << "\033[33mSource lattice after emplacing bounce-back values: \n"
                    //         << "-------------------------------------------------------------------------------\033[0m\n";
                    // lbm::console::print_distribution_values(*simulation_data.distribution_values_0, *simulation_data.lbm_accessor);

                    // Stream and collide
                    stream_and_collide_debug(simulation_data, simulation_results, properties, step);

                    // Inout update
                    // lbm::boundary_conditions::update_velocity_input_density_output(properties, *simulation_data.lbm_accessor, *simulation_data.distribution_values_1);
                    
                    // std::cout << "\033[33mDestination lattice after updating inlets and outlets: \n"
                    //         << "-------------------------------------------------------------------------------\033[0m\n";
                    // lbm::console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);

                    std::cout << "\033[33mFinished iteration " << step << "\033[0m \n\n\n";
                    // Swap source and destination lattice
                    simulation_data.distribution_values_1.swap(simulation_data.distribution_values_0);
                }

                std::cout << "\033[33mAll done, exiting simulation. \033[0m\n";
                lbm::console::print_simulation_results(properties, simulation_results);
            }

            namespace kernels
            {
                template <class LBMAccessor> class StreamKernel 
                {
                    static_assert(
                    std::is_base_of<lbm::access::LBMAccessorObject, LBMAccessor>::value, 
                    "Template class must have base class lbm::access::LBMAccessorObject.");

                    private:

                        static constexpr auto read_mode = sycl::access::mode::read;
                        static constexpr auto write_mode = sycl::access::mode::write;

                        sycl::accessor<uint8_t, 1, read_mode> phase_info_accessor;
                        sycl::accessor<double, 1, read_mode> source_accessor;
                        sycl::accessor<double, 1, write_mode> destination_accessor;
                        LBMAccessor lbm_accessor;
                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;

                    public:

                        StreamKernel
                        (
                            const sycl::accessor<uint8_t, 1, read_mode> &phase_info_accessor,
                            const sycl::accessor<double, 1, read_mode> &source_accessor,
                            const sycl::accessor<double, 1, write_mode> &destination_accessor,
                            const LBMAccessor &lbm_accessor,
                            const Properties &properties
                        )
                        : 
                        phase_info_accessor(phase_info_accessor), 
                        source_accessor(source_accessor),
                        destination_accessor(destination_accessor), 
                        lbm_accessor(lbm_accessor),
                        vertical_nodes(properties.vertical_nodes),
                        horizontal_nodes(properties.horizontal_nodes)
                        {std::cout << "Vertical x horizontal:" << vertical_nodes << " x " << horizontal_nodes << "\n"; }

                        void operator()(sycl::item<1> item) const
                        {
                            auto id = item.get_linear_id();

                            if(!lbm::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_info_accessor[id]))
                            {
                                for (const auto& direction : lbm::constants::all_directions)
                                {
                                    destination_accessor[lbm_accessor(id, direction)] =
                                        source_accessor[lbm_accessor(lbm::access::get_neighbor(id, invert_direction(direction), horizontal_nodes), direction)];
                                }
                            }
                        }
                };

                template <class LBMAccessor> class MacroscopicObservablesKernel
                {
                    static_assert(
                    std::is_base_of<lbm::access::LBMAccessorObject, LBMAccessor>::value, 
                    "Template class must have base class lbm::access::LBMAccessorObject.");

                    private:

                        static constexpr auto read_mode = sycl::access::mode::read;
                        static constexpr auto write_mode = sycl::access::mode::write;

                        sycl::accessor<uint8_t, 1, read_mode> phase_info_accessor;
                        sycl::accessor<double, 1, read_mode> destination_accessor;
                        sycl::accessor<double, 1, write_mode> densities_accessor;
                        sycl::accessor<double, 1, write_mode> x_velocities_accessor;
                        sycl::accessor<double, 1, write_mode> y_velocities_accessor;

                        LBMAccessor lbm_accessor;
                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;
                        unsigned int domain_node_count;
                        unsigned int iteration;

                    public:

                        MacroscopicObservablesKernel
                        (
                            const sycl::accessor<uint8_t, 1, read_mode> phase_info_accessor,
                            const sycl::accessor<double, 1, read_mode> destination_accessor,
                            const sycl::accessor<double, 1, write_mode> densities_accessor,
                            const sycl::accessor<double, 1, write_mode> x_velocities_accessor,
                            const sycl::accessor<double, 1, write_mode> y_velocities_accessor,
                            const LBMAccessor &lbm_accessor,
                            const Properties &properties,
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

                        void operator()(sycl::item<1> item) const
                        {
                            auto id = item.get_linear_id();

                            if(!lbm::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_info_accessor[id]))
                            {
                                unsigned int iteration_node_offset =
                                    lbm::access::results::get_result_index_no_ghosts(id, horizontal_nodes, domain_node_count, iteration);
                                double dist_vals[9];
                                double density = 0;

                                for (const auto& direction : lbm::constants::all_directions)
                                {
                                    dist_vals[direction] = destination_accessor[lbm_accessor(id, direction)];
                                    density += dist_vals[direction];
                                }
                                densities_accessor[iteration_node_offset] = density;
                                
                                sycl::vec<double,2> flow_velocity{0,0};

                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 
                                
                                for(const auto& i : lbm::constants::all_directions)
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

                template <class LBMAccessor> class CollideKernel
                {
                    static_assert(
                    std::is_base_of<lbm::access::LBMAccessorObject, LBMAccessor>::value, 
                    "Template class must have base class lbm::access::LBMAccessorObject.");

                    private:

                        static constexpr auto read_mode = sycl::access::mode::read;
                        static constexpr auto write_mode = sycl::access::mode::write;
                        static constexpr auto read_write = sycl::access::mode::read_write;

                        sycl::accessor<uint8_t, 1, read_mode> phase_info_accessor;
                        sycl::accessor<double, 1, read_write> destination_accessor;
                        sycl::accessor<double, 1, write_mode> densities_accessor;
                        sycl::accessor<double, 1, write_mode> x_velocities_accessor;
                        sycl::accessor<double, 1, write_mode> y_velocities_accessor;

                        LBMAccessor lbm_accessor;
                        unsigned int vertical_nodes;
                        unsigned int horizontal_nodes;
                        unsigned int domain_node_count;
                        unsigned int iteration;
                        double relaxation_time;

                    public:

                        CollideKernel
                        (
                            const sycl::accessor<uint8_t, 1, read_mode> phase_info_accessor,
                            const sycl::accessor<double, 1, read_write> destination_accessor,
                            const sycl::accessor<double, 1, write_mode> densities_accessor,
                            const sycl::accessor<double, 1, write_mode> x_velocities_accessor,
                            const sycl::accessor<double, 1, write_mode> y_velocities_accessor,
                            const LBMAccessor &lbm_accessor,
                            const Properties &properties,
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

                        void operator()(sycl::item<1> item) const
                        {
                            auto id = item.get_linear_id();

                            if(!lbm::is_ghost_node(id, vertical_nodes, horizontal_nodes, phase_info_accessor[id]))
                            {
                                unsigned int iteration_node_offset =
                                    lbm::access::results::get_result_index_no_ghosts(id, horizontal_nodes, domain_node_count, iteration);

                                double& x_velocity = x_velocities_accessor[iteration_node_offset];
                                double& y_velocity = y_velocities_accessor[iteration_node_offset];
                                double& density = densities_accessor[iteration_node_offset];

                                int velocity_x_component = 0; 
                                int velocity_y_component = 0; 

                                double value;
                                double result;

                                for (const auto& direction : lbm::constants::all_directions)
                                {
                                    value = destination_accessor[lbm_accessor.get_index(id, direction)];

                                    velocity_x_component = (direction % 3) - 1; 
                                    velocity_y_component = (direction / 3) - 1; 

                                    result = lbm::constants::weights[direction] *
                                        (
                                            density + 3 * (velocity_x_component * x_velocity + velocity_y_component * y_velocity)
                                            + 9.0/2 * std::pow(velocity_x_component * x_velocity + velocity_y_component * y_velocity, 2)
                                            - 3.0/2 * (x_velocity * x_velocity + y_velocity * y_velocity)
                                        );

                                    result = -(1/relaxation_time) * (value - result) + value;
                                    destination_accessor[lbm_accessor.get_index(id, direction)] = result;
                                }
                            }
                        }
                };
                
            } // ! namespace kernels


            /**
             * @brief This is the debug and development variant of the stream and collide method on the GPU.
             *        It performs the streaming and the collision step.
             *        Various debug comments are printed in the console.
             * 
             * @tparam T any child class of lbm::access::LBMAccessorObject
             * 
             * @param simulation_data   a structure containing the data on which the algorithm operates
             * @param sim_results       a structure containing the macroscopic simulation results produced by the algorithm
             * @param iteration         the current time step
             */
            template <class T> void stream_and_collide_debug_new
            (
                const lbm::SimulationData<T> &simulation_data,
                const lbm::SimulationResults &simulation_results,
                const lbm::Properties &properties,
                const unsigned int iteration
            )
            {
                static_assert(
                    std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                    "Template class must have base class lbm::access::LBMAccessorObject.");

                sycl::default_selector device_selector; 
                sycl::queue queue(device_selector);
                std::cout << "Running on "
                            << queue.get_device().get_info<sycl::info::device::name>()
                            << "\n";   

                {
                    sycl::buffer<uint8_t, 1> phase_info_sycl(simulation_data.phase_information->begin(), simulation_data.phase_information->end());
                    sycl::buffer<double, 1> src_sycl(simulation_data.distribution_values_0->data(), sycl::range<1>(simulation_data.distribution_values_0->size()));
                    sycl::buffer<double, 1> dst_sycl(simulation_data.distribution_values_1->data(), sycl::range<1>(simulation_data.distribution_values_1->size()));

                    queue.submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            sycl::accessor<uint8_t, 1, sycl::access::mode::read> phase_info_acc = phase_info_sycl.get_access<sycl::access::mode::read>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::read> src_acc = src_sycl.get_access<sycl::access::mode::read>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::write> dst_acc = dst_sycl.get_access<sycl::access::mode::write>(cgh);
                            
                            auto kernel = kernels::StreamKernel<T>
                            (
                                phase_info_acc,
                                src_acc,
                                dst_acc,
                                *simulation_data.lbm_accessor,
                                properties
                            );

                            cgh.parallel_for(sycl::range<1>(properties.vertical_nodes * properties.horizontal_nodes), kernel);
                        }
                    );
                }

                std::cout << "\033[33mDestination lattice after streaming: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";
                lbm::console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);

                {
                    sycl::buffer<uint8_t, 1> phase_info_sycl(simulation_data.phase_information->begin(), simulation_data.phase_information->end());
                    sycl::buffer<double, 1> dst_sycl(simulation_data.distribution_values_1->data(), sycl::range<1>(simulation_data.distribution_values_1->size()));
                    sycl::buffer<double, 1> densities_sycl(simulation_results.densities->data(), sycl::range<1>(simulation_results.densities->size()));
                    sycl::buffer<double, 1> x_velocities_sycl(simulation_results.x_velocities->data(), sycl::range<1>(simulation_results.x_velocities->size()));
                    sycl::buffer<double, 1> y_velocities_sycl(simulation_results.y_velocities->data(), sycl::range<1>(simulation_results.y_velocities->size()));

                    queue.submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            sycl::accessor<uint8_t, 1, sycl::access::mode::read> phase_info_acc = phase_info_sycl.get_access<sycl::access::mode::read>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::read> dst_acc = dst_sycl.get_access<sycl::access::mode::read>(cgh);

                            sycl::accessor<double, 1, sycl::access::mode::write> densities_acc = densities_sycl.get_access<sycl::access::mode::write>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::write> x_velocities_acc = x_velocities_sycl.get_access<sycl::access::mode::write>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::write> y_velocities_acc = y_velocities_sycl.get_access<sycl::access::mode::write>(cgh);
                            
                            auto kernel = kernels::MacroscopicObservablesKernel<T>
                            (
                                phase_info_acc,
                                dst_acc,
                                densities_acc,
                                x_velocities_acc,
                                y_velocities_acc,
                                *simulation_data.lbm_accessor,
                                properties,
                                iteration
                            );
                            cgh.parallel_for(sycl::range<1>(properties.vertical_nodes * properties.horizontal_nodes), kernel);
                        }
                    );
                }
            
                std::cout << "Velocities: \n"
                        << "-------------------------------------------------------------------------------\n";
                lbm::console::print_velocities(properties, *simulation_results.x_velocities, *simulation_results.y_velocities, iteration);

                std::cout << "Densities: \n"
                        << "-------------------------------------------------------------------------------\n";
                lbm::console::print_densities(properties, *simulation_results.densities, iteration);
                        
                // // Collide
                // for(unsigned int node = 0; node < simulation_data.end_node_index_buffered; ++node)
                // {   
                //     if(!lbm::is_ghost_node(node, properties, (*simulation_data.phase_information)[node]))
                //     {
                //         lbm::collision::collide_bgk<T>
                //         (
                //             simulation_data,
                //             simulation_results,
                //             properties.relaxation_time, 
                //             lbm::access::results::get_result_index_no_ghosts(node, properties.horizontal_nodes, properties.domain_node_count, iteration),
                //             node
                //         );
                //     }
                // }

                {
                    sycl::buffer<uint8_t, 1> phase_info_sycl(simulation_data.phase_information->begin(), simulation_data.phase_information->end());
                    sycl::buffer<double, 1> dst_sycl(simulation_data.distribution_values_1->data(), sycl::range<1>(simulation_data.distribution_values_1->size()));
                    sycl::buffer<double, 1> densities_sycl(simulation_results.densities->data(), sycl::range<1>(simulation_results.densities->size()));
                    sycl::buffer<double, 1> x_velocities_sycl(simulation_results.x_velocities->data(), sycl::range<1>(simulation_results.x_velocities->size()));
                    sycl::buffer<double, 1> y_velocities_sycl(simulation_results.y_velocities->data(), sycl::range<1>(simulation_results.y_velocities->size()));

                    queue.submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            sycl::accessor<uint8_t, 1, sycl::access::mode::read> phase_info_acc = phase_info_sycl.get_access<sycl::access::mode::read>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::read_write> dst_acc = dst_sycl.get_access<sycl::access::mode::read_write>(cgh);

                            sycl::accessor<double, 1, sycl::access::mode::write> densities_acc = densities_sycl.get_access<sycl::access::mode::write>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::write> x_velocities_acc = x_velocities_sycl.get_access<sycl::access::mode::write>(cgh);
                            sycl::accessor<double, 1, sycl::access::mode::write> y_velocities_acc = y_velocities_sycl.get_access<sycl::access::mode::write>(cgh);
                            
                            auto kernel = kernels::CollideKernel<T>
                            (
                                phase_info_acc,
                                dst_acc,
                                densities_acc,
                                x_velocities_acc,
                                y_velocities_acc,
                                *simulation_data.lbm_accessor,
                                properties,
                                iteration
                            );
                            cgh.parallel_for(sycl::range<1>(properties.vertical_nodes * properties.horizontal_nodes), kernel);
                        }
                    );
                }

                std::cout << "\033[33mDestination lattice after collision: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";
                lbm::console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);
            }

            /**
             * @brief Performs the GPU two-lattice algorithm with the specified properties.
             *        The macroscopic observables are stored in the `SimulationData` structure.
             *        This debug variant prints several pieces of information to the console.
             * 
             * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
             * 
             * @param[in]       properties       a structure containing all properties of the simulation
             * @param[in, out]  simulation_data  a structure containing the macroscopic observables
             * @param[in]       bsi              a lbm::border_swap_information required for the bounce-back part
             */
            template <class T> void run_debug_new
            (  
                const lbm::Properties &properties,
                const lbm::border_swap_information &bsi,
                SimulationData<T> &simulation_data
            )
            {
                static_assert(
                    std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                    "Template class must have base class lbm::access::LBMAccessorObject.");

                std::cout << "\033[33mNow running GPU two-lattice for " << properties.time_steps << " iterations.\033[0m\n\n";
                std::cout.flush();
                
                SimulationResults simulation_results(properties);

                std::vector<lbm::velocity> velocities(properties.non_buffered_node_count, {0,0});
                std::vector<double> densities(properties.non_buffered_node_count, 0);

                for(auto step = 0; step < properties.time_steps; ++step)
                {
                    std::cout << "\033[33m===============================================================================\n"
                            << "Iteration " << step << "\n"
                            << "===============================================================================\033[0m\n\n";

                    // Bounce-back
                    // lbm::bounce_back::emplace_bounce_back_values(bsi, *simulation_data.distribution_values_0, *simulation_data.lbm_accessor);
                    lbm::gpu::two_lattice::emplace_bounce_back(properties, simulation_data);
                    lbm::boundary_conditions::boundary_update(properties, *simulation_data.lbm_accessor, *simulation_data.distribution_values_0);

                    // std::cout << "\033[33mSource lattice after emplacing bounce-back values: \n"
                    //         << "-------------------------------------------------------------------------------\033[0m\n";
                    // lbm::console::print_distribution_values(*simulation_data.distribution_values_0, *simulation_data.lbm_accessor);

                    // Stream and collide
                    stream_and_collide_debug_new(simulation_data, simulation_results, properties, step);

                    // Inout update
                    // lbm::boundary_conditions::update_velocity_input_density_output(properties, *simulation_data.lbm_accessor, *simulation_data.distribution_values_1);
                    
                    // std::cout << "\033[33mDestination lattice after updating inlets and outlets: \n"
                    //         << "-------------------------------------------------------------------------------\033[0m\n";
                    // lbm::console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);

                    std::cout << "\033[33mFinished iteration " << step << "\033[0m \n\n\n";
                    // Swap source and destination lattice
                    simulation_data.distribution_values_1.swap(simulation_data.distribution_values_0);
                }

                std::cout << "\033[33mAll done, exiting simulation. \033[0m\n";
                lbm::console::print_simulation_results(properties, simulation_results);
            }

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! GPU_TWO_LATTICE_HPP
