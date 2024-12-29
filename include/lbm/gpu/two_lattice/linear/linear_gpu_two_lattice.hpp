/**
 * @file        linear_gpu_two_lattice.hpp
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
 * @version     3.0
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LINEAR_GPU_TWO_LATTICE_HPP
#define LINEAR_GPU_TWO_LATTICE_HPP

// Console output
#include "../../../console/console_output.hpp"

// LBM core features
#include "../../../core/access.hpp"
#include "../../../core/boundaries.hpp"
#include "../../../core/domain_initialization.hpp"
#include "../../../core/simulation.hpp"

// Algorithm
#include "../../../execution/algorithm.hpp"

// SYCL constants
#include "../../sycl_constants.hpp"

// Kernels
#include "linear_two_lattice_kernels.hpp"

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

            /**
             * @brief This namespace contains implementations using linear kernels.
             */
            namespace linear
            {

                /**
                 * @brief Class representation of the linear SYCL implementation of the two-lattice algorithm.
                 * 
                 * @tparam A any `core::access::experimental::AccessorConcept` from access.hpp
                 */
                template <core::access::experimental::AccessorConcept A>
                class LinearGpuTwoLattice : public execution::Algorithm
                {
                    private:

                    /**
                     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                     *          the macroscopic observables in the process.
                     */
                    inline void stream_and_collide()
                    {
                        {
                            sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                            sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));
                            sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));
                            sycl::buffer<double, 1> densities_sycl(simulation->results->densities->data(), sycl::range<1>(simulation->results->densities->size()));
                            sycl::buffer<double, 1> x_velocities_sycl(simulation->results->x_velocities->data(), sycl::range<1>(simulation->results->x_velocities->size()));
                            sycl::buffer<double, 1> y_velocities_sycl(simulation->results->y_velocities->data(), sycl::range<1>(simulation->results->y_velocities->size()));

                            queue->submit
                            (
                                [&](sycl::handler &cgh)
                                {
                                    sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                    sycl::accessor<double, 1, constants::read> src_acc = src_sycl.get_access<constants::read>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> dst_acc = dst_sycl.get_access<constants::read_write>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> densities_acc = densities_sycl.get_access<constants::read_write>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> x_velocities_acc = x_velocities_sycl.get_access<constants::read_write>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> y_velocities_acc = y_velocities_sycl.get_access<constants::read_write>(cgh);
                                    
                                    auto kernel = linear::kernels::StreamCollideKernel<A>
                                    (
                                        phase_info_acc,
                                        src_acc,
                                        dst_acc,
                                        densities_acc,
                                        x_velocities_acc,
                                        y_velocities_acc,
                                        *simulation->properties
                                    );
                                    cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                                }
                            );
                        }
                    }

                    /**
                     * @brief   Emplaces the values as preparation for the bounce-back scheme for the two-lattice algorithm.
                     */
                    void emplace_bounce_back()
                    {
                        sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->data(), simulation->data->phase_information->size());
                        sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                sycl::accessor<double, 1, constants::read_write> src_acc = src_sycl.get_access<constants::read_write>(cgh);
                                
                                auto kernel = kernels::boundaries::EmplaceBounceBackKernel<A>(
                                    phase_info_acc, src_acc, simulation->properties->horizontal_nodes, simulation->properties->buffered_node_count);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Updates the inlets and outlets as preparation for the two-laatice algorithm.
                     */
                    void perform_inout_update()
                    {
                        sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));
 
                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<double, 1, constants::read_write> src_acc = src_sycl.get_access<constants::read_write>(cgh);
                                auto kernel = kernels::boundaries::InoutUpdateKernel<A>(src_acc, *simulation->properties);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); // set to simulation->properties->vertical_nodes - 2 to also treat corners
                            }
                        );
                    }

                    public:

                    /**
                     * @brief   Performs one iteration of the linear two-lattice algorithm. 
                     */
                    inline void execute() override
                    {
                        future = hpx::async
                        (
                            [&]
                            {
                                emplace_bounce_back();
                                perform_inout_update();
                                stream_and_collide();
                                simulation->data->distribution_values_1.swap(simulation->data->distribution_values_0);
                            }
                        );
                    }

                    /**
                     * @brief   Performs the specified amount of iterations of the linear two-lattice algorithm. 
                     */
                    inline void execute(unsigned int time_steps) override
                    {
                        future = hpx::async
                        (
                            [&]
                            {
                                for(unsigned int time_step = 0; time_step < time_steps; ++time_step)
                                {
                                    emplace_bounce_back();
                                    perform_inout_update();
                                    stream_and_collide();
                                    simulation->data->distribution_values_1.swap(simulation->data->distribution_values_0);
                                }
                            }
                        );
                    }

                    explicit LinearGpuTwoLattice(const sycl::queue &queue) : Algorithm(queue) {}; 
                };

                /**
                 * @brief Class representation of the linear SYCL debug implementation of the two-lattice algorithm.
                 * 
                 * @tparam A any `core::access::experimental::AccessorConcept` from access.hpp
                 */
                template <core::access::experimental::AccessorConcept A>
                class LinearGpuTwoLatticeDebug : public execution::Algorithm
                {
                    private:

                    /**
                     * @brief   Performs the streaming step of the two-lattice algorithm.
                     */
                    void stream()
                    {
                        sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                        sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));
                        sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                sycl::accessor<double, 1, constants::read> src_acc = src_sycl.get_access<constants::read>(cgh);
                                sycl::accessor<double, 1, constants::write> dst_acc = dst_sycl.get_access<constants::write>(cgh);
                                
                                auto kernel = linear::kernels::StreamKernel<A>
                                (
                                    phase_info_acc,
                                    src_acc,
                                    dst_acc,
                                    *simulation->properties
                                );

                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                     *          the macroscopic observables in the process.
                     */
                    void update_macroscopic_observables()
                    {
                        sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                        sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));
                        sycl::buffer<double, 1> densities_sycl(simulation->results->densities->data(), sycl::range<1>(simulation->results->densities->size()));
                        sycl::buffer<double, 1> x_velocities_sycl(simulation->results->x_velocities->data(), sycl::range<1>(simulation->results->x_velocities->size()));
                        sycl::buffer<double, 1> y_velocities_sycl(simulation->results->y_velocities->data(), sycl::range<1>(simulation->results->y_velocities->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                sycl::accessor<double, 1, constants::read> dst_acc = dst_sycl.get_access<constants::read>(cgh);

                                sycl::accessor<double, 1, constants::write> densities_acc = densities_sycl.get_access<constants::write>(cgh);
                                sycl::accessor<double, 1, constants::write> x_velocities_acc = x_velocities_sycl.get_access<constants::write>(cgh);
                                sycl::accessor<double, 1, constants::write> y_velocities_acc = y_velocities_sycl.get_access<constants::write>(cgh);
                                
                                auto kernel = linear::kernels::MacroscopicObservablesKernel<A>
                                (
                                    phase_info_acc,
                                    dst_acc,
                                    densities_acc,
                                    x_velocities_acc,
                                    y_velocities_acc,
                                    *simulation->properties
                                );
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Performs the collision step of the two-lattice algorithm.
                     * 
                     */
                    void collide()
                    {
                        sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                        sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));
                        sycl::buffer<double, 1> densities_sycl(simulation->results->densities->data(), sycl::range<1>(simulation->results->densities->size()));
                        sycl::buffer<double, 1> x_velocities_sycl(simulation->results->x_velocities->data(), sycl::range<1>(simulation->results->x_velocities->size()));
                        sycl::buffer<double, 1> y_velocities_sycl(simulation->results->y_velocities->data(), sycl::range<1>(simulation->results->y_velocities->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                sycl::accessor<double, 1, constants::read_write> dst_acc = dst_sycl.get_access<constants::read_write>(cgh);

                                sycl::accessor<double, 1, constants::write> densities_acc = densities_sycl.get_access<constants::write>(cgh);
                                sycl::accessor<double, 1, constants::write> x_velocities_acc = x_velocities_sycl.get_access<constants::write>(cgh);
                                sycl::accessor<double, 1, constants::write> y_velocities_acc = y_velocities_sycl.get_access<constants::write>(cgh);
                                
                                auto kernel = linear::kernels::CollideKernel<A>
                                (
                                    phase_info_acc,
                                    dst_acc,
                                    densities_acc,
                                    x_velocities_acc,
                                    y_velocities_acc,
                                    *simulation->properties
                                );
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                     *          the macroscopic observables in the process.
                     */
                    inline void stream_and_collide()
                    {
                        stream();

                        std::cout << "\033[36mDestination lattice after streaming: \n"
                                << "-------------------------------------------------------------------------------\n\033[0m";
                        lbm::console::print_distribution_values<A>(*simulation->data->distribution_values_1, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);
                    
                        update_macroscopic_observables();

                        std::cout << "Velocities: \n"
                                << "-------------------------------------------------------------------------------\n";
                        lbm::console::print_velocities(*simulation->properties, *simulation->results->x_velocities, *simulation->results->y_velocities, 0);

                        std::cout << "Densities: \n"
                                << "-------------------------------------------------------------------------------\n";
                        lbm::console::print_densities(*simulation->properties, *simulation->results->densities, 0);

                        // for(int i = 0; i < simulation->results->densities->size(); i++)
                        // {
                        //     std::cout << (*simulation->results->densities)[i] << ", ";
                        // }
                        // std::cout << "\n";

                        collide();

                        std::cout << "\033[36mDestination lattice after collision: \n"
                                << "-------------------------------------------------------------------------------\n\033[0m";
                        lbm::console::print_distribution_values<A>(*simulation->data->distribution_values_1, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);    
                    }

                    /**
                     * @brief   Emplaces the values as preparation for the bounce-back scheme for the two-lattice algorithm.
                     */
                    void emplace_bounce_back()
                    {
                        sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->data(), simulation->data->phase_information->size());
                        sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                sycl::accessor<double, 1, constants::read_write> src_acc = src_sycl.get_access<constants::read_write>(cgh);
                                
                                auto kernel = kernels::boundaries::EmplaceBounceBackKernel<A>(
                                    phase_info_acc, src_acc, simulation->properties->horizontal_nodes, simulation->properties->buffered_node_count);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Updates the inlets and outlets as preparation for the two-laatice algorithm.
                     */
                    void perform_inout_update()
                    {
                        sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<double, 1, constants::read_write> src_acc = src_sycl.get_access<constants::read_write>(cgh);
                                auto kernel = kernels::boundaries::InoutUpdateKernel<A>(src_acc, *simulation->properties);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); // set to simulation->properties->vertical_nodes - 2 to also treat corners
                            }
                        );

                        // double densities [2] = {simulation->properties->inlet_density, simulation->properties->outlet_density};
                        // kernels::boundaries::inout_update_debugger<A>(densities, *simulation->properties, *simulation->data->distribution_values_0);
                    }

                    /**
                     * @brief   Runs the two-lattice algorithm with the specified simulation->properties->
                     *          It operates on the provided data.
                     *          This debug variant prints several pieces of information to the console.
                     */
                    void run()
                    {
                        std::cout << "\033[36mNow running GPU two-lattice for " << simulation->properties->time_steps << " iterations.\033[0m\n\n";
                        lbm::core::Results all_simulation_results({}, {}, {}, {}, {});

                        all_simulation_results.densities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
                        all_simulation_results.x_velocities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
                        all_simulation_results.y_velocities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);

                        sycl::default_selector device_selector; 
                        sycl::queue queue(device_selector);

                        std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << "\n";   

                        for(auto step = 0; step < simulation->properties->time_steps; ++step)
                        {
                            std::cout << "\033[36m===============================================================================\n"
                                    << "Iteration " << step << "\n"
                                    << "===============================================================================\033[0m\n\n";

                            emplace_bounce_back();

                            std::cout << "\033[36mSource lattice after emplacing bounce-back values: \n"
                                    << "-------------------------------------------------------------------------------\033[0m\n";
                            lbm::console::print_distribution_values<A>(*simulation->data->distribution_values_0, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);

                            perform_inout_update();

                            std::cout << "\033[36mDestination lattice after updating inlets and outlets: \n"
                                    << "-------------------------------------------------------------------------------\033[0m\n";
                            lbm::console::print_distribution_values<A>(*simulation->data->distribution_values_0, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);

                            stream_and_collide();

                            std::cout << "\033[36mFinished iteration " << step << "\033[0m \n\n\n";

                            // Swap source and destination lattice
                            simulation->data->distribution_values_1.swap(simulation->data->distribution_values_0);

                            all_simulation_results.densities->insert(all_simulation_results.densities->end(), simulation->results->densities->begin(), simulation->results->densities->end());
                            all_simulation_results.x_velocities->insert(all_simulation_results.x_velocities->end(), simulation->results->x_velocities->begin(), simulation->results->x_velocities->end());
                            all_simulation_results.y_velocities->insert(all_simulation_results.y_velocities->end(), simulation->results->y_velocities->begin(), simulation->results->y_velocities->end());
                        }

                        std::cout << "\033[36mAll done, exiting simulation. \033[0m\n";
                        lbm::console::print_simulation_results(*simulation->properties, all_simulation_results);
                    }

                    public:

                    /**
                     * @brief   Performs one iteration of the linear two-lattice algorithm. 
                     */
                    inline void execute() override { future = hpx::async([&]{ run(); });}

                    /**
                     * @brief   Performs the specified amount of iterations of the linear two-lattice algorithm. 
                     */
                    inline void execute(unsigned int time_steps) override { future = hpx::async([&]{ run(); });}

                    explicit LinearGpuTwoLatticeDebug(const sycl::queue &queue) : Algorithm(queue) {}; 
                };

            } // ! namespace linear

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! GPU_TWO_LATTICE_HPP
