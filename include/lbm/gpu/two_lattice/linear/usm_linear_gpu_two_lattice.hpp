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
 * @version     4.0
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024-2025
 * 
 */

#ifndef LINEAR_GPU_TWO_LATTICE_HPP
#define LINEAR_GPU_TWO_LATTICE_HPP

// Console output
#include "../../../console/console_output.hpp"

// LBM core features
#include "../../../core/access.hpp"
#include "../../../core/domain_initialization.hpp"
#include "../../../core/simulation.hpp"

// Algorithm
#include "../../../execution/algorithm.hpp"

// SYCL constants
#include "../../sycl_constants.hpp"

// Kernels
#include "usm_linear_two_lattice_kernels.hpp"

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
                 * @tparam A any `core::access::AccessorConcept` from access.hpp
                 */
                template <core::access::AccessorConcept A>
                class LinearGpuTwoLattice : public execution::Algorithm
                {
                    private:

                    /**
                     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                     *          the macroscopic observables in the process.
                     */
                    inline void stream_and_collide()
                    {
                        auto event = queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = linear::kernels::StreamCollideKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                        event.wait();
                        queue->copy(simulation->results->densities_gpu, simulation->results->densities_cpu->data(), simulation->results->densities_cpu->size()).wait();
                        queue->copy(simulation->results->x_velocities_gpu, simulation->results->x_velocities_cpu->data(), simulation->results->x_velocities_cpu->size()).wait();
                        queue->copy(simulation->results->y_velocities_gpu, simulation->results->y_velocities_cpu->data(), simulation->results->y_velocities_cpu->size()).wait();
                        queue->copy(simulation->results->absolute_velocities_gpu, simulation->results->absolute_velocities_cpu->data(), simulation->results->absolute_velocities_cpu->size()).wait();
                    }

                    /**
                     * @brief   Emplaces the values as preparation for the bounce-back scheme for the two-lattice algorithm.
                     */
                    void emplace_bounce_back()
                    {
                        auto event = queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = kernels::boundaries::EmplaceBounceBackKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                        event.wait();
                    }

                    /**
                     * @brief   Updates the inlets and outlets as preparation for the two-laatice algorithm.
                     */
                    void perform_inout_update()
                    {
                        auto event = queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = kernels::boundaries::InoutUpdateKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); // set to simulation->properties->vertical_nodes - 2 to also treat corners
                            }
                        );
                        event.wait();
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

                                double *tmp = simulation->data->distribution_values_1;
                                simulation->data->distribution_values_1 = simulation->data->distribution_values_0;
                                simulation->data->distribution_values_0 = tmp;
                            }
                        );
                    }

                    /**
                     * @brief   Performs the specified amount of iterations of the linear two-lattice algorithm. 
                     */
                    inline void execute(unsigned int time_steps) override
                    {
                        std::cout << "Executing USM linear two-lattice algorithm\n";
                        future = hpx::async
                        (
                            [&]
                            {
                                for(unsigned int time_step = 0; time_step < time_steps; ++time_step)
                                {
                                    emplace_bounce_back();
                                    perform_inout_update();
                                    stream_and_collide();
                                    double *tmp = simulation->data->distribution_values_1;
                                    simulation->data->distribution_values_1 = simulation->data->distribution_values_0;
                                    simulation->data->distribution_values_0 = tmp;
                                }
                            }
                        );
                    }

                    explicit LinearGpuTwoLattice(sycl::queue &queue) : Algorithm(queue) {}; 
                };

                /**
                 * @brief Class representation of the linear SYCL debug implementation of the two-lattice algorithm.
                 * 
                 * @tparam A any `core::access::AccessorConcept` from access.hpp
                 */
                template <core::access::AccessorConcept A>
                class LinearGpuTwoLatticeDebug : public execution::Algorithm
                {
                    private:

                    /**
                     * @brief   Performs the streaming step of the two-lattice algorithm.
                     */
                    void stream()
                    {
                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = linear::kernels::StreamKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->buffered_node_count), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                     *          the macroscopic observables in the process.
                     */
                    void update_macroscopic_observables()
                    {
                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = linear::kernels::MacroscopicObservablesKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->buffered_node_count), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Performs the collision step of the two-lattice algorithm.
                     * 
                     */
                    void collide()
                    {
                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = linear::kernels::CollideKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->buffered_node_count), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                     *          the macroscopic observables in the process.
                     */
                    inline void stream_and_collide()
                    {
                        std::vector<double> dist_vals(9 * simulation->properties->buffered_node_count, 0);

                        stream();
                        queue->copy(simulation->data->distribution_values_1, dist_vals.data(), 9 * simulation->properties->buffered_node_count).wait();

                        std::cout << "\033[36mDestination lattice after streaming: \n"
                                  << "-------------------------------------------------------------------------------\n\033[0m";
                        
                        lbm::console::print_distribution_values<A>(dist_vals, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);
                    
                        update_macroscopic_observables();
                        queue->copy(simulation->results->x_velocities_gpu, simulation->results->x_velocities_cpu->data(), simulation->properties->domain_node_count).wait();
                        queue->copy(simulation->results->y_velocities_gpu, simulation->results->y_velocities_cpu->data(), simulation->properties->domain_node_count).wait();
                        queue->copy(simulation->results->absolute_velocities_gpu, simulation->results->absolute_velocities_cpu->data(), simulation->properties->domain_node_count).wait();
                        queue->copy(simulation->results->densities_gpu, simulation->results->densities_cpu->data(), simulation->properties->domain_node_count).wait();

                        std::cout << "Velocities: \n"
                                << "-------------------------------------------------------------------------------\n";
                        lbm::console::print_velocities(*simulation->properties, *simulation->results->x_velocities_cpu, *simulation->results->y_velocities_cpu, 0);

                        std::cout << "Densities: \n"
                                << "-------------------------------------------------------------------------------\n";
                        lbm::console::print_densities(*simulation->properties, *simulation->results->densities_cpu, 0);

                        collide();
                        queue->copy(simulation->data->distribution_values_1, dist_vals.data(), 9 * simulation->properties->buffered_node_count).wait();

                        std::cout << "\033[36mDestination lattice after collision: \n"
                                << "-------------------------------------------------------------------------------\n\033[0m";
                        lbm::console::print_distribution_values<A>(dist_vals, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);    
                    }

                    /**
                     * @brief   Emplaces the values as preparation for the bounce-back scheme for the two-lattice algorithm.
                     */
                    void emplace_bounce_back()
                    {
                        auto event = queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = kernels::boundaries::EmplaceBounceBackKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(6/*simulation->properties->buffered_node_count*/), kernel);
                            }
                        );
                        event.wait();
                    }

                    /**
                     * @brief   Updates the inlets and outlets as preparation for the two-laatice algorithm.
                     */
                    void perform_inout_update()
                    {
                        auto event = queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = kernels::boundaries::InoutUpdateKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); // set to simulation->properties->vertical_nodes - 2 to also treat corners
                            }
                        );
                        event.wait();
                    }

                    /**
                     * @brief   Runs the two-lattice algorithm with the specified simulation->properties->
                     *          It operates on the provided data.
                     *          This debug variant prints several pieces of information to the console.
                     */
                    void run()
                    {
                        std::cout << "\033[36mNow running GPU two-lattice for " << simulation->properties->time_steps << " iterations.\033[0m\n\n";
                        std::vector<double> densities;
                        std::vector<double> x_velocities;
                        std::vector<double> y_velocities;

                        densities.reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
                        x_velocities.reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
                        y_velocities.reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);

                        std::vector<double> dist_vals(9 * simulation->properties->buffered_node_count, 0);

                        std::cout << "Running on " << queue->get_device().get_info<sycl::info::device::name>() << "\n";   

                        for(auto step = 0; step < simulation->properties->time_steps; ++step)
                        {
                            std::cout << "\033[36m===============================================================================\n"
                                    << "Iteration " << step << "\n"
                                    << "===============================================================================\033[0m\n\n";

                            emplace_bounce_back();
                            queue->copy(simulation->data->distribution_values_0, dist_vals.data(), 9 * simulation->properties->buffered_node_count).wait();

                            std::cout << "\033[36mSource lattice after emplacing bounce-back values: \n"
                                    << "-------------------------------------------------------------------------------\033[0m\n";
                            lbm::console::print_distribution_values<A>(dist_vals, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);

                            perform_inout_update();
                            queue->copy(simulation->data->distribution_values_0, dist_vals.data(), 9 * simulation->properties->buffered_node_count).wait();

                            std::cout << "\033[36mDestination lattice after updating inlets and outlets: \n"
                                    << "-------------------------------------------------------------------------------\033[0m\n";
                            lbm::console::print_distribution_values<A>(dist_vals, simulation->properties->horizontal_nodes, simulation->properties->vertical_nodes);

                            stream_and_collide();

                            std::cout << "\033[36mFinished iteration " << step << "\033[0m \n\n\n";

                            // Swap source and destination lattice
                            double *tmp = simulation->data->distribution_values_1;
                            simulation->data->distribution_values_1 = simulation->data->distribution_values_0;
                            simulation->data->distribution_values_0 = tmp;

                            std::vector<double> temp(simulation->properties->domain_node_count, 0);
                            queue->copy(simulation->results->x_velocities_gpu, temp.data(), simulation->properties->domain_node_count).wait();
                            x_velocities.insert(x_velocities.end(), temp.begin(), temp.end());
                            queue->copy(simulation->results->y_velocities_gpu, temp.data(), simulation->properties->domain_node_count).wait();
                            y_velocities.insert(y_velocities.end(), temp.begin(), temp.end());
                            queue->copy(simulation->results->densities_gpu, temp.data(), simulation->properties->domain_node_count).wait();
                            densities.insert(densities.end(), temp.begin(), temp.end());
                        }

                        std::cout << "\033[36mAll done, exiting simulation. \033[0m\n";
                        lbm::console::print_simulation_results(*simulation->properties, densities, x_velocities, y_velocities);
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

                    explicit LinearGpuTwoLatticeDebug(sycl::queue &queue) : Algorithm(queue) {}; 
                };

            } // ! namespace linear

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! GPU_TWO_LATTICE_HPP
