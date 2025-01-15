/**
 * @file        linear_gpu_two_lattice.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header, classes for the GPU-based linear two-lattice algorithm is declared.
 *              Both inherit from the `lbm::execution::Algorithm` class which defines the interface 
 *              of all algorithms. The kernel functions are implemented in 
 *              `linear_two_lattice_kernels.hpp`.
 * 
 * @version     4.1
 * 
 * @date        January 2025
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

// Performance linear two-lattice /////////////////////////////////////////////////////////////////////////////////////

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
                                cgh.parallel_for(
                                    sycl::range<1>(
                                        simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes
                                    ), kernel
                                );
                            }
                        );
                        event.wait();
                        if(!(simulation->control->get_current_iteration() % 4))
                        {
                            queue->copy(
                                simulation->results->densities_gpu, 
                                simulation->results->densities_cpu->data(), 
                                simulation->results->densities_cpu->size()
                            );//.wait();
                            queue->copy(
                                simulation->results->x_velocities_gpu, 
                                simulation->results->x_velocities_cpu->data(), 
                                simulation->results->x_velocities_cpu->size()
                            );//.wait();
                            queue->copy(
                                simulation->results->y_velocities_gpu, 
                                simulation->results->y_velocities_cpu->data(), 
                                simulation->results->y_velocities_cpu->size()
                            );//.wait();
                            queue->copy(
                                simulation->results->absolute_velocities_gpu, 
                                simulation->results->absolute_velocities_cpu->data(), 
                                simulation->results->absolute_velocities_cpu->size()
                            );//.wait();
                        }
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
                                auto kernel = kernels::EmplaceBounceBackKernel<A>(*simulation);
                                cgh.parallel_for(
                                    sycl::range<1>(
                                        simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes
                                    ), kernel
                                );
                            }
                        );
                        event.wait();
                    }

                    /**
                     * @brief   Updates the inlets and outlets as preparation for the two-lattice algorithm.
                     */
                    void perform_inout_update()
                    {
                        auto event = queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = kernels::InoutUpdateKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); 
                                // set to simulation->properties->vertical_nodes - 2 to also treat corners
                            }
                        );
                        event.wait();
                    }

                    public:

                    /**
                     * @brief   Runs the linear two-lattice algorithm until it is paused or it reaches the last iteration. 
                     */
                    inline void execute() override
                    {
                        future = hpx::async
                        (
                            [&]
                            {
                                while(simulation->control->is_execution_allowed())
                                {
                                    simulation->control->reset_timer();

                                    emplace_bounce_back();
                                    perform_inout_update();
                                    stream_and_collide();

                                    double *tmp = simulation->data->distribution_values_1;
                                    simulation->data->distribution_values_1 = simulation->data->distribution_values_0;
                                    simulation->data->distribution_values_0 = tmp;

                                    simulation->control->finalize_iteration();
                                }
                            }
                        );
                    }

                    explicit LinearGpuTwoLattice(sycl::queue &queue) : Algorithm(queue) {}; 
                };

// Debug linear two-lattice ///////////////////////////////////////////////////////////////////////////////////////////

                /**
                 * @brief Class representation of the linear SYCL debug implementation of the two-lattice algorithm.
                 * 
                 * @tparam A any `core::access::AccessorConcept` from access.hpp
                 */
                template <core::access::AccessorConcept A>
                class LinearGpuTwoLatticeDebug : public execution::Algorithm
                {
                    private:

                    std::unique_ptr<std::vector<double>> all_densities;
                    std::unique_ptr<std::vector<double>> all_x_velocities;
                    std::unique_ptr<std::vector<double>> all_y_velocities;
                    std::unique_ptr<std::vector<double>> distribution_values;
                    std::unique_ptr<std::vector<double>> temp_macroscopic_observables;
                    std::unique_ptr<std::vector<int8_t>> phase_information;

                    unsigned int current_iteration;

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
                        stream();
                        queue->copy(
                            simulation->data->distribution_values_1, 
                            distribution_values->data(), 
                            9 * simulation->properties->buffered_node_count
                        ).wait();

                        std::cout 
                        << "\033[36mDestination lattice after streaming: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";
                        
                        lbm::console::print_distribution_values<A>(
                            *distribution_values, 
                            simulation->properties->horizontal_nodes, 
                            simulation->properties->vertical_nodes
                        );
                    
                        update_macroscopic_observables();
                        queue->copy(
                            simulation->results->x_velocities_gpu, 
                            simulation->results->x_velocities_cpu->data(), 
                            simulation->properties->domain_node_count
                        ).wait();

                        queue->copy(
                            simulation->results->y_velocities_gpu, 
                            simulation->results->y_velocities_cpu->data(), 
                            simulation->properties->domain_node_count
                        ).wait();

                        queue->copy(
                            simulation->results->absolute_velocities_gpu, 
                            simulation->results->absolute_velocities_cpu->data(), 
                            simulation->properties->domain_node_count
                        ).wait();

                        queue->copy(
                            simulation->results->densities_gpu, 
                            simulation->results->densities_cpu->data(), 
                            simulation->properties->domain_node_count
                        ).wait();

                        std::cout 
                        << "Velocities: \n"
                        << "-------------------------------------------------------------------------------\n";

                        lbm::console::print_velocities(
                            *simulation->properties, 
                            *simulation->results->x_velocities_cpu, 
                            *simulation->results->y_velocities_cpu, 
                            0
                        );

                        std::cout << "Densities: \n"
                                << "-------------------------------------------------------------------------------\n";
                        lbm::console::print_densities(
                            *simulation->properties, 
                            *simulation->results->densities_cpu, 
                            0
                        );

                        collide();

                        queue->copy(
                            simulation->data->distribution_values_1, 
                            distribution_values->data(), 
                            9 * simulation->properties->buffered_node_count
                        ).wait();

                        std::cout 
                        << "\033[36mDestination lattice after collision: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";

                        lbm::console::print_distribution_values<A>(
                            *distribution_values,
                            simulation->properties->horizontal_nodes, 
                            simulation->properties->vertical_nodes
                        );    
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
                                auto kernel = kernels::EmplaceBounceBackKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->buffered_node_count), kernel);
                            }
                        );
                        event.wait();

                        queue->copy(
                            simulation->data->distribution_values_0, 
                            distribution_values->data(), 
                            9 * simulation->properties->buffered_node_count
                        ).wait();

                        std::cout << "\033[36mSource lattice after emplacing bounce-back values: \n"
                                << "-------------------------------------------------------------------------------\033[0m\n";

                        lbm::console::print_distribution_values<A>(
                            *distribution_values, 
                            simulation->properties->horizontal_nodes, 
                            simulation->properties->vertical_nodes
                        );
                    }

                    /**
                     * @brief   Updates the inlets and outlets as preparation for the two-lattice algorithm.
                     */
                    void perform_inout_update()
                    {
                        auto event = queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                auto kernel = kernels::InoutUpdateKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); 
                                // set to simulation->properties->vertical_nodes - 2 to also treat corners
                            }
                        );
                        event.wait();

                        queue->copy(
                            simulation->data->distribution_values_0, 
                            distribution_values->data(), 
                            9 * simulation->properties->buffered_node_count
                        ).wait();

                        std::cout 
                        << "\033[36mDestination lattice after updating inlets and outlets: \n"
                        << "-------------------------------------------------------------------------------\033[0m\n";

                        lbm::console::print_distribution_values<A>(
                            *distribution_values, 
                            simulation->properties->horizontal_nodes, 
                            simulation->properties->vertical_nodes
                        );
                    }

                    public:

                    /**
                     * @brief   Runs the debug variant of the linear two-lattice algorithm until it is paused or it 
                     *          reaches the last iteration. 
                     */
                    inline void execute() override 
                    { 
                        console::print_ansi_color_message();
                        console::print_color_legend();

                        queue->copy(
                            simulation->data->distribution_values_0, 
                            distribution_values->data(), 
                            9 * simulation->properties->buffered_node_count
                        ).wait();

                        console::print_distribution_values<A>(
                            *distribution_values, 
                            simulation->properties->horizontal_nodes, 
                            simulation->properties->vertical_nodes
                        );

                        queue->copy(
                            simulation->data->phase_information, 
                            phase_information->data(), 
                            simulation->properties->buffered_node_count
                        ).wait();

                        console::print_phase_vector(*phase_information, simulation->properties->horizontal_nodes);

                        std::cout 
                        << "\033[36mNow running GPU two-lattice for " 
                        << simulation->properties->time_steps 
                        << " iterations.\033[0m\n\n";

                        std::cout 
                        << "Running on " 
                        << queue->get_device().get_info<sycl::info::device::name>() 
                        << "\n\n";   

                        future = hpx::async
                        (
                            [&]
                            { 
                                while(simulation->control->is_execution_allowed())
                                {
                                    simulation->control->reset_timer();

                                    std::cout 
                                    << "\033[36m===============================================================================\n"
                                    << "Iteration " << current_iteration << "\n"
                                    << "===============================================================================\033[0m\n\n";

                                    emplace_bounce_back();
                                    perform_inout_update();
                                    stream_and_collide();

                                    std::cout 
                                    << "\033[36mFinished iteration " 
                                    << current_iteration 
                                    << " after "
                                    << simulation->control->get_last_frame_time() 
                                    << " milliseconds.\033[0m\n\n";

                                    double *tmp = simulation->data->distribution_values_1;
                                    simulation->data->distribution_values_1 = simulation->data->distribution_values_0;
                                    simulation->data->distribution_values_0 = tmp;

                                    queue->copy(
                                        simulation->results->x_velocities_gpu, 
                                        temp_macroscopic_observables->data(), 
                                        simulation->properties->domain_node_count
                                    ).wait();

                                    all_x_velocities->insert(
                                        all_x_velocities->end(), 
                                        temp_macroscopic_observables->begin(), 
                                        temp_macroscopic_observables->end()
                                    );

                                    queue->copy(
                                        simulation->results->y_velocities_gpu, 
                                        temp_macroscopic_observables->data(), 
                                        simulation->properties->domain_node_count
                                    ).wait();

                                    all_y_velocities->insert(
                                        all_y_velocities->end(), 
                                        temp_macroscopic_observables->begin(), 
                                        temp_macroscopic_observables->end()
                                    );

                                    queue->copy(
                                        simulation->results->densities_gpu, 
                                        temp_macroscopic_observables->data(), 
                                        simulation->properties->domain_node_count
                                    ).wait();

                                    all_densities->insert(
                                        all_densities->end(), 
                                        temp_macroscopic_observables->begin(), 
                                        temp_macroscopic_observables->end()
                                    );

                                    simulation->control->finalize_iteration();
                                }

                                std::cout << "\033[36mAll done, exiting simulation. \033[0m\n\n";

                                lbm::console::print_simulation_results(
                                    *simulation->properties, 
                                    *all_densities, 
                                    *all_x_velocities, 
                                    *all_y_velocities
                                );
                            }
                        );
                    }

                    explicit LinearGpuTwoLatticeDebug(sycl::queue &queue) 
                    : 
                    Algorithm(queue),
                    all_densities(std::make_unique<std::vector<double>>()), 
                    all_x_velocities(std::make_unique<std::vector<double>>()), 
                    all_y_velocities(std::make_unique<std::vector<double>>()), 
                    distribution_values(
                        std::make_unique<std::vector<double>>(9 * simulation->properties->buffered_node_count, 0)
                    ),
                    temp_macroscopic_observables(std::make_unique<std::vector<double>>(simulation->properties->domain_node_count, 0)),
                    phase_information(std::make_unique<std::vector<int8_t>>(simulation->properties->buffered_node_count, 0)),
                    current_iteration(0)
                    {
                        all_densities->reserve(
                            simulation->properties->time_steps * simulation->properties->domain_node_count
                        );
                        all_densities->shrink_to_fit();

                        all_x_velocities->reserve(
                            simulation->properties->time_steps * simulation->properties->domain_node_count
                        );
                        all_x_velocities->shrink_to_fit();

                        all_y_velocities->reserve(
                            simulation->properties->time_steps * simulation->properties->domain_node_count
                        );
                        all_y_velocities->shrink_to_fit();

                        distribution_values->shrink_to_fit();
                        temp_macroscopic_observables->shrink_to_fit();
                        phase_information->shrink_to_fit();
                    }; 
                };

            } // ! namespace linear

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! GPU_TWO_LATTICE_HPP
