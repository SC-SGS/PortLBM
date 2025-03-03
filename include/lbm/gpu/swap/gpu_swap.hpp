/**
 * @file        gpu_swap.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header, a debug and a performance class for the GPU-based swap algorithm are declared. Both 
 *              inherit from the `lbm::execution::SYCLAlgorithm` class which defines the interface of all algorithms. 
 *              The kernel functions are implemented in `swap_kernels.hpp`.
 * 
 * @version     1.2
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_GPU_SWAP_HPP
#define LBM_GPU_SWAP_HPP

// Console output
#include "../../console/console_output.hpp"

// LBM core features
#include "../../core/domain_initialization.hpp"

// Algorithm
#include "../../execution/sycl_algorithm.hpp"

// SYCL constants
#include "../sycl_constants.hpp"

// Kernels
#include "../general/buffered.hpp"
#include "swap_kernels.hpp"

namespace lbm
{

    /**
     * @brief This namespace contains the GPU implementations of the swap and the swap algorithm.
     */
    namespace gpu
    {

        /**
         * @brief This namespace contains the GPU implementation of the swap algorithm.
         */
        namespace swap
        {

// Performance swap ///////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief   Class representation of the SYCL implementation of the swap algorithm.
             * 
             * @tparam  A   any `core::access::AccessorConcept` from access.hpp
             */
            template <core::access::AccessorConcept A>
            class GpuSwap : public execution::SYCLAlgorithm
            {
                private:

                real_type inlet_values [9];

                /**
                 * @brief   Performs the combined streaming and collision step of the swap algorithm and updates the 
                 *          macroscopic observables in the process.
                 */
                inline void stream_and_collide()
                {
                    queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = 
                                swap::kernels::StreamCollideKernel<A>(
                                    *simulation, 
                                    cgh, 
                                    simulation->properties->work_group_size
                                );

                            cgh.parallel_for(
                                sycl::nd_range<2>(
                                    sycl::range<2>(
                                        simulation->domain->vertical_nodes - 1,
                                        simulation->domain->subdomain_count_horizontal * 
                                        (simulation->domain->subdomain_horizontal_nodes + 2)
                                    ),
                                    sycl::range<2>(
                                        simulation->domain->subdomain_vertical_nodes + 1,
                                        simulation->domain->subdomain_horizontal_nodes + 2
                                    )
                                ), 
                                kernel
                            );
                        }
                    ).wait();

                    if(!(simulation->control->get_current_iteration() % simulation->properties->frame_update_interval))
                    {
                        queue->copy(
                            simulation->results->densities_gpu, 
                            simulation->results->densities_cpu->data(), 
                            simulation->results->densities_cpu->size()
                        );
                        queue->copy(
                            simulation->results->x_velocities_gpu, 
                            simulation->results->x_velocities_cpu->data(), 
                            simulation->results->x_velocities_cpu->size()
                        );
                        queue->copy(
                            simulation->results->y_velocities_gpu, 
                            simulation->results->y_velocities_cpu->data(), 
                            simulation->results->y_velocities_cpu->size()
                        );
                        queue->copy(
                            simulation->results->absolute_velocities_gpu, 
                            simulation->results->absolute_velocities_cpu->data(), 
                            simulation->results->absolute_velocities_cpu->size()
                        );
                    }
                }

                /**
                 * @brief   Emplaces the distribution values as preparation for the bounce-back scheme for the swap 
                 *          algorithm.
                 */
                void emplace_bounce_back()
                {
                    auto event1 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::HorizontalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->subdomain_count_vertical - 1,
                                    simulation->domain->horizontal_nodes
                                ), kernel
                            );
                        }
                    );
                    event1.wait();

                    auto event2 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::VerticalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->vertical_nodes,
                                    simulation->domain->subdomain_count_horizontal - 1
                                ), kernel
                            );
                        }
                    );
                    event2.wait();

                    auto event3 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = kernels::EmplaceBounceBackKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::nd_range<2>(
                                    sycl::range<2>(
                                        simulation->domain->subdomain_count_vertical * 
                                        simulation->domain->subdomain_vertical_nodes, 
                                        simulation->domain->subdomain_count_horizontal * 
                                        simulation->domain->subdomain_horizontal_nodes
                                    ),
                                    sycl::range<2>(
                                        simulation->domain->subdomain_vertical_nodes, 
                                        simulation->domain->subdomain_horizontal_nodes
                                    )
                                ), kernel
                            );
                        }
                    );
                    event3.wait();
                }

                /**
                 * @brief   Updates the inlets and outlets as preparation for the swap algorithm.
                 */
                void perform_inout_update()
                {
                    auto inlet_event = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::InletUpdateKernel<A>(*simulation, inlet_values);
                            cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 2), kernel); 
                        }
                    );
                    inlet_event.wait();

                    auto event1 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::HorizontalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->subdomain_count_vertical - 1,
                                    simulation->domain->horizontal_nodes
                                ), kernel
                            );
                        }
                    );
                    event1.wait();

                    auto event2 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::VerticalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->vertical_nodes,
                                    simulation->domain->subdomain_count_horizontal - 1
                                ), kernel
                            );
                        }
                    );
                    event2.wait();

                    auto outlet_event = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::OutletUpdateKernel<A>(*simulation);
                            cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes), kernel); 
                        }
                    );
                    outlet_event.wait();
                }

                public:

                /**
                 * @brief   Runs the swap algorithm until it is paused or reaches the last iteration. 
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

                                simulation->control->finalize_iteration();
                            }
                        }
                    );
                }

                /**
                 * @brief   Constructs a new `GpuSwap` algorithm object and initializes its domain.
                 * 
                 * @param[in]   queue   the SYCL queue that is used to allocate the device data
                 */
                explicit GpuSwap(sycl::queue &queue) : SYCLAlgorithm(queue) 
                {
                    std::vector<real_type> distribution = core::maxwell_boltzmann_distribution(
                        simulation->properties->inlet_velocity_x,
                        simulation->properties->inlet_velocity_y,
                        simulation->properties->inlet_density
                    );

                    memcpy(inlet_values, distribution.data(), 9 * sizeof(real_type));

                    core::domain_initialization::setup_domain<A, core::access::decomposed::BufferedNodeAccess>(
                        *simulation, queue
                    );
                }; 
            };

// Debug swap /////////////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief Class representation of the SYCL debug implementation of the swap algorithm.
             * 
             * @tparam A any `core::access::AccessorConcept` from access.hpp
             */
            template <core::access::AccessorConcept A>
            class GpuSwapDebug : public execution::SYCLAlgorithm
            {
                private:

                std::unique_ptr<std::vector<real_type>> all_densities;
                std::unique_ptr<std::vector<real_type>> all_x_velocities;
                std::unique_ptr<std::vector<real_type>> all_y_velocities;
                std::unique_ptr<std::vector<real_type>> distribution_values;
                std::unique_ptr<std::vector<real_type>> temp_macroscopic_observables;
                std::unique_ptr<std::vector<int8_t>> phase_information;

                real_type inlet_values [9];

                unsigned int current_iteration;

                /**
                 * @brief   Performs the streaming and collision step of the swap algorithm and updates the macroscopic
                 *          observables in the process.
                 */
                inline void stream_and_collide()
                {
                    queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = swap::kernels::StreamCollideKernel<A>(*simulation, cgh, 16);
                            cgh.parallel_for(
                                sycl::nd_range<2>(
                                    sycl::range<2>(
                                        simulation->domain->vertical_nodes - 1,
                                        simulation->domain->subdomain_count_horizontal * 
                                        (simulation->domain->subdomain_horizontal_nodes + 2)
                                    ),
                                    sycl::range<2>(
                                        simulation->domain->subdomain_vertical_nodes + 1,
                                        simulation->domain->subdomain_horizontal_nodes + 2
                                    )
                                ), 
                                kernel
                            );
                        }
                    ).wait();

                    queue->copy(
                        simulation->results->densities_gpu, 
                        simulation->results->densities_cpu->data(), 
                        simulation->results->densities_cpu->size()
                    ).wait();
                    queue->copy(
                        simulation->results->x_velocities_gpu, 
                        simulation->results->x_velocities_cpu->data(), 
                        simulation->results->x_velocities_cpu->size()
                    ).wait();
                    queue->copy(
                        simulation->results->y_velocities_gpu, 
                        simulation->results->y_velocities_cpu->data(), 
                        simulation->results->y_velocities_cpu->size()
                    ).wait();
                    queue->copy(
                        simulation->results->absolute_velocities_gpu, 
                        simulation->results->absolute_velocities_cpu->data(), 
                        simulation->results->absolute_velocities_cpu->size()
                    ).wait();
                    queue->copy(
                        simulation->data->distribution_values_0, 
                        distribution_values->data(), 
                        9 * simulation->domain->total_node_count
                    ).wait();

                    std::cout 
                    << "\033[36mLattice after combined stream and collide: \n"
                    << "-------------------------------------------------------------------------------\n\033[0m";
                    
                    lbm::console::buffered::print_distribution_values<A>(
                        *distribution_values, 
                        *phase_information,
                        *simulation
                    );

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
                }

                /**
                 * @brief   Emplaces the values as preparation for the bounce-back scheme for the swap algorithm.
                 */
                void emplace_bounce_back()
                {
                    auto event1 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::HorizontalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->subdomain_count_vertical - 1,
                                    simulation->domain->horizontal_nodes
                                ), kernel
                            );
                        }
                    );
                    event1.wait();

                    auto event2 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::VerticalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->vertical_nodes,
                                    simulation->domain->subdomain_count_horizontal - 1
                                ), kernel
                            );
                        }
                    );
                    event2.wait();

                    queue->copy(
                        simulation->data->distribution_values_0, 
                        distribution_values->data(), 
                        9 * simulation->domain->total_node_count
                    ).wait();

                    std::cout 
                    << "\033[36mLattice after performing copy to buffers: \n"
                    << "-------------------------------------------------------------------------------\033[0m\n";

                    lbm::console::buffered::print_distribution_values<A>(
                        *distribution_values, 
                        *phase_information,
                        *simulation
                    );

                    auto event3 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = kernels::EmplaceBounceBackKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::nd_range<2>(
                                    sycl::range<2>(
                                        simulation->domain->subdomain_count_vertical * 
                                        simulation->domain->subdomain_vertical_nodes, 
                                        simulation->domain->subdomain_count_horizontal * 
                                        simulation->domain->subdomain_horizontal_nodes
                                    ),
                                    sycl::range<2>(
                                        simulation->domain->subdomain_vertical_nodes, 
                                        simulation->domain->subdomain_horizontal_nodes
                                    )
                                ), kernel
                            );
                        }
                    );
                    event3.wait();

                    queue->copy(
                        simulation->data->distribution_values_0, 
                        distribution_values->data(), 
                        9 * simulation->domain->total_node_count
                    ).wait();

                    std::cout 
                    << "\033[36mLattice after emplacing bounce-back values: \n"
                    << "-------------------------------------------------------------------------------\033[0m\n";

                    lbm::console::buffered::print_distribution_values<A>(
                        *distribution_values, 
                        *phase_information,
                        *simulation
                    );

                    auto event4 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::HorizontalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->subdomain_count_vertical - 1,
                                    simulation->domain->horizontal_nodes
                                ), kernel
                            );
                        }
                    );
                    event4.wait();

                    auto event5 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::VerticalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->vertical_nodes,
                                    simulation->domain->subdomain_count_horizontal - 1
                                ), kernel
                            );
                        }
                    );
                    event5.wait();

                    queue->copy(
                        simulation->data->distribution_values_0, 
                        distribution_values->data(), 
                        9 * simulation->domain->total_node_count
                    ).wait();

                    std::cout 
                    << "\033[36mLattice after performing copy to buffers: \n"
                    << "-------------------------------------------------------------------------------\033[0m\n";

                    lbm::console::buffered::print_distribution_values<A>(
                        *distribution_values, 
                        *phase_information,
                        *simulation
                    );
                }

                /**
                 * @brief   Updates the inlets and outlets as preparation for the swap algorithm.
                 */
                void perform_inout_update()
                {
                    auto inlet_event = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::InletUpdateKernel<A>(*simulation, inlet_values);
                            cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 2), kernel); 
                        }
                    );
                    
                    inlet_event.wait();
                    auto event1 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::HorizontalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->subdomain_count_vertical - 1,
                                    simulation->domain->horizontal_nodes
                                ), kernel
                            );
                        }
                    );
                    event1.wait();

                    auto event2 = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::VerticalCopyToBufferKernel<A>(*simulation);
                            cgh.parallel_for(
                                sycl::range<2>(
                                    simulation->domain->vertical_nodes,
                                    simulation->domain->subdomain_count_horizontal - 1
                                ), kernel
                            );
                        }
                    );
                    event2.wait();

                    auto outlet_event = queue->submit
                    (
                        [&](sycl::handler &cgh)
                        {
                            auto kernel = general::buffered::OutletUpdateKernel<A>(*simulation);
                            cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); 
                        }
                    );
                    outlet_event.wait();

                    queue->copy(
                        simulation->data->distribution_values_0, 
                        distribution_values->data(), 
                        9 * simulation->domain->total_node_count
                    ).wait();

                    std::cout 
                    << "\033[36mDestination lattice after updating inlets and outlets: \n"
                    << "-------------------------------------------------------------------------------\033[0m\n";

                    console::buffered::print_distribution_values<A>(
                        *distribution_values, 
                        *phase_information,
                        *simulation
                    );
                }

                public:

                /**
                 * @brief   Runs the debug variant of the swap algorithm until it is paused or it reaches the last 
                 *          iteration. 
                 */
                inline void execute() override 
                { 
                    console::print_ansi_color_message();
                    console::print_color_legend();

                    auto event1 = queue->copy(
                        simulation->data->distribution_values_0, 
                        distribution_values->data(), 
                        9 * simulation->domain->total_node_count
                    );

                    auto event2 = queue->copy(
                        simulation->data->phase_information, 
                        phase_information->data(), 
                        simulation->domain->total_node_count
                    );

                    event2.wait();
                    event1.wait();

                    console::buffered::print_distribution_values<A>(
                        *distribution_values, 
                        *phase_information,
                        *simulation
                    );

                    console::print_phase_vector(*phase_information, simulation->domain->horizontal_nodes);

                    std::cout 
                    << "\033[36mNow running GPU swap for " 
                    << simulation->properties->time_steps 
                    << " iterations.\033[0m\n\n";

                    std::cout 
                    << "Running on " 
                    << queue->get_device().get_info<sycl::info::device::name>() 
                    << "\n\n";

                    fmt::print
                    (
                        "Simulation properties:\n"
                        "-------------------------------------------------------------------------------\n"
                    );

                    fmt::print(fmt::runtime(simulation->properties->to_string()));   

                    std::cout << "\n";
                    std::cout << "Buffered domain:\n";
                    std::cout 
                    << "------------------------------------------------------------------------------------\n";
                    std::cout << "\texpanded_node_count = " << simulation->domain->total_node_count << "\n";
                    std::cout << "\texpanded_horizontal_nodes = " << simulation->domain->horizontal_nodes << "\n";
                    std::cout << "\texpanded_vertical_nodes = " << simulation->domain->vertical_nodes << "\n";

                    std::cout << "\tsubdomain_width = " << simulation->domain->subdomain_horizontal_nodes << "\n";
                    std::cout << "\tsubdomain_height = " << simulation->domain->subdomain_vertical_nodes << "\n";

                    std::cout 
                    << "\tsubdomain_count_vertical = " << simulation->domain->subdomain_count_vertical << "\n";
                    std::cout 
                    << "\tsubdomain_count_horizontal = " << simulation->domain->subdomain_count_horizontal << "\n";
                    std::cout << "\n";

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
                                << simulation->control->get_last_frametime() 
                                << " milliseconds.\033[0m\n\n";

                                current_iteration++;

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

                /**
                 * @brief   Constructs a new `GpuSwapDebug` algorithm object and initializes its domain.
                 * 
                 * @param[in]   queue   the SYCL queue that is used to allocate the device data
                 */
                explicit GpuSwapDebug(sycl::queue &queue) 
                : 
                SYCLAlgorithm(queue),
                all_densities(std::make_unique<std::vector<real_type>>()), 
                all_x_velocities(std::make_unique<std::vector<real_type>>()), 
                all_y_velocities(std::make_unique<std::vector<real_type>>()), 
                distribution_values(
                    std::make_unique<std::vector<real_type>>(9 * simulation->domain->total_node_count, 0)
                ),
                temp_macroscopic_observables(std::make_unique<std::vector<real_type>>(simulation->properties->domain_node_count, 0)),
                phase_information(std::make_unique<std::vector<int8_t>>(simulation->domain->total_node_count, 0)),
                current_iteration(0)
                {
                    core::domain_initialization::setup_domain<A, core::access::decomposed::BufferedNodeAccess>(
                        *simulation, queue
                    );

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

                    std::vector<real_type> distribution = core::maxwell_boltzmann_distribution(
                                simulation->properties->inlet_velocity_x,
                                simulation->properties->inlet_velocity_y,
                                simulation->properties->inlet_density
                            );

                    for(int i = 0; i < 9; ++i) inlet_values[i] = distribution[i];
                }; 
            };

        } // ! namespace swap

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! LBM_GPU_SWAP_HPP
