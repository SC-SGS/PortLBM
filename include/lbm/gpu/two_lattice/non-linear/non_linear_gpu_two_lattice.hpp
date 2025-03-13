/**
 * @file        non_linear_gpu_two_lattice.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header, two classes for the GPU-based linear two-lattice algorithm are declared. Both inherit 
 *              from the `lbm::execution::SYCLAlgorithm` class which defines the interface of all algorithms. The 
 *              kernel functions are implemented in `linear_two_lattice_kernels.hpp`.
 * 
 * @version     4.3
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef NON_LINEAR_GPU_TWO_LATTICE_HPP
#define NON_LINEAR_GPU_TWO_LATTICE_HPP

// Console output
#include "../../../console/console_output.hpp"

// LBM core features
#include "../../../core/domain_initialization.hpp"

// Algorithm
#include "../../../execution/sycl_algorithm.hpp"

// SYCL constants
#include "../../sycl_constants.hpp"

// Kernels
#include "../../general/non_buffered.hpp"
#include "non_linear_two_lattice_kernels.hpp"

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
             * @brief This namespace contains implementations using non-linear kernels.
             */
            namespace non_linear
            {

// Performance non-linear two-lattice /////////////////////////////////////////////////////////////////////////////////

                /**
                 * @brief Class representation of the linear SYCL implementation of the two-lattice algorithm.
                 * 
                 * @tparam A any `core::access::AccessorConcept` from access.hpp
                 */
                template <core::access::AccessorConcept A>
                class NonLinearGpuTwoLattice : public execution::SYCLAlgorithm
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
                                auto kernel = non_linear::kernels::StreamCollideKernel<A>(*simulation);
                                cgh.parallel_for(
                                    sycl::nd_range<2>(
                                        sycl::range<2>(
                                            simulation->domain->vertical_nodes - 2,
                                            simulation->domain->horizontal_nodes - 2
                                        ),
                                        sycl::range<2>(
                                            simulation->domain->subdomain_vertical_nodes,
                                            simulation->domain->subdomain_horizontal_nodes
                                        )
                                    ), 
                                    kernel
                                );
                            }
                        );
                        event.wait();
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
                                    sycl::nd_range<2>(
                                        sycl::range<2>(
                                            simulation->domain->vertical_nodes - 2,
                                            simulation->domain->horizontal_nodes - 2
                                        ),
                                        sycl::range<2>(
                                            simulation->domain->subdomain_vertical_nodes,
                                            simulation->domain->subdomain_horizontal_nodes
                                        )
                                    ), 
                                    kernel
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
                                auto kernel = general::non_buffered::OutletUpdateKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); 
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

                                    real_type *tmp = simulation->data->distribution_values_1;
                                    simulation->data->distribution_values_1 = simulation->data->distribution_values_0;
                                    simulation->data->distribution_values_0 = tmp;

                                    simulation->control->finalize_iteration();
                                }
                            }
                        );
                    }

                    explicit NonLinearGpuTwoLattice(sycl::queue &queue) : SYCLAlgorithm(queue) 
                    {
                        core::domain_initialization::setup_domain<A, core::access::decomposed::NonBufferedNodeAccess>(
                            *simulation, queue
                        );
                    }; 
                };

// Debug non-linear two-lattice ///////////////////////////////////////////////////////////////////////////////////////

                /**
                 * @brief Class representation of the linear SYCL debug implementation of the two-lattice algorithm.
                 * 
                 * @tparam A any `core::access::AccessorConcept` from access.hpp
                 */
                template <core::access::AccessorConcept A>
                class NonLinearGpuTwoLatticeDebug : public execution::SYCLAlgorithm
                {
                    private:

                    std::unique_ptr<std::vector<real_type>> all_densities;
                    std::unique_ptr<std::vector<real_type>> all_x_velocities;
                    std::unique_ptr<std::vector<real_type>> all_y_velocities;
                    std::unique_ptr<std::vector<real_type>> distribution_values;
                    std::unique_ptr<std::vector<real_type>> temp_macroscopic_observables;
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
                                auto kernel = non_linear::kernels::StreamKernel<A>(*simulation);
                                cgh.parallel_for(
                                    sycl::nd_range<2>(
                                        sycl::range<2>(
                                            simulation->domain->vertical_nodes - 2,
                                            simulation->domain->horizontal_nodes - 2
                                        ),
                                        sycl::range<2>(
                                            simulation->domain->subdomain_vertical_nodes,
                                            simulation->domain->subdomain_horizontal_nodes
                                        )
                                    ), 
                                    kernel
                                );
                            }
                        ).wait();
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
                                auto kernel = non_linear::kernels::MacroscopicObservablesKernel<A>(*simulation);
                                cgh.parallel_for(
                                    sycl::nd_range<2>(
                                        sycl::range<2>(
                                            simulation->domain->vertical_nodes - 2,
                                            simulation->domain->horizontal_nodes - 2
                                        ),
                                        sycl::range<2>(
                                            simulation->domain->subdomain_vertical_nodes,
                                            simulation->domain->subdomain_horizontal_nodes
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
                                auto kernel = non_linear::kernels::CollideKernel<A>(*simulation);
                                cgh.parallel_for(
                                    sycl::nd_range<2>(
                                        sycl::range<2>(
                                            simulation->domain->vertical_nodes - 2,
                                            simulation->domain->horizontal_nodes - 2
                                        ),
                                        sycl::range<2>(
                                            simulation->domain->subdomain_vertical_nodes,
                                            simulation->domain->subdomain_horizontal_nodes
                                        )
                                    ), 
                                    kernel
                                );
                            }
                        ).wait();
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
                            9 * simulation->domain->total_node_count
                        ).wait();

                        std::cout 
                        << "\033[36mDestination lattice after streaming: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";
                        
                        console::print_distribution_values<A>(
                            *distribution_values, 
                            *phase_information,
                            *simulation
                        );
                    
                        update_macroscopic_observables();

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
                            9 * simulation->domain->total_node_count
                        ).wait();

                        std::cout 
                        << "\033[36mDestination lattice after collision: \n"
                        << "-------------------------------------------------------------------------------\n\033[0m";

                        console::print_distribution_values<A>(
                            *distribution_values, 
                            *phase_information,
                            *simulation
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
                                cgh.parallel_for(
                                    sycl::nd_range<2>(
                                        sycl::range<2>(
                                            simulation->domain->vertical_nodes - 2,
                                            simulation->domain->horizontal_nodes - 2
                                        ),
                                        sycl::range<2>(
                                            simulation->domain->subdomain_vertical_nodes,
                                            simulation->domain->subdomain_horizontal_nodes
                                        )
                                    ), 
                                    kernel
                                );
                            }
                        );
                        event.wait();

                        queue->copy(
                            simulation->data->distribution_values_0, 
                            distribution_values->data(), 
                            9 * simulation->domain->total_node_count
                        ).wait();

                        std::cout << "\033[36mSource lattice after emplacing bounce-back values: \n"
                                << "-------------------------------------------------------------------------------\033[0m\n";

                        console::print_distribution_values<A>(
                            *distribution_values, 
                            *phase_information,
                            *simulation
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
                                auto kernel = general::non_buffered::OutletUpdateKernel<A>(*simulation);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel); 
                            }
                        );
                        event.wait();

                        queue->copy(
                            simulation->data->distribution_values_0, 
                            distribution_values->data(), 
                            9 * simulation->domain->total_node_count
                        ).wait();

                        std::cout 
                        << "\033[36mDestination lattice after updating inlets and outlets: \n"
                        << "-------------------------------------------------------------------------------\033[0m\n";

                        console::print_distribution_values<A>(
                            *distribution_values, 
                            *phase_information,
                            *simulation
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
                            simulation->data->phase_information, 
                            phase_information->data(), 
                            simulation->domain->total_node_count
                        ).wait();

                        console::print_phase_vector(*phase_information, simulation->domain->horizontal_nodes);

                        std::cout 
                        << "\033[36mNow running GPU two-lattice for " 
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
                        std::cout << "------------------------------------------------------------------------------------\n";
                        std::cout << "\texpanded_node_count = " << simulation->domain->total_node_count << "\n";
                        std::cout << "\texpanded_horizontal_nodes = " << simulation->domain->horizontal_nodes << "\n";
                        std::cout << "\texpanded_vertical_nodes = " << simulation->domain->vertical_nodes << "\n";

                        std::cout << "\tsubdomain_width = " << simulation->domain->subdomain_horizontal_nodes << "\n";
                        std::cout << "\tsubdomain_height = " << simulation->domain->subdomain_vertical_nodes << "\n";

                        std::cout << "\tsubdomain_count_vertical = " << simulation->domain->subdomain_count_vertical << "\n";
                        std::cout << "\tsubdomain_count_horizontal = " << simulation->domain->subdomain_count_horizontal << "\n";
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

                                    real_type *tmp = simulation->data->distribution_values_1;
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

                    /**
                     * @brief   Constructs a new `NonLinearGpuTwoLatticeDebug` algorithm object and initializes its 
                     *          domain.
                     * 
                     * @param[in]   queue   the SYCL queue that is used to allocate the device data
                     */
                    explicit NonLinearGpuTwoLatticeDebug(sycl::queue &queue) 
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
                        core::domain_initialization::setup_domain<A, core::access::decomposed::NonBufferedNodeAccess>(
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
                    }; 
                };

            } // ! namespace linear

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! GPU_TWO_LATTICE_HPP
