/**
 * @file        buffered_gpu_two_lattice.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       In this header, classes for the GPU-based buffered two-lattice algorithm are declared. Both inherit
 *              from the `lbm::execution::SYCLAlgorithm` class which defines the interface of all algorithms.
 *              The kernel functions are implemented in `buffered_two_lattice_kernels.hpp`.
 *
 * @version     1.4
 *
 * @date        April 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#ifndef BUFFERED_GPU_TWO_LATTICE_HPP
#define BUFFERED_GPU_TWO_LATTICE_HPP

// Console output
#include "../../../console/console_output.hpp"

// LBM core features
#include "../../../core/access.hpp"
#include "../../../core/domain_initialization.hpp"
#include "../../../core/simulation.hpp"

// Algorithm
#include "../../../execution/sycl_algorithm.hpp"

// SYCL constants
#include "../../sycl_constants.hpp"

// Kernels
#include "../../general/buffered.hpp"
#include "buffered_two_lattice_kernels.hpp"

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
 * @brief This namespace contains the implementation of the buffered two-lattice algorithms.
 */
namespace buffered
{

// PERFORMANCE BUFFERED TWO-LATTICE ///////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Class representation of the buffered SYCL implementation of the two-lattice algorithm.
 *
 * @tparam A any `core::access::AccessorConcept` from access.hpp
 */
template <core::access::AccessorConcept A>
class BufferedGpuTwoLattice : public execution::SYCLAlgorithm
{
  private:
    /**
     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
     *          the macroscopic observables in the process.
     */
    inline void stream_and_collide()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = buffered::kernels::StreamCollideKernel<A>(*simulation);
                    cgh.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>(simulation->domain->subdomain_vertical_nodes
                                                             * simulation->domain->subdomain_count_vertical,
                                                         simulation->domain->subdomain_horizontal_nodes
                                                             * simulation->domain->subdomain_count_horizontal),
                                          sycl::range<2>(simulation->domain->subdomain_vertical_nodes,
                                                         simulation->domain->subdomain_horizontal_nodes)),
                        kernel);
                })
            .wait();

        if (simulation->control->get_current_iteration() % simulation->properties->frame_update_interval
            == simulation->properties->frame_update_interval - 1)
        {
            copy_macroscopic_observables_to_cpu();
        }
    }

    /**
     * @brief   Updates the horizontal buffers by copying adjacent values above and below.
     */
    void update_horizontal_buffers()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = general::buffered::HorizontalCopyToBufferKernel<A>(*simulation);
                    cgh.parallel_for(sycl::range<2>(simulation->domain->subdomain_count_vertical - 1,
                                                    simulation->domain->horizontal_nodes),
                                     kernel);
                })
            .wait();
    }

    /**
     * @brief   Updates the vertical buffers by copying adjacent values left and right.
     *
     */
    void update_vertical_buffers()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = general::buffered::VerticalCopyToBufferKernel<A>(*simulation);
                    cgh.parallel_for(sycl::range<2>(simulation->domain->vertical_nodes,
                                                    simulation->domain->subdomain_count_horizontal - 1),
                                     kernel);
                })
            .wait();
    }

    /**
     * @brief   Emplaces the values as preparation for the bounce-back scheme for the buffered
     *          two-lattice algorithm.
     */
    void emplace_bounce_back()
    {
        if (simulation->domain->subdomain_count_vertical > 1)
        {
            update_horizontal_buffers();
        }

        if (simulation->domain->subdomain_count_horizontal > 1)
        {
            update_vertical_buffers();
        }

        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = kernels::EmplaceBounceBackKernel<A>(*simulation);
                    cgh.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>(simulation->domain->subdomain_vertical_nodes
                                                             * simulation->domain->subdomain_count_vertical,
                                                         simulation->domain->subdomain_horizontal_nodes
                                                             * simulation->domain->subdomain_count_horizontal),
                                          sycl::range<2>(simulation->domain->subdomain_vertical_nodes,
                                                         simulation->domain->subdomain_horizontal_nodes)),
                        kernel);
                })
            .wait();

        if (simulation->domain->subdomain_count_vertical > 1)
        {
            update_horizontal_buffers();
        }

        if (simulation->domain->subdomain_count_horizontal > 1)
        {
            update_vertical_buffers();
        }
    }

    /**
     * @brief   Updates the inlets and outlets as preparation for the two-lattice algorithm.
     */
    void perform_inout_update()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = general::buffered::OutletUpdateKernel<A>(*simulation);
                    cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel);
                })
            .wait();
    }

  public:
    /**
     * @brief   Runs the buffered two-lattice algorithm until it is paused or it reaches the last iteration.
     */
    inline void execute() override
    {
        future = std::async(
            [&]
            {
                while (simulation->control->is_execution_allowed())
                {
                    simulation->control->reset_timer();

                    emplace_bounce_back();
                    perform_inout_update();
                    stream_and_collide();

                    simulation->control->finalize_iteration();
                }

                if (simulation->control->is_finished())
                {
                    copy_macroscopic_observables_to_cpu();
                }
            });
    }

    explicit BufferedGpuTwoLattice(sycl::queue &queue, const std::string &settings_path) :
        SYCLAlgorithm(queue, settings_path)
    {
        core::domain_initialization::setup_domain<A, core::access::decomposed::BufferedNodeAccess>(*simulation, queue);
    }

    explicit BufferedGpuTwoLattice(sycl::queue &queue, const core::Properties &props) :
        SYCLAlgorithm(queue, props)
    {
        core::domain_initialization::setup_domain<A, core::access::decomposed::BufferedNodeAccess>(*simulation, queue);
    }
};

// DEBUG BUFFERED TWO-LATTICE /////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Class representation of the buffered SYCL debug implementation of the two-lattice
 *          algorithm.
 *
 * @tparam  A   any `core::access::AccessorConcept` from access.hpp
 */
template <core::access::AccessorConcept A>
class BufferedGpuTwoLatticeDebug : public execution::SYCLAlgorithm
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
     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
     *          the macroscopic observables in the process.
     */
    inline void stream_and_collide()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = buffered::kernels::StreamCollideKernel<A>(*simulation);
                    cgh.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>(simulation->domain->subdomain_vertical_nodes
                                                             * simulation->domain->subdomain_count_vertical,
                                                         simulation->domain->subdomain_horizontal_nodes
                                                             * simulation->domain->subdomain_count_horizontal),
                                          sycl::range<2>(simulation->domain->subdomain_vertical_nodes,
                                                         simulation->domain->subdomain_horizontal_nodes)),
                        kernel);
                })
            .wait();

        queue
            ->copy(simulation->data->distribution_values_0,
                   distribution_values->data(),
                   9 * simulation->domain->total_node_count)
            .wait();

        copy_macroscopic_observables_to_cpu();

        std::cout << "\033[36mDestination lattice after stream and collide: \n"
                  << "-------------------------------------------------------------------------------\n\033[0m";

        console::print_distribution_values<A>(*distribution_values, *phase_information, *simulation);

        std::cout << "Velocities: \n"
                  << "-------------------------------------------------------------------------------\n";

        lbm::console::print_velocities(
            *simulation->properties, *simulation->results->x_velocities_cpu, *simulation->results->y_velocities_cpu, 0);

        std::cout << "Densities: \n"
                  << "-------------------------------------------------------------------------------\n";
        lbm::console::print_densities(*simulation->properties, *simulation->results->densities_cpu, 0);
    }

    /**
     * @brief   Updates the horizontal buffers by copying adjacent values above and below.
     */
    void update_horizontal_buffers()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = general::buffered::HorizontalCopyToBufferKernel<A>(*simulation);
                    cgh.parallel_for(sycl::range<2>(simulation->domain->subdomain_count_vertical - 1,
                                                    simulation->domain->horizontal_nodes),
                                     kernel);
                })
            .wait();
    }

    /**
     * @brief   Updates the vertical buffers by copying adjacent values left and right.
     *
     */
    void update_vertical_buffers()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = general::buffered::VerticalCopyToBufferKernel<A>(*simulation);
                    cgh.parallel_for(sycl::range<2>(simulation->domain->vertical_nodes,
                                                    simulation->domain->subdomain_count_horizontal - 1),
                                     kernel);
                })
            .wait();
    }

    /**
     * @brief   Emplaces the values as preparation for the bounce-back scheme for the two-lattice algorithm.
     */
    void emplace_bounce_back()
    {
        if (simulation->domain->subdomain_count_vertical > 1)
        {
            update_horizontal_buffers();
        }

        if (simulation->domain->subdomain_count_horizontal > 1)
        {
            update_vertical_buffers();
        }

        auto event = queue->submit(
            [&](sycl::handler &cgh)
            {
                auto kernel = kernels::EmplaceBounceBackKernel<A>(*simulation);
                cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(simulation->domain->subdomain_vertical_nodes
                                                                      * simulation->domain->subdomain_count_vertical,
                                                                  simulation->domain->subdomain_horizontal_nodes
                                                                      * simulation->domain->subdomain_count_horizontal),
                                                   sycl::range<2>(simulation->domain->subdomain_vertical_nodes,
                                                                  simulation->domain->subdomain_horizontal_nodes)),
                                 kernel);
            });
        event.wait();

        if (simulation->domain->subdomain_count_vertical > 1)
        {
            update_horizontal_buffers();
        }

        if (simulation->domain->subdomain_count_horizontal > 1)
        {
            update_vertical_buffers();
        }

        queue
            ->copy(simulation->data->distribution_values_0,
                   distribution_values->data(),
                   9 * simulation->domain->total_node_count)
            .wait();

        std::cout << "\033[36mSource lattice after emplacing bounce-back values: \n"
                  << "-------------------------------------------------------------------------------\033[0m\n";

        console::print_distribution_values<A>(*distribution_values, *phase_information, *simulation);
    }

    /**
     * @brief   Updates the inlets and outlets as preparation for the two-lattice algorithm.
     */
    void perform_inout_update()
    {
        queue
            ->submit(
                [&](sycl::handler &cgh)
                {
                    auto kernel = general::buffered::OutletUpdateKernel<A>(*simulation);
                    cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes - 4), kernel);
                    // set to simulation->properties->vertical_nodes - 2 to also treat corners
                })
            .wait();

        queue
            ->copy(simulation->data->distribution_values_0,
                   distribution_values->data(),
                   9 * simulation->domain->total_node_count)
            .wait();

        std::cout << "\033[36mDestination lattice after updating inlets and outlets: \n"
                  << "-------------------------------------------------------------------------------\033[0m\n";

        console::print_distribution_values<A>(*distribution_values, *phase_information, *simulation);
    }

  public:
    /**
     * @brief   Runs the debug variant of the buffered two-lattice algorithm until it is paused or it
     *          reaches the last iteration.
     */
    inline void execute() override
    {
        if (current_iteration == 0)
        {
            console::print_ansi_color_message();
            console::print_color_legend();

            queue
                ->copy(simulation->data->distribution_values_0,
                       distribution_values->data(),
                       9 * simulation->domain->total_node_count)
                .wait();

            queue
                ->copy(simulation->data->phase_information,
                       phase_information->data(),
                       simulation->domain->total_node_count)
                .wait();

            std::cout << "Initial distribution values: \n";
            std::cout << "-------------------------------------------------------------------------------\n";

            console::print_distribution_values<A>(*distribution_values, *phase_information, *simulation);

            std::cout << "Phase information: \n";
            std::cout << "-------------------------------------------------------------------------------\n";

            console::print_phase_vector(*phase_information, simulation->domain->horizontal_nodes);

            std::cout << "\033[36mNow running buffered GPU two-lattice for " << simulation->properties->time_steps
                      << " iterations.\033[0m\n\n";

            std::cout << "Running on " << queue->get_device().get_info<sycl::info::device::name>() << "\n\n";

            fmt::print("Simulation properties:\n"
                       "-------------------------------------------------------------------------------\n");

            fmt::print(fmt::runtime(simulation->properties->to_string()));
        }

        future = std::async(
            [&]
            {
                while (simulation->control->is_execution_allowed())
                {
                    simulation->control->reset_timer();

                    std::cout
                        << "\033[36m===============================================================================\n"
                        << "Iteration " << current_iteration << "\n"
                        << "===============================================================================\033[0m\n\n";

                    emplace_bounce_back();
                    perform_inout_update();
                    stream_and_collide();

                    current_iteration++;

                    queue
                        ->copy(simulation->results->x_velocities_gpu,
                               temp_macroscopic_observables->data(),
                               simulation->properties->domain_node_count)
                        .wait();

                    all_x_velocities->insert(all_x_velocities->end(),
                                             temp_macroscopic_observables->begin(),
                                             temp_macroscopic_observables->end());

                    queue
                        ->copy(simulation->results->y_velocities_gpu,
                               temp_macroscopic_observables->data(),
                               simulation->properties->domain_node_count)
                        .wait();

                    all_y_velocities->insert(all_y_velocities->end(),
                                             temp_macroscopic_observables->begin(),
                                             temp_macroscopic_observables->end());

                    queue
                        ->copy(simulation->results->densities_gpu,
                               temp_macroscopic_observables->data(),
                               simulation->properties->domain_node_count)
                        .wait();

                    all_densities->insert(all_densities->end(),
                                          temp_macroscopic_observables->begin(),
                                          temp_macroscopic_observables->end());

                    simulation->control->finalize_iteration();

                    std::cout << "\033[36mFinished iteration " << current_iteration << " after "
                              << simulation->control->get_last_frametime() << " milliseconds.\033[0m\n\n";
                }

                if (simulation->control->is_finished())
                {
                    std::cout << "\033[36mAll done, exiting simulation. \033[0m\n\n";

                    lbm::console::print_simulation_results(
                        *simulation->properties, *all_densities, *all_x_velocities, *all_y_velocities);
                }
            });
    }

    explicit BufferedGpuTwoLatticeDebug(sycl::queue &queue, const std::string &settings_path) :
        SYCLAlgorithm(queue, settings_path),
        all_densities(std::make_unique<std::vector<real_type>>()),
        all_x_velocities(std::make_unique<std::vector<real_type>>()),
        all_y_velocities(std::make_unique<std::vector<real_type>>()),
        distribution_values(std::make_unique<std::vector<real_type>>(9 * simulation->domain->total_node_count, 0)),
        temp_macroscopic_observables(
            std::make_unique<std::vector<real_type>>(simulation->properties->domain_node_count, 0)),
        phase_information(std::make_unique<std::vector<int8_t>>(simulation->domain->total_node_count, 0)),
        current_iteration(0)
    {
        core::domain_initialization::setup_domain<A, core::access::decomposed::BufferedNodeAccess>(*simulation, queue);

        all_densities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
        all_densities->shrink_to_fit();

        all_x_velocities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
        all_x_velocities->shrink_to_fit();

        all_y_velocities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
        all_y_velocities->shrink_to_fit();

        distribution_values->shrink_to_fit();
        temp_macroscopic_observables->shrink_to_fit();
        phase_information->shrink_to_fit();
    }

    explicit BufferedGpuTwoLatticeDebug(sycl::queue &queue, const core::Properties &props) :
        SYCLAlgorithm(queue, props),
        all_densities(std::make_unique<std::vector<real_type>>()),
        all_x_velocities(std::make_unique<std::vector<real_type>>()),
        all_y_velocities(std::make_unique<std::vector<real_type>>()),
        distribution_values(std::make_unique<std::vector<real_type>>(9 * simulation->domain->total_node_count, 0)),
        temp_macroscopic_observables(
            std::make_unique<std::vector<real_type>>(simulation->properties->domain_node_count, 0)),
        phase_information(std::make_unique<std::vector<int8_t>>(simulation->domain->total_node_count, 0)),
        current_iteration(0)
    {
        core::domain_initialization::setup_domain<A, core::access::decomposed::BufferedNodeAccess>(*simulation, queue);

        all_densities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
        all_densities->shrink_to_fit();

        all_x_velocities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
        all_x_velocities->shrink_to_fit();

        all_y_velocities->reserve(simulation->properties->time_steps * simulation->properties->domain_node_count);
        all_y_velocities->shrink_to_fit();

        distribution_values->shrink_to_fit();
        temp_macroscopic_observables->shrink_to_fit();
        phase_information->shrink_to_fit();
    }
};

}  // namespace buffered

}  // namespace two_lattice

}  // namespace gpu

}  // namespace lbm

#endif  // ! BUFFERED_GPU_TWO_LATTICE_HPP
