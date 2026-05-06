/**
 * @file        sycl_algorithm.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       In this header file, an abstract class for algorithms is defined.
 *              All algorithms inherit from this this base class and implement the execute methods.
 *
 * @version     1.2
 *
 * @date        March 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#ifndef LBM_SYCL_ALGORITHM_HPP
#define LBM_SYCL_ALGORITHM_HPP

// Dependencies on other LBM core features
#include "../core/simulation.hpp"

// LBM exceptions
#include "../exceptions/exceptions.hpp"

// SYCL
#include <sycl/sycl.hpp>

namespace lbm
{

namespace execution
{

/**
 * @brief   Abstract base class of all lattice Boltzmann algorithms. Its non-abstract child classes must
 *          implement the execute methods.
 */
class SYCLAlgorithm
{
  protected:
    /**
     * @brief   This future is used to launch the algorithm.
     */
    std::future<void> future;

    /**
     * @brief   Shared pointer to the queue that is used for communication with the GPU.
     */
    std::shared_ptr<sycl::queue> queue;

    /**
     * @brief   The constructor of a SYCLAlgorithm object initializes the future with a dummy value.
     *
     * @param[in]   queue           the SYCL queue used for communication with the device
     * @param[in]   settings_path   path to the JSON settings file; forwarded to core::Simulation
     */
    explicit SYCLAlgorithm(sycl::queue &queue, const std::string &settings_path) :
        future(std::async([] {})),
        queue(std::make_shared<sycl::queue>(queue)),
        simulation(std::make_unique<core::Simulation>(queue, settings_path)){};

  public:
    /**
     * @brief   Unique pointer to the simulation object on which this algorithm operates.
     */
    std::unique_ptr<core::Simulation> simulation;

    /**
     * @brief   Blocks the thread on which this method is accessed until the algorithm is paused or reaches its
     *          final iteration.
     */
    inline void block_until_finished()
    {
        try
        {
            future.get();
        }
        catch (const std::exception &e)
        {
            throw exceptions::algorithm::WaitException(
                "A thread accessed a blocking statement causing it to wait for an algorithm that has not been "
                "launched.");
        }
    }

    inline void copy_macroscopic_observables_to_cpu()
    {
        queue->copy(simulation->results->densities_gpu,
                    simulation->results->densities_cpu->data(),
                    simulation->results->densities_cpu->size());
        queue->copy(simulation->results->x_velocities_gpu,
                    simulation->results->x_velocities_cpu->data(),
                    simulation->results->x_velocities_cpu->size());
        queue->copy(simulation->results->y_velocities_gpu,
                    simulation->results->y_velocities_cpu->data(),
                    simulation->results->y_velocities_cpu->size());
        queue->copy(simulation->results->absolute_velocities_gpu,
                    simulation->results->absolute_velocities_cpu->data(),
                    simulation->results->absolute_velocities_cpu->size());
    }

    /**
     * @brief   Performs the algorithm until it is either paused or it reaches the last iteration. The
     *          instructions are enqueued in the queue stored by this `Algorithm` object.
     */
    virtual void execute() = 0;

    /**
     * @brief   Explicit declaration of default destructor is necessary for polymorphism.
     */
    virtual ~SYCLAlgorithm() = default;
};

}  // namespace execution

}  // namespace lbm

#endif  // ! LBM_SYCL_ALGORITHM_HPP
