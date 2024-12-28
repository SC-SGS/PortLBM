/**
 * @file        algorithm.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, an abstract class for algorithms is defined.
 *              All algorithms inherit from this this base class and implement the execute method.
 * 
 * @version     1.0
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_ALGORITHM_HPP
#define LBM_ALGORITHM_HPP

// Dependencies on other LBM core features
#include "../core/simulation.hpp"

// HPX
#include <hpx/future.hpp>

// SYCL
#include <sycl/sycl.hpp>

namespace lbm
{

    namespace execution
    {

        /**
         * @brief Abstract base class of all Lattice Boltzmann algorithms.
         */
        class Algorithm
        {
            public:

            std::unique_ptr<core::Simulation> simulation;
  
            /**
             * @brief   Returns whether the algorithm is currently within an iteration (`true`) or not (`false`).
             */
            inline bool is_ready() const { return future.is_ready(); }

            /**
             * @brief   Performs one iteration of this algorithm.
             *          The instructions are enqueued in the queue stored by this `Algorithm` object.
             */
            virtual void execute() = 0; 

            /**
             * @brief   Performs the specified amount of iterations of this algorithm.
             *          The instructions are enqueued in the queue stored by this `Algorithm` object.
             */
            virtual void execute(unsigned int time_steps) = 0; 

            virtual ~Algorithm() = default;

            /**
             * @brief This future is used to launch the algorithm and to check whether it is currently within an iteration or not.
             */
            hpx::future<void> future;

            protected:
            

            std::shared_ptr<sycl::queue> queue;

            /**
             * @brief   The constructor of an LBMAlgorithm object initializes the HPX future with a dummy value.
             */
            explicit Algorithm(const sycl::queue &queue)
            : 
            future(hpx::async([&]{})), 
            queue(std::make_shared<sycl::queue>(queue)), 
            simulation(std::make_unique<core::Simulation>())
            {};
        };
    }
}

#endif