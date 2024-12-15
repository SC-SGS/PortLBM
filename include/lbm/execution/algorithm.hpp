/**
 * @file algorithm.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-14
 * 
 * @copyright Copyright (c) 2024
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
             * @brief   Performs one iteration of this algorithm. The instructions are enqueued in the specified queue.
             * 
             * @param[in, out]  queue   the SYCL queue 
             */
            virtual void execute() = 0; 

            protected:
            
            /**
             * @brief This future is used to launch the algorithm and to check whether it is currently within an iteration or not.
             */
            hpx::future<void> future;
            std::shared_ptr<sycl::queue> queue;

            /**
             * @brief   The constructor of an LBMAlgorithm object initializes the HPX future with a dummy value.
             */
            explicit Algorithm(const sycl::queue &queue);
        };
    }
}

#endif