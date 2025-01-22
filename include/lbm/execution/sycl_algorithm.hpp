/**
 * @file        sycl_algorithm.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, an abstract class for algorithms is defined.
 *              All algorithms inherit from this this base class and implement the execute methods.
 * 
 * @version     1.0
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_SYCL_ALGORITHM_HPP
#define LBM_SYCL_ALGORITHM_HPP

// Dependencies on other LBM core features
#include "../core/simulation.hpp"

// LBM exceptions
#include "../exceptions/exceptions.hpp"

// HPX
#include <hpx/future.hpp>

// SYCL
#include <sycl/sycl.hpp>

namespace lbm
{

    namespace execution
    {

        /**
         * @brief   Abstract base class of all Lattice Boltzmann algorithms.
         *          Its non-abstract child classes must implement the execute methods.
         */
        class SYCLAlgorithm
        {

            protected:
            
            /**
             * @brief   This future is used to launch the algorithm.
             */
            hpx::future<void> future;

            std::shared_ptr<sycl::queue> queue;

            /**
             * @brief   The constructor of an LBMAlgorithm object initializes the HPX future with a dummy value.
             */
            explicit SYCLAlgorithm(sycl::queue &queue)
            : 
            future(hpx::async([]{})), 
            queue(std::make_shared<sycl::queue>(queue)), 
            simulation(std::make_unique<core::Simulation>(queue))
            {};

            public:

            std::unique_ptr<core::Simulation> simulation;
  
            /**
             * @brief   Returns whether the algorithm is currently within an iteration (`true`) or not (`false`).
             */
            inline bool is_ready() const { return future.is_ready(); }

            /**
             * @brief   Blocks the thread on which this method is accessed until the algorithm
             */
            inline void block_until_finished()
            {
                try { future.get(); }
                catch(const std::exception& e) 
                {
                    throw exceptions::algorithm::WaitException(
                        "A thread accessed a blocking statement causing it to wait for an algorithm that has not been launched."
                    ); 
                }
            }

            /**
             * @brief   Performs the algorithm until it is either paused or it reaches the last iteration.
             *          The instructions are enqueued in the queue stored by this `Algorithm` object.
             */
            virtual void execute() = 0; 

            virtual ~SYCLAlgorithm() = default;
        };

    } // ! namespace execution

} // ! namespace lbm

#endif // ! LBM_ALGORITHM_HPP
