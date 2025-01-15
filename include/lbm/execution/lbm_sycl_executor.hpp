/**
 * @file        lbm_sycl_executor.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of a class for a SYCL executor. Through this executor, it is possible
 *              to launch simulations using a SYCL-based backend.
 * 
 * @version     2.2
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024 Marcel Graf
 */

#ifndef SYCL_EXECUTOR_HPP
#define SYCL_EXECUTOR_HPP

// Dependencies on LBM execution features
#include "algorithm.hpp"
#include "lbm_executor.hpp"

// Dependencies on other LBM core features
#include "../core/domain_initialization.hpp"

// Linear GPU two-lattice
#include "../gpu/two_lattice/linear/linear_gpu_two_lattice.hpp"

// Standard library
#include <iostream>

namespace lbm
{

    namespace execution
    {

        /**
         * @brief   Executor class for a SYCL backend. It stores an `lbm::execution::Algorithm` specifying both the algorithm
         *          and the data it operates on. That is, the algorithm is a small closed system of its own. The executor can
         *          query whether or not an algorithm is ready, that is, whether it has finished its assigned work load or not.
         */
        class SYCLExecutor : public Executor
        {

            private:

            std::unique_ptr<sycl::default_selector> device_selector; 
            std::shared_ptr<sycl::queue> queue;

            public:

            std::unique_ptr<Algorithm> algorithm;

            /**
             * @brief Construct a new SYCL executor and initializes it such that it stores
             *        the current simulation data present by default.
             */
            explicit SYCLExecutor();

            /**
             * @brief   Starts a new iteration of the simulation. The computations are managed by the SYCL runtime.
             */
            inline void execute() override { algorithm->execute(); }

            /**
             * @brief   Waits for the queue to finish its operations, e.g. for a graceful exit.
             */
            inline void wait_for_queue() { queue->wait(); }

            /**
             * @brief   Returns whether or not the future of this executor is ready.
             */
            inline bool is_ready() const override { return algorithm->is_ready(); }
        };

    } // ! namespace execution

} // ! namespace lbm

#endif // ! SYCL_EXECUTOR_HPP