/**
 * @file        sycl_executor.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of a class for a SYCL executor.
 *              Through this executor, it is possible to launch simulation iterations.
 * 
 * @version     2.1
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024 Marcel Graf
 */

#ifndef SYCL_EXECUTOR_HPP
#define SYCL_EXECUTOR_HPP

#include "lbm_executor.hpp"
#include "algorithm.hpp"

// Dependencies on other LBM core features
#include "../core/domain_initialization.hpp"

// Standard library
#include <iostream>

// Linear GPU two-lattice
#include "../gpu/two_lattice/linear/usm_linear_gpu_two_lattice.hpp"

namespace lbm
{

    namespace execution
    {

        /**
         * @brief   Executor class for a SYCL backend.
         *          It is used to execute the specified algorithm using the specified accessor object class.
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
             * @brief   Starts a new simulation running for the specified amount of time steps. 
             *          The computations are managed by the SYCL runtime.
             */
            inline void execute(unsigned int time_steps) override { algorithm->execute(time_steps); }

            /**
             * @brief   Returns whether or not the future of this executor is ready.
             */
            inline bool is_ready() const override { return algorithm->is_ready(); }
        };

    } // ! namespace execution

} // ! namespace lbm

#endif