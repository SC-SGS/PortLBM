/**
 * @file    sycl_executor.hpp
 * 
 * @author  Marcel Graf
 * 
 * @brief   This header file contains the declaration of a class for an HPX executor.
 *          Through this executor, it is possible for the GUI to launch simulation iterations
 *          and to retrieve results.
 * 
 * @version 2.1
 * 
 * @date    December 2024
 * 
 * @copyright Copyright (c) 2024 Marcel Graf
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
#include "../gpu/two_lattice/linear/linear_gpu_two_lattice.hpp"

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
            explicit SYCLExecutor()
            : 
            device_selector(std::make_unique<sycl::default_selector>()), 
            queue(std::make_unique<sycl::queue>(*device_selector))
            {
                
                *(algorithm->simulation->data->distribution_values_1) = *(algorithm->simulation->data->distribution_values_0); 

                if(algorithm->simulation->properties->algorithm == "gpu-two-lattice-linear")
                {
                    if(algorithm->simulation->properties->data_layout == "stream")
                    {
                        algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::StreamAccessor>>(*queue);
                        lbm::core::setup_pipe_flow_environment<core::access::experimental::StreamAccessor>(*algorithm->simulation);
                    }
                    else if(algorithm->simulation->properties->data_layout == "collision")
                    {
                        algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::CollisionAccessor>>(*queue);
                        lbm::core::setup_pipe_flow_environment<core::access::experimental::CollisionAccessor>(*algorithm->simulation);
                    }
                    else if(algorithm->simulation->properties->data_layout == "bundle")
                    {
                        algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::BundleAccessor>>(*queue);
                        lbm::core::setup_pipe_flow_environment<core::access::experimental::BundleAccessor>(*algorithm->simulation);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: ", algorithm->simulation->properties->data_layout));
                    }
                }
                else if(algorithm->simulation->properties->algorithm == "gpu-two-lattice")
                {
                    throw exceptions::Exception("This algorithm is not implemented yet.");
                }
                else if(algorithm->simulation->properties->algorithm == "gpu-swap")
                {
                    throw exceptions::Exception("This algorithm is not implemented yet.");
                }
                else
                {
                    throw exceptions::Exception(fmt::format("Unknown algorithm: ", algorithm->simulation->properties->data_layout));
                }
            };

            /**
             * @brief Launches an HPX task to start a new iteration of the simulation.
             *        The computations are managed by the SYCL runtime.
             * 
             */
            inline void execute() override
            {
                algorithm->execute();
            }

            /**
             * @brief Returns whether or not the future of this executor is ready.
             * 
             */
            inline bool is_ready() const override
            {
                return algorithm->is_ready();
            }

            // /**
            //  * @brief Initializes the future of this executor using the specified data.
            //  * 
            //  * @param data a reference to the simulation data tuple that will be the content of the future
            //  */
            // inline void initialize() override
            // {
            //     properties = lbm::file_interaction::json_to_properties();
            //     simulation_data = std::make_unique<lbm::core::SimulationData<LBMAccessor>>(*properties);
            //     lbm::core::setup_pipe_flow_environment(*properties, *simulation_data);
            //     *(simulation_data->distribution_values_1) = *(simulation_data->distribution_values_0); 
            //     simulation_results = std::make_unique<core::SimulationResults>(*properties);
            // }
        };

    } // ! namespace execution

} // ! namespace lbm

#endif