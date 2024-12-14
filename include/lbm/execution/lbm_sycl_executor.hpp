/**
 * @file    sycl_executor.hpp
 * 
 * @author  Marcel Graf
 * 
 * @brief   This header file contains the declaration of a class for an HPX executor.
 *          Through this executor, it is possible for the GUI to launch simulation iterations
 *          and to retrieve results.
 * 
 * @version 2.0
 * 
 * @date    2024-08-28
 * 
 * @copyright Copyright (c) 2024 Marcel Graf
 */

#ifndef SYCL_EXECUTOR_HPP
#define SYCL_EXECUTOR_HPP

#include "lbm_executor.hpp"
#include "../core/domain_initialization.hpp"
#include "../core/simulation.hpp"

#include <hpx/future.hpp>
#include <iostream>

namespace lbm
{

    namespace execution
    {

        /**
         * @brief Abstract base class of all Lattice Boltzmann algorithms.
         * 
         * @tparam LBMAccessor any child class of `lbm::core::access::LBMAccessorObject`
         */
        class Algorithm
        {
            protected:
            
            /**
             * @brief This future is used to launch the algorithm and to check whether it is currently within an iteration or not.
             */
            hpx::future<void> future;

            std::unique_ptr<core::Properties> properties;
            std::unique_ptr<core::Simulation> simulation;
            std::unique_ptr<sycl::queue> queue;

            explicit Algorithm();

            public:

            /**
             * @brief The constructor of an LBMAlgorithm object initializes the HPX future with a dummy value.
             */
            inline explicit Algorithm() : future(hpx::async([&]{})) {};
  
            /**
             * @brief Returns whether the algorithm is currently within an iteration (`true`) or not (`false`).
             */
            inline bool is_ready() const { return future.is_ready(); }

            /**
             * @brief Performs one iteration of this algorithm. The instructions are enqueued in the specified queue.
             * 
             * @param[in]       properties          the structue containing the simulation properties
             * @param[in, out]  simulation_data     the simulation object on which the algorithm operates
             * @param[in, out]  queue               the SYCL queue 
             */
            virtual void execute() = 0; 
        };

        /**
         * @brief Executor class for a SYCL backend.
         *        It is used to execute the specified algorithm using the specified accessor object class.
         * 
         * @tparam  LBMAccessor any child class of `lbm::core::access::LBMAccessorObject
         */
        class SYCLExecutor : public Executor
        {

            private:

            std::unique_ptr<sycl::default_selector> device_selector; 
            std::unique_ptr<sycl::queue> queue;

            public:

            std::unique_ptr<LBMAlgorithm> algorithm;
            std::unique_ptr<core::Properties> properties;
            std::unique_ptr<core::Simulation> simulation;

            /**
             * @brief Construct a new SYCL executor and initializes it such that it stores
             *        the current simulation data present by default.
             */
            explicit SYCLExecutor()
            : 
            properties(lbm::file_interaction::json_to_properties()),
            simulation_data(std::make_unique<lbm::core::SimulationData<LBMAccessor>>(*properties)),
            simulation_results(std::make_unique<ResultsType>(properties->domain_node_count))
            device_selector(std::make_unique<sycl::default_selector>()), 
            queue(std::make_unique<sycl::queue>(*device_selector)) 
            {
                lbm::core::setup_pipe_flow_environment(*properties, *simulation_data);
                *(simulation_data->distribution_values_1) = *(simulation_data->distribution_values_0); 

                if(properties->algorithm == "gpu-two-lattice-linear")
                {
                    if(properties->data_layout == "stream")
                    {
                        executor = std::make_unique<execution::SYCLExecutor<core::access::LBMStreamAccessor>>(new execution::SYCLExecutor<core::access::LBMStreamAccessor>());
                    }
                    else if(properties->data_layout == "collision")
                    {
                        executor = std::make_unique<execution::SYCLExecutor<core::access::LBMCollisionAccessor>>(new execution::SYCLExecutor<core::access::LBMCollisionAccessor>());
                    }
                    else if(properties->data_layout == "bundle")
                    {
                        executor = std::make_unique<execution::SYCLExecutor<core::access::LBMBundleAccessor>>(new execution::SYCLExecutor<core::access::LBMBundleAccessor>());
                    }
                    else
                    {
                        throw exceptions::Exception("Unknown data layout: " + data_layout);
                    }
                }
                else if(properties->algorithm == "gpu-two-lattice")
                {
                    throw exceptions::Exception("This algorithm is not implemented yet.");
                }
                else if(properties->algorithm == "gpu-swap")
                {
                    throw exceptions::Exception("This algorithm is not implemented yet.");
                }
                else
                {
                    throw exceptions::Exception("Unknown algorithm: " + algorithm);
                }
            };

            /**
             * @brief Launches an HPX task to start a new iteration of the simulation.
             *        The computations are managed by the SYCL runtime.
             * 
             */
            inline void execute() override
            {
                algorithm->execute(*properties, *simulation_data, *simulation_results, *queue);
            }

            /**
             * @brief Returns whether or not the future of this executor is ready.
             * 
             */
            inline bool is_ready() const override
            {
                return algorithm->is_ready();
            }

            /**
             * @brief Initializes the future of this executor using the specified data.
             * 
             * @param data a reference to the simulation data tuple that will be the content of the future
             */
            inline void initialize() override
            {
                properties = lbm::file_interaction::json_to_properties();
                simulation_data = std::make_unique<lbm::core::SimulationData<LBMAccessor>>(*properties);
                lbm::core::setup_pipe_flow_environment(*properties, *simulation_data);
                *(simulation_data->distribution_values_1) = *(simulation_data->distribution_values_0); 
                simulation_results = std::make_unique<core::SimulationResults>(*properties);
            }
        };

    } // ! namespace execution

} // ! namespace lbm

#endif