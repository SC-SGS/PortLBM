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
#include "../gpu/two_lattice/linear/linear_gpu_two_lattice.hpp"

#include <hpx/future.hpp>
#include <iostream>
//#include <sycl/sycl.hpp>

namespace lbm
{

    namespace execution
    {

        /**
         * @brief Executor class for a SYCL backend.
         */
        template <class LBMAccessor>
        class SYCLExecutor : public Executor<core::SimulationResults>
        {
            static_assert
            (
                std::is_base_of<core::access::LBMAccessorObject, LBMAccessor>::value, 
                "Template class must have base class core::access::LBMAccessorObject."
            );

            public:

            std::unique_ptr<core::SimulationData<LBMAccessor>> simulation_data;

            /**
             * @brief Construct a new SYCL executor and initializes it such that it stores
             *        the current simulation data present by default.
             * 
             */
            SYCLExecutor()
            :
            Executor(),
            simulation_data(std::make_unique<lbm::core::SimulationData<LBMAccessor>>(*properties)),
            future(hpx::async([&]{})),
            device_selector(),
            queue(device_selector)
            {
                lbm::core::setup_pipe_flow_environment(*properties, *simulation_data);
                *(simulation_data->distribution_values_1) = *(simulation_data->distribution_values_0); 
            };

            /**
             * @brief Launches an HPX task to start a new iteration of the simulation.
             *        The computations are managed by the SYCL runtime.
             * 
             */
            void execute() override
            {
                future = hpx::async
                (
                    [&]
                    {
                        gpu::two_lattice::linear::emplace_bounce_back(*properties, *simulation_data, queue);
                        gpu::two_lattice::linear::perform_inout_update(*properties, *simulation_data, queue);
                        gpu::two_lattice::linear::stream_and_collide(*properties, *simulation_data, *simulation_results, queue);
                        simulation_data->distribution_values_1.swap(simulation_data->distribution_values_0);
                    }
                );
            }

            /**
             * @brief Returns whether or not the future of this executor is ready.
             * 
             */
            inline bool is_ready() const override
            {
                return future.is_ready();
            }

            /**
             * @brief Initializes the future of this executor using the specified data.
             * 
             * @param data a reference to the simulation data tuple that will be the content of the future
             */
            void initialize() override
            {
                std::cout << "Initializing executor...\n";
                properties = lbm::file_interaction::json_to_properties();
                std::cout << properties->to_string();
                simulation_data = std::make_unique<lbm::core::SimulationData<LBMAccessor>>(*properties);
                lbm::core::setup_pipe_flow_environment(*properties, *simulation_data);
                *(simulation_data->distribution_values_1) = *(simulation_data->distribution_values_0); 
                simulation_results = std::make_unique<core::SimulationResults>(*properties);
            }

            private:

            hpx::future<void> future;
            sycl::default_selector device_selector; 
            sycl::queue queue;
        };

    } // ! namespace execution

} // ! namespace lbm

#endif