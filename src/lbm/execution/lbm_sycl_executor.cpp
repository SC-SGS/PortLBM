/**
 * @file lbm_sycl_executor.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/execution/lbm_sycl_executor.hpp"
#include "../../../include/lbm/file_interaction/file_interaction.hpp"

/**
 * @brief Construct a new SYCL executor and initializes it such that it stores
 *        the current simulation data present by default.
 */
lbm::execution::SYCLExecutor::SYCLExecutor()
: 
device_selector(std::make_unique<sycl::default_selector>()), 
queue(std::make_unique<sycl::queue>(*device_selector))
{
    std::cout << "lbm_sycl_executor.cpp:\tWithin constructor\n";
    std::unique_ptr<core::Properties> properties = std::make_unique<core::Properties>(lbm::file_interaction::json_to_properties());

    if(properties->algorithm == "gpu-two-lattice-linear")
    {
        std::cout << "lbm_sycl_executor.cpp:\tAlgorithm detected: Linear GPU two-lattice\n";
        if(properties->data_layout == "stream")
        {
            std::cout << "lbm_sycl_executor.cpp:\tCreating algorithm object\n";
            algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::StreamAccessor>>(*queue);
            std::cout << "lbm_sycl_executor.cpp:\tSetting up pipe flow environment\n";
            lbm::core::setup_pipe_flow_environment<core::access::experimental::StreamAccessor>(*algorithm->simulation, *queue, core::domain_initialization::Obstacle::NONE);
        }
        else if(properties->data_layout == "collision")
        {
            algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::CollisionAccessor>>(*queue);
            lbm::core::setup_pipe_flow_environment<core::access::experimental::CollisionAccessor>(*algorithm->simulation, *queue, core::domain_initialization::Obstacle::NONE);
        }
        else if(properties->data_layout == "bundle")
        {
            algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::BundleAccessor>>(*queue);
            lbm::core::setup_pipe_flow_environment<core::access::experimental::BundleAccessor>(*algorithm->simulation, *queue, core::domain_initialization::Obstacle::NONE);
        }
        else
        {
            throw exceptions::Exception(fmt::format("Unknown data layout: ", properties->data_layout));
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
        throw exceptions::Exception(fmt::format("Unknown algorithm: ", properties->data_layout));
    }
    *(algorithm->simulation->data->distribution_values_1) = *(algorithm->simulation->data->distribution_values_0); 
};