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
    std::unique_ptr<core::Properties> properties = lbm::file_interaction::json_to_properties();

    if(properties->algorithm == "gpu-two-lattice-linear")
    {
        if(properties->data_layout == "stream")
        {
            algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::StreamAccessor>>(*queue);
            lbm::core::setup_pipe_flow_environment<core::access::experimental::StreamAccessor>(*algorithm->simulation);
        }
        else if(properties->data_layout == "collision")
        {
            algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::CollisionAccessor>>(*queue);
            lbm::core::setup_pipe_flow_environment<core::access::experimental::CollisionAccessor>(*algorithm->simulation);
        }
        else if(properties->data_layout == "bundle")
        {
            algorithm = std::make_unique<gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::experimental::BundleAccessor>>(*queue);
            lbm::core::setup_pipe_flow_environment<core::access::experimental::BundleAccessor>(*algorithm->simulation);
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