/**
 * @file        main_gpu.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Entry point of the program executing the GPU lattice Boltzmann application.
 * 
 * @version     1.0
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 */

#include "../include/lbm/file_interaction/file_interaction.hpp"
#include "../include/lbm/core/simulation.hpp"
#include "../include/lbm/gpu/two_lattice/linear/linear_gpu_two_lattice.hpp"
#include "../include/lbm/execution/lbm_sycl_executor.hpp"
#include "../include/lbm/gui/lbm_gui.hpp"

#include <hpx/hpx_init.hpp>
#include <hpx/execution.hpp>

int hpx_main(hpx::program_options::variables_map& vm)
{
    lbm::gui::run<lbm::execution::SYCLExecutor<lbm::core::access::LBMStreamAccessor>>("Task-based lattice Boltzmann with HPX");

    /*
    std::unique_ptr<lbm::core::Properties> properties = lbm::file_interaction::json_to_properties();

    std::unique_ptr<lbm::core::SimulationData<lbm::core::access::LBMStreamAccessor>> simulation_data = 
        std::make_unique<lbm::core::SimulationData<lbm::core::access::LBMStreamAccessor>>(*properties);

    lbm::core::setup_pipe_flow_environment(*properties, *simulation_data);

    *(simulation_data->distribution_values_1) = *(simulation_data->distribution_values_0); 

    if(properties->debug_mode)
    {
        fmt::print("Starting LBM on GPU test with two-lattice algorithm...\n\n");

        lbm::console::print_ansi_color_message();
        lbm::console::print_color_legend();

        fmt::print
        (
            "Simulation properties:\n"
            "-------------------------------------------------------------------------------\n"
        );
        fmt::print(fmt::runtime(properties->to_string()));


        lbm::console::debug_prints(*simulation_data);

        try
        {
            lbm::gpu::two_lattice::linear::debug::run<lbm::core::access::LBMStreamAccessor>(*properties, *simulation_data);
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << "Exception while executing " << properties->algorithm << "algorithm.\n\n" << exception.to_string();
        }
    }
    else
    {
        try
        {
            lbm::gpu::two_lattice::linear::run<lbm::core::access::LBMStreamAccessor>(*properties, *simulation_data);
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << "Exception while executing " << properties->algorithm << "algorithm.\n\n" << exception.to_string();
        }
    }
    */
   std::cout << "About to finalize \n";
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}