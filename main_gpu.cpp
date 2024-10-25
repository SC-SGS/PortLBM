/**
 * @file        main_gpu.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Entry point of the program executing the GPU lattice Boltzmann application.
 * 
 * @version     1.0
 * 
 * @date        2024-10-09
 * 
 * @copyright   Copyright (c) 2024
 */

#include "include/lbm/file_interaction/file_interaction.hpp"
#include "include/lbm/core/simulation.hpp"
#include "include/lbm/gpu/two_lattice/gpu_two_lattice.hpp"

#include <hpx/hpx_init.hpp>
#include <hpx/execution.hpp>

int hpx_main(hpx::program_options::variables_map& vm)
{
    fmt::print("Starting LBM on GPU test with two-lattice algorithm...\n\n");

    lbm::console::print_ansi_color_message();
    lbm::console::print_color_legend();

    std::shared_ptr<lbm::Properties> properties = lbm::file_interaction::json_to_properties();

    fmt::print
    (
        "Simulation properties:\n"
        "-------------------------------------------------------------------------------\n"
    );

    fmt::print(fmt::runtime(properties->to_string()));

    std::shared_ptr<lbm::SimulationResults> simulation_results = std::make_shared<lbm::SimulationResults>(*properties);
    std::shared_ptr<lbm::SimulationData<lbm::access::LBMStreamAccessor>> simulation_data = std::make_shared<lbm::SimulationData<lbm::access::LBMStreamAccessor>>(*properties);

    lbm::border_swap_information swap_info;

    lbm::setup_pipe_flow_environment(*properties, *simulation_data);

    *(simulation_data->distribution_values_1) = *(simulation_data->distribution_values_0); 

    swap_info = lbm::bounce_back::retrieve_border_swap_info(*properties, *simulation_data);

    if(properties->debug_mode)
    {
        lbm::console::debug_prints(*simulation_data, swap_info);

        try
        {
            lbm::gpu::two_lattice::run_debug_new(*properties, swap_info, *simulation_data);
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << "Exception while executing " << properties->algorithm << "algorithm.\n\n" << exception.to_string();
        }
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}