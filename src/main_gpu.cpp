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
#include "../execution/lbm_sycl_executor.hpp"

#ifdef WITH_VISUALIZATION
#include "../include/lbm/gui/lbm_gui.hpp"
#else
#include "../execution/lbm_sycl_executor.hpp"
#endif

#include <hpx/hpx_init.hpp>
#include <hpx/execution.hpp>

int hpx_main(hpx::program_options::variables_map& vm)
{
    #ifdef WITH_VISUALIZATION

    try
    {
        lbm::gui::Gui gui("SYCL Lattice Boltzmann");
        gui.run();
    }
    catch(const lbm::exceptions::Exception &exception)
    {
        std::cerr << exception.to_string();
    }

    #else

    try
    {
        std::unique_ptr<lbm::execution::SYCLExecutor> executor = std::make_unique<lbm::execution::SYCLExecutor>();
        lbm::console::print_ansi_color_message();
        lbm::console::print_color_legend();

        fmt::print
        (
            "Simulation properties:\n"
            "-------------------------------------------------------------------------------\n"
        );
        fmt::print(fmt::runtime(executor->algorithm->simulation->properties->to_string()));
        for(unsigned int time_step = 0; time_step < executor->algorithm->simulation->properties->time_steps; ++time_step)
        {
            // if(executor->is_ready())
            executor->execute();
            // auto test = executor->algorithm->future.get();
        }
    }
    catch(const lbm::exceptions::Exception &exception)
    {
        std::cerr << exception.to_string();
    }

    #endif

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
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}