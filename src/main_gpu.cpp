/**
 * @file        main_gpu.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Entry point of the program executing the GPU lattice Boltzmann application.
 * 
 * @version     1.2
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024
 */

// INCLUDES ///////////////////////////////////////////////////////////////////////////////////////

// Fundamental LBM includes that are always necessary regardless of the CMake configuration
#include "../include/lbm/file_interaction/file_interaction.hpp"
#include "../include/lbm/core/simulation.hpp"

// HPX includes that are always necessary regardless of the CMake configuration
#include <hpx/hpx_init.hpp>
#include <hpx/execution.hpp>

// Conditional includes
#ifdef WITH_VISUALIZATION                       // If the flag for building with GUI features is set, ...
#include "../include/lbm/gui/lbm_gui.hpp"       // include the LBM GUI features, ...
#else                                           // else, ...
#include "../execution/lbm_sycl_executor.hpp"   // include only the executor.
#endif

// DEFINITIONS ////////////////////////////////////////////////////////////////////////////////////

#ifdef WITH_VISUALIZATION // Flag is set if -DWITH_VISUALIZATION=ON, must be specified by user

int hpx_main(hpx::program_options::variables_map& vm)
{
    try
    {
        lbm::gui::Gui gui("SYCL Lattice Boltzmann");
        gui.run();
    }
    catch(const lbm::exceptions::Exception &exception)
    {
        std::cerr << exception.to_string();
    }

    return hpx::local::finalize();
}

#else // ! WITH_VISUALIZATION

int hpx_main(hpx::program_options::variables_map& vm)
{
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

        executor->execute(executor->algorithm->simulation->properties->time_steps);
        executor->algorithm->block_until_finished();
    }
    catch(const lbm::exceptions::Exception &exception)
    {
        std::cerr << exception.to_string();
    }

    return hpx::local::finalize();
}

#endif // End of macro-managed definitions based on flag WITH_VISUALIZATION


int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
