/**
 * @file        main_gpu.cpp
 *
 * @author      Marcel Graf
 *
 * @brief       Entry point of the parallel_lbm driver executable.
 *
 * @version     1.7
 *
 * @date        March 2025 (Phase-1 library refactor: May 2025)
 *
 * @copyright   Copyright (c) Marcel Graf
 */

// INCLUDES ///////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <filesystem>
#include <iostream>

// Library public API
#include <lbm/file_interaction/file_interaction.hpp>
#include <lbm/core/simulation.hpp>
#include <lbm/execution/sycl_algorithm_handler.hpp>

#ifdef WITH_VISUALIZATION
#include <lbm/gui/lbm_gui.hpp>
#endif

#ifdef BENCHMARK_MODE
#include <lbm/benchmark/benchmark.hpp>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// The settings file is located relative to the executable by convention.
// Application code owns this path — the library never hard-codes it.
static const std::string SETTINGS_PATH = "../settings/settings.json";

// ---------------------------------------------------------------------------
// main() — four variants controlled by preprocessor flags
// ---------------------------------------------------------------------------

#ifdef WITH_VISUALIZATION

    #ifdef BENCHMARK_MODE // WITH_VISUALIZATION && BENCHMARK_MODE

    int main(int argc, char* argv[])
    {
        std::filesystem::create_directories("../benchmarks/gui");

        try
        {
            lbm::gui::Gui<lbm::execution::SYCLAlgorithmHandler> gui("SYCL Lattice Boltzmann", SETTINGS_PATH);
            gui.run();
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << exception.to_string();
        }

        return 0;
    }

    #else // WITH_VISUALIZATION && !BENCHMARK_MODE

    int main(int argc, char* argv[])
    {
        try
        {
            lbm::gui::Gui<lbm::execution::SYCLAlgorithmHandler> gui("SYCL Lattice Boltzmann", SETTINGS_PATH);
            gui.run();
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << exception.to_string();
        }

        return 0;
    }

    #endif

#else // !WITH_VISUALIZATION

    #ifdef BENCHMARK_MODE // !WITH_VISUALIZATION && BENCHMARK_MODE

    int main(int argc, char* argv[])
    {
        std::filesystem::create_directories("../benchmarks/phase0");
        std::filesystem::create_directories("../benchmarks/phase1");
        std::filesystem::create_directories("../benchmarks/phase2");

        try
        {
            std::cout << "Starting benchmark. \n\n";

            std::shared_ptr<lbm::execution::SYCLAlgorithmHandler> algorithm_handler =
                std::make_shared<lbm::execution::SYCLAlgorithmHandler>(SETTINGS_PATH);

            lbm::benchmark::Benchmark benchmark(algorithm_handler);
            benchmark.phase_0();
            benchmark.phase_1();
            benchmark.phase_2();

            std::cout << "\nDone with all benchmarks. Exiting. \n\n";
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << exception.to_string();
        }

        return 0;
    }

    #else // !WITH_VISUALIZATION && !BENCHMARK_MODE

    int main(int argc, char* argv[])
    {
        try
        {
            std::unique_ptr<lbm::execution::SYCLAlgorithmHandler> algorithm_handler =
                std::make_unique<lbm::execution::SYCLAlgorithmHandler>(SETTINGS_PATH);
            algorithm_handler->start();
            algorithm_handler->block_until_finished();
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << exception.to_string();
        }

        return 0;
    }

    #endif // !BENCHMARK_MODE

#endif // !WITH_VISUALIZATION

// ! main_gpu.cpp
