/**
 * @file        main_gpu.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Entry point of the program executing the GPU lattice Boltzmann application.
 * 
 * @version     1.6
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) Marcel Graf
 */

// INCLUDES ///////////////////////////////////////////////////////////////////////////////////////////////////////////

// Fundamental LBM includes that are always necessary regardless of the CMake configuration
#include "../include/lbm/file_interaction/file_interaction.hpp"
#include "../include/lbm/core/simulation.hpp"
#include "../execution/sycl_algorithm_handler.hpp"

// Conditional includes
#ifdef WITH_VISUALIZATION                       // If the flag for building with GUI features is set, ...
#include "../include/lbm/gui/lbm_gui.hpp"       // include the LBM GUI features
#endif

#ifdef BENCHMARK_MODE
#include "../include/lbm/benchmark/benchmark.hpp"
#endif

// DEFINITIONS ////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef WITH_VISUALIZATION // Flag is set if -DWITH_VISUALIZATION=ON, must be specified by user

    #ifdef BENCHMARK_MODE // WITH WITH_VISUALIZATION && BENCHMARK_MODE

    int main(int argc, char* argv[])
    {
        system("mkdir ../benchmarks ../benchmarks/gui");

        try
        {
            lbm::gui::Gui<lbm::execution::SYCLAlgorithmHandler> gui("SYCL Lattice Boltzmann");
            gui.run();
        }
        catch(const lbm::exceptions::Exception &exception)
        {
            std::cerr << exception.to_string();
        }

        return 0;
    }

    #else // WITH WITH_VISUALIZATION && ! BENCHMARK_MODE

        int main(int argc, char* argv[])
        {
            try
            {
                lbm::gui::Gui<lbm::execution::SYCLAlgorithmHandler> gui("SYCL Lattice Boltzmann");
                gui.run();
            }
            catch(const lbm::exceptions::Exception &exception)
            {
                std::cerr << exception.to_string();
            }

            return 0;
        }

    #endif


#else // ! WITH_VISUALIZATION

    #ifdef BENCHMARK_MODE // ! WITH WITH_VISUALIZATION && BENCHMARK_MODE

        int main(int argc, char* argv[])
        {
            try
            {
                std::cout << "Starting benchmark. \n\n";
                system("mkdir ../benchmarks ../benchmarks/phase0 ../benchmarks/phase1 ../benchmarks/phase2");
                std::shared_ptr<lbm::execution::SYCLAlgorithmHandler> algorithm_handler = 
                    std::make_shared<lbm::execution::SYCLAlgorithmHandler>();

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

    #else // ! WITH WITH_VISUALIZATION && ! BENCHMARK_MODE

        int main(int argc, char* argv[])
        {
            try
            {
                std::unique_ptr<lbm::execution::SYCLAlgorithmHandler> algorithm_handler = 
                    std::make_unique<lbm::execution::SYCLAlgorithmHandler>();
                algorithm_handler->start();
                algorithm_handler->block_until_finished();
            }
            catch(const lbm::exceptions::Exception &exception)
            {
                std::cerr << exception.to_string();
            }

            return 0;
        }

    #endif // End of non-benchmark non-visualized main

#endif // End of macro-managed definitions based on flag WITH_VISUALIZATION

// ! main_gpu.cpp
