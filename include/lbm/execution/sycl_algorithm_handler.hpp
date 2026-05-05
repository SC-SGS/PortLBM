/**
 * @file        sycl_algorithm_handler.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of a SYCL algorithm handler class.
 * 
 * @version     1.4
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) 2025 Marcel Graf
 */

#ifndef LBM_SYCL_ALGORITHM_HANDLER_HPP
#define LBM_SYCL_ALGORITHM_HANDLER_HPP

// Dependencies on LBM execution features
#include "sycl_algorithm.hpp"
#include "algorithm_handler.hpp"

// Dependencies on other LBM features
#include "../core/domain_initialization.hpp"
#include "../file_interaction/file_interaction.hpp"

// SYCL-based LBM algorithms
#include "../gpu/two_lattice/linear/linear_gpu_two_lattice.hpp"
#include "../gpu/two_lattice/non-linear/non_linear_gpu_two_lattice.hpp"
#include "../gpu/two_lattice/buffered/buffered_gpu_two_lattice.hpp"
#include "../gpu/swap/gpu_swap.hpp"

#ifdef BENCHMARK_MODE
#include "../core/timer.hpp"
#endif

// Standard library
#include <iostream>

namespace lbm
{

    namespace execution
    {

        /**
         * @brief   Handler class for SYCL-based lattice Boltzmann algorithms. It stores an `lbm::execution::Algorithm`
         *          specifying both the algorithm itself and the data it operates on. That is, the algorithm is a small
         *          closed system of its own. This class inherits from the abstract `lbm::execution::AlgorithmHandler`
         *          class and implements its interface. 
         */
        class SYCLAlgorithmHandler : public AlgorithmHandler
        {

// PRIVATE INTERNALS //////////////////////////////////////////////////////////////////////////////////////////////////

            private:

            #ifdef FORCE_USE_CPU
            std::unique_ptr<sycl::cpu_selector> cpu_selector;
            #else
            std::unique_ptr<sycl::default_selector> device_selector;
            #endif

            std::unique_ptr<SYCLAlgorithm> algorithm;
            bool active;

            // Path to the JSON settings file. Callers supply this at construction; it is
            // forwarded to every algorithm/simulation object so the library never hard-codes
            // a relative path to the settings file.
            std::string settings_path_;

            #ifdef BENCHMARK_MODE
            std::unique_ptr<core::Timer> timer;
            #endif

            /**
             * @brief   Initializes the performance variant of an algorithm, as specified in the properties object.
             * 
             * @param[in]   properties  contains the algorithm and the data layout
             */
            inline void initialize_non_debug_algorithm(const core::Properties &properties)
            {
                if(properties.algorithm == "gpu-two-lattice-linear")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::StreamAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::CollisionAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::BundleAccessor>
                                >(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: {}", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-two-lattice")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLattice<core::access::StreamAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLattice<core::access::CollisionAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLattice<core::access::BundleAccessor>
                                >(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: {}", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-two-lattice-buffered")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::buffered::BufferedGpuTwoLattice<core::access::StreamAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::buffered::BufferedGpuTwoLattice<core::access::CollisionAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::buffered::BufferedGpuTwoLattice<core::access::BundleAccessor>
                                >(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: {}", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-swap")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<gpu::swap::GpuSwap<core::access::StreamAccessor>>(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<gpu::swap::GpuSwap<core::access::CollisionAccessor>>(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<gpu::swap::GpuSwap<core::access::BundleAccessor>>(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(
                            fmt::format("Unknown data layout: {}", properties.data_layout)
                        );
                    }
                }
                else
                {
                    throw exceptions::Exception(fmt::format("Unknown algorithm: {}", properties.algorithm));
                }
            }

            /**
             * @brief   Initializes the debug variant of an algorithm, as specified in the properties object.
             * 
             * @param[in]   properties  contains the algorithm and the data layout
             */
            inline void initialize_debug_algorithm(const core::Properties &properties)
            {
                if(properties.algorithm == "gpu-two-lattice-linear")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLatticeDebug<core::access::StreamAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLatticeDebug<core::access::CollisionAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLatticeDebug<core::access::BundleAccessor>
                                >(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: {}", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-two-lattice")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLatticeDebug<core::access::StreamAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLatticeDebug<core::access::CollisionAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLatticeDebug<core::access::BundleAccessor>
                                >(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: {}", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-two-lattice-buffered")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::buffered::BufferedGpuTwoLatticeDebug<core::access::StreamAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::buffered::BufferedGpuTwoLatticeDebug<core::access::CollisionAccessor>
                                >(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::buffered::BufferedGpuTwoLatticeDebug<core::access::BundleAccessor>
                                >(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: {}", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-swap")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = 
                            std::make_unique<gpu::swap::GpuSwapDebug<core::access::StreamAccessor>>(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = 
                            std::make_unique<gpu::swap::GpuSwapDebug<core::access::CollisionAccessor>>(*queue, settings_path_);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = 
                            std::make_unique<gpu::swap::GpuSwapDebug<core::access::BundleAccessor>>(*queue, settings_path_);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: {}", properties.data_layout));
                    }
                }
                else
                {
                    throw exceptions::Exception(fmt::format("Unknown algorithm: {}", properties.algorithm));
                }
            }

            /**
             * @brief   (Re-)Initializes the `lbm::execution::SYCLAlgorithm` object belonging to this SYCL algorithm 
             *          handler object.
             */
            inline void initialize_algorithm()
            {
                #ifdef BENCHMARK_MODE
                timer->restart();
                #endif
                core::Properties properties = lbm::file_interaction::json_to_properties(settings_path_);

                if(!properties.debug_mode) { initialize_non_debug_algorithm(properties); }
                else { initialize_debug_algorithm(properties); }        
                
                #ifdef BENCHMARK_MODE
                initialization_time = timer->elapsed();
                #endif        
            }

// PUBLIC API /////////////////////////////////////////////////////////////////////////////////////////////////////////

            public:

            std::shared_ptr<sycl::queue> queue;

            #ifdef BENCHMARK_MODE
            double initialization_time;
            #endif

            #ifdef FORCE_USE_CPU

            /**
             * @brief   Constructs a new SYCL algorithm handler and immediately initializes the simulation.
             *
             * @param[in]   settings_path   path to the JSON settings file (e.g. "settings/settings.json").
             *                              No default — callers must supply an explicit path.
             */
            explicit SYCLAlgorithmHandler(const std::string &settings_path)
            :
            AlgorithmHandler(0),
            cpu_selector(std::make_unique<sycl::cpu_selector>()),
            queue(std::make_unique<sycl::queue>(*cpu_selector)),
            active(false),
            settings_path_(settings_path)
            {
                #ifdef BENCHMARK_MODE
                timer = std::make_unique<core::Timer>();
                #endif

                initialize_algorithm();
                processing_element_constraint = queue->get_device().get_info<sycl::info::device::max_work_group_size>();
            };

            #else

            /**
             * @brief   Constructs a new SYCL algorithm handler and immediately initializes the simulation.
             *
             * @param[in]   settings_path   path to the JSON settings file (e.g. "settings/settings.json").
             *                              No default — callers must supply an explicit path.
             */
            explicit SYCLAlgorithmHandler(const std::string &settings_path)
            :
            AlgorithmHandler(0),
            device_selector(std::make_unique<sycl::default_selector>()),
            queue(std::make_unique<sycl::queue>(*device_selector)),
            active(false),
            settings_path_(settings_path)
            {
                #ifdef BENCHMARK_MODE
                timer = std::make_unique<core::Timer>();
                #endif

                initialize_algorithm();
                processing_element_constraint = queue->get_device().get_info<sycl::info::device::max_work_group_size>();
            };

            #endif

            inline void initialize() override
            {
                algorithm.reset();
                initialize_algorithm();
            }

            inline void start() override 
            { 
                if(!active) 
                {
                    algorithm->simulation->control->allow_execution();
                    algorithm->execute(); 
                    active = true;
                }
            }

            inline void pause() override
            {
                algorithm->simulation->control->forbid_execution();
                queue->wait();
                active = false;
            }

            inline bool is_finished() const override { return algorithm->simulation->control->is_finished(); }

            inline void block_until_finished() override 
            { 
                algorithm->block_until_finished();
                queue->wait(); 
                active = false;
            }

            inline std::vector<real_type> &get_densities() const override 
            { return *algorithm->simulation->results->densities_cpu; }

            inline std::vector<real_type> &get_x_velocities() const override 
            { return *algorithm->simulation->results->x_velocities_cpu; }

            inline std::vector<real_type> &get_y_velocities() const override 
            { return *algorithm->simulation->results->y_velocities_cpu; }

            inline std::vector<real_type> &get_absolute_velocities() const override 
            { return *algorithm->simulation->results->absolute_velocities_cpu; }

            inline unsigned int get_horizontal_nodes() const override 
            { return algorithm->simulation->properties->horizontal_nodes; }
        
            inline unsigned int get_vertical_nodes() const override 
            { return algorithm->simulation->properties->vertical_nodes; }

            inline real_type get_progress() const override 
            { return algorithm->simulation->control->get_progress(); }

            inline real_type get_last_frametime() const override 
            { return algorithm->simulation->control->get_last_frametime(); }    

            inline real_type get_inlet_density() const override 
            { return algorithm->simulation->properties->inlet_density; }

            inline real_type get_outlet_density() const override
            { return algorithm->simulation->properties->outlet_density; }

            inline const std::string &get_settings_path() const { return settings_path_; }
        };

    } // ! namespace execution

} // ! namespace lbm

#endif // ! LBM_SYCL_ALGORITHM_HANDLER_HPP
