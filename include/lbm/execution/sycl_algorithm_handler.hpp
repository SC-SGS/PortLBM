/**
 * @file        sycl_algorithm_handler.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of a SYCL algorithm handler class.
 * 
 * @version     1.2
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
// #include "../gpu/two_lattice/optimized/optimized_gpu_two_lattice.hpp"
#include "../gpu/swap/gpu_swap.hpp"

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

            std::unique_ptr<sycl::default_selector> device_selector; 
            std::shared_ptr<sycl::queue> queue;
            std::unique_ptr<SYCLAlgorithm> algorithm;
            bool active;

            /**
             * @brief   Initializes the non-debug variant of the algorithm specified in the according properties object.
             * 
             * @param[in]   properties  the algorithm that is to be set up is stored in this object
             */
            inline void initialize_non_debug_algorithm(const core::Properties &properties)
            {
                if(properties.algorithm == "gpu-two-lattice-linear")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::StreamAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::CollisionAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLattice<core::access::BundleAccessor>
                                >(*queue);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: ", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-two-lattice")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLattice<core::access::StreamAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLattice<core::access::CollisionAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLattice<core::access::BundleAccessor>
                                >(*queue);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: ", properties.data_layout));
                    }
                }
                // else if(properties->algorithm == "gpu-two-lattice-optimized")
                // {

                // }
                else if(properties.algorithm == "gpu-swap")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<gpu::swap::GpuSwap<core::access::StreamAccessor>>(*queue);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<gpu::swap::GpuSwap<core::access::CollisionAccessor>>(*queue);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<gpu::swap::GpuSwap<core::access::BundleAccessor>>(*queue);
                    }
                    else
                    {
                        throw exceptions::Exception(
                            fmt::format("Unknown data layout: ", properties.data_layout)
                        );
                    }
                }
                else
                {
                    throw exceptions::Exception(fmt::format("Unknown algorithm: ", properties.data_layout));
                }
            }

            /**
             * @brief   Initializes the non-debug variant of the algorithm specified in the according properties object.
             * 
             * @param[in]   properties  the algorithm that is to be set up is stored in this object
             */
            inline void initialize_debug_algorithm(const core::Properties &properties)
            {
                if(properties.algorithm == "gpu-two-lattice-linear")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLatticeDebug<core::access::StreamAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLatticeDebug<core::access::CollisionAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::linear::LinearGpuTwoLatticeDebug<core::access::BundleAccessor>
                                >(*queue);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: ", properties.data_layout));
                    }
                }
                else if(properties.algorithm == "gpu-two-lattice")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLatticeDebug<core::access::StreamAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLatticeDebug<core::access::CollisionAccessor>
                                >(*queue);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = std::make_unique<
                            gpu::two_lattice::non_linear::NonLinearGpuTwoLatticeDebug<core::access::BundleAccessor>
                                >(*queue);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: ", properties.data_layout));
                    }
                }
                // else if(properties->algorithm == "gpu-two-lattice-optimized")
                // {
                //     
                // }
                else if(properties.algorithm == "gpu-swap")
                {
                    if(properties.data_layout == "stream")
                    {
                        algorithm = 
                            std::make_unique<gpu::swap::GpuSwapDebug<core::access::StreamAccessor>>(*queue);
                    }
                    else if(properties.data_layout == "collision")
                    {
                        algorithm = 
                            std::make_unique<gpu::swap::GpuSwapDebug<core::access::CollisionAccessor>>(*queue);
                    }
                    else if(properties.data_layout == "bundle")
                    {
                        algorithm = 
                            std::make_unique<gpu::swap::GpuSwapDebug<core::access::BundleAccessor>>(*queue);
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown data layout: ", properties.data_layout));
                    }
                }
                else
                {
                    throw exceptions::Exception(fmt::format("Unknown algorithm: ", properties.data_layout));
                }
            }

            /**
             * @brief   (Re-)Initializes the `lbm::execution::SYCLAlgorithm` object belonging to this SYCL algorithm 
             *          handler object.
             */
            inline void initialize_algorithm()
            {
                std::unique_ptr<core::Properties> properties = 
                    std::make_unique<core::Properties>(lbm::file_interaction::json_to_properties());             

                if(!properties->debug_mode)
                {
                    initialize_non_debug_algorithm(*properties);
                }
                else
                {
                    initialize_debug_algorithm(*properties);
                }                
            }

// PUBLIC API /////////////////////////////////////////////////////////////////////////////////////////////////////////

            public:

            /**
             * @brief   Construct a new SYCL executor and initializes it such that it stores the current simulation data
             *          present by default.
             */
            explicit SYCLAlgorithmHandler()
            : 
            AlgorithmHandler(0),
            device_selector(std::make_unique<sycl::default_selector>()), 
            queue(std::make_unique<sycl::queue>(*device_selector)),
            active(false)
            {
                initialize_algorithm();
                processing_element_constraint = queue->get_device().get_info<sycl::info::device::max_work_group_size>();
            };

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
        };

    } // ! namespace execution

} // ! namespace lbm

#endif // ! LBM_SYCL_ALGORITHM_HANDLER_HPP
