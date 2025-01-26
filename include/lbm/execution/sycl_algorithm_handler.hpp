/**
 * @file        sycl_algorithm_handler.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of a SYCL algorithm handler class.
 * 
 * @version     1.0
 * 
 * @date        January 2025
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
             * @brief   (Re-)Initializes the `lbm::execution::SYCLAlgorithm` object belonging to this SYCL algorithm 
             *          handler object.
             */
            inline void initialize_algorithm()
            {
                std::unique_ptr<core::Properties> properties = 
                    std::make_unique<core::Properties>(lbm::file_interaction::json_to_properties());

                if(!properties->debug_mode)
                {
                    if(properties->algorithm == "gpu-two-lattice-linear")
                    {
                        algorithm = gpu::two_lattice::linear::get_algorithm_pointer(*properties, *queue);
                    }
                    else if(properties->algorithm == "gpu-two-lattice")
                    {
                        algorithm = gpu::two_lattice::non_linear::get_algorithm_pointer(*properties, *queue);
                    }
                    else if(properties->algorithm == "gpu-swap")
                    {
                        throw exceptions::Exception("This algorithm is not implemented yet.");
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown algorithm: ", properties->data_layout));
                    }
                }
                else
                {
                    if(properties->algorithm == "gpu-two-lattice-linear")
                    {
                        algorithm = gpu::two_lattice::linear::get_debug_algorithm_pointer(*properties, *queue);
                    }
                    else if(properties->algorithm == "gpu-two-lattice")
                    {
                        algorithm = gpu::two_lattice::non_linear::get_debug_algorithm_pointer(*properties, *queue);
                    }
                    else if(properties->algorithm == "gpu-swap")
                    {
                        throw exceptions::Exception("This algorithm is not implemented yet.");
                    }
                    else
                    {
                        throw exceptions::Exception(fmt::format("Unknown algorithm: ", properties->data_layout));
                    }
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
            device_selector(std::make_unique<sycl::default_selector>()), 
            queue(std::make_unique<sycl::queue>(*device_selector)),
            active(false)
            {
                initialize_algorithm();
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

            inline void block_until_finished() override { algorithm->block_until_finished(); }

            inline std::vector<double> &get_densities() const override 
            { return *algorithm->simulation->results->densities_cpu; }

            inline std::vector<double> &get_x_velocities() const override 
            { return *algorithm->simulation->results->x_velocities_cpu; }

            inline std::vector<double> &get_y_velocities() const override 
            { return *algorithm->simulation->results->y_velocities_cpu; }

            inline std::vector<double> &get_absolute_velocities() const override 
            { return *algorithm->simulation->results->absolute_velocities_cpu; }

            inline unsigned int get_horizontal_nodes() const override 
            { return algorithm->simulation->properties->horizontal_nodes; }
        
            inline unsigned int get_vertical_nodes() const override 
            { return algorithm->simulation->properties->vertical_nodes; }

            inline double get_progress() const override 
            { return algorithm->simulation->control->get_progress(); }

            inline double get_last_frametime() const override 
            { return algorithm->simulation->control->get_last_frametime(); }    

            inline double get_inlet_density() const override 
            { return algorithm->simulation->properties->inlet_density; }

            inline double get_outlet_density() const override 
            { return algorithm->simulation->properties->outlet_density; }          
        };

    } // ! namespace execution

} // ! namespace lbm

#endif // ! LBM_SYCL_ALGORITHM_HANDLER_HPP
