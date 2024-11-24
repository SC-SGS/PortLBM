/**
 * @file    executor.hpp
 * 
 * @author  Marcel Graf
 * 
 * @brief   This header file contains the declaration of an abstract class for an executor.
 *          The GUI is designed to be compatible with various backends.
 *          To ensure compatibility, it is isolated from the implementation,
 *          and any interaction with the backend happens through specific executioners.
 *          They are used to launch the simulation and to retrieve the results.
 *          The only prerequisite is that the backend must provide the following functionality:
 *          - Check whether the simulation result is available without accessing its value
 *          - Asynchronous execution of the computations behind the simulation
 *          Executioners for all supported implementations are supposed to inherit from the abstract class.
 *          Any additional values that may be necessary for custom executioners should be added
 *          in its own class.
 * 
 * @version 2.0
 * 
 * @date    2024-08-28
 * 
 * @copyright Copyright (c) 2024 Marcel Graf
 */

#ifndef LBM_EXECUTOR_HPP
#define LBM_EXECUTOR_HPP

#include "../core/simulation.hpp"
#include "../file_interaction/file_interaction.hpp"

namespace lbm
{

    namespace execution
    {
        /**
         * @brief This abstract class determines the minimum basic functionality for an executor such as it is 
         *        required for use with the GUI.
         */
        template <typename ResultsType>
        class Executor
        {
            public:

                /**
                 * @brief Executes one iteration of the lattice Boltzmann algorithm and stores the value
                 *        in the specified object.
                 * 
                 */
                virtual void execute() = 0;

                /**
                 * @brief Returns whether or not the computational result of an iteration is available at the time of access.
                 * 
                 */
                virtual bool is_ready() const = 0;

                /**
                 * @brief Initializes an object representing the simulation data with the specified data.
                 */
                virtual void initialize() = 0;

                explicit Executor() 
                : 
                properties(lbm::file_interaction::json_to_properties()),
                simulation_results(std::make_unique<ResultsType>(properties->domain_node_count))
                {};

            std::unique_ptr<core::Properties> properties;
            std::unique_ptr<ResultsType> simulation_results;
        };

    } // ! namespace execution

} // ! namespace lbm

#endif // ! LBM_EXECUTOR_HPP
