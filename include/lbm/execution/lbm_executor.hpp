/**
 * @file        lbm_executor.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of an abstract class for an executor. The GUI is designed to be compatible 
 *              with various backends. To ensure compatibility, it is isolated from the implementation, and any interaction with the
 *              backend happens through specific executioners. They are used to launch the simulation and to retrieve the results.
 *              In principle, an executor can be used "synchronously" in the sense that iteration launches are synchronized with the
 *              GUI, or "asynchronously", that is, the simulation can run independently from the GUI.
 * 
 *              The only prerequisite is that the backend must provide the following functionality:
 * 
 *              - Check whether the simulation result is available without accessing its value
 *              - Asynchronous execution of the computations behind the simulation
 * 
 *              Executioners for all supported implementations are supposed to inherit from the abstract class declared here. 
 *              Any additional values that may be necessary for custom executioners should be added in its own class.
 * 
 * @version     2.0
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024 Marcel Graf
 */

#ifndef LBM_EXECUTOR_HPP
#define LBM_EXECUTOR_HPP

namespace lbm
{

    namespace execution
    {
        /**
         * @brief This abstract class determines the minimum basic functionality for an executor.
         */
        class Executor
        {

            public:

            /**
             * @brief   Executes the algorithm belonging to this executor until it is paused or until it reaches 
             *          the final iteration.
             */
            virtual void execute() = 0;

            /**
             * @brief Returns whether or not the computational result of an iteration is available at the time of access.
             */
            virtual bool is_ready() const = 0;
        };

    } // ! namespace execution

} // ! namespace lbm

#endif // ! LBM_EXECUTOR_HPP
