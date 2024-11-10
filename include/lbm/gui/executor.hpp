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
 * @version 1.0
 * 
 * @date    2024-08-28
 * 
 * @copyright Copyright (c) 2024 Marcel Graf
 */

#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <memory>

/**
 * @brief This abstract class determines the minimum basic functionality for an executor such as it is 
 *        required for use with the GUI.
 */
template <typename ContainerType, typename ResultsType>
class Executor
{
    public:

        /**
         * @brief Executes one iteration of the lattice Boltzmann algorithm and stores the value
         *        in the specified object.
         * 
         * @param gui_simulation_data a reference to a struct containing the relevant data
         */
        virtual void execute(ContainerType &data) = 0;

        /**
         * @brief Returns whether or not the computational result of an iteration is available at the time of access.
         * 
         */
        virtual bool is_ready() const = 0;

        /**
         * @brief Returns the simulation data generated during the current iteration.
         * 
         */
        virtual std::unique_ptr<ResultsType> get() = 0;

        /**
         * @brief Initializes an object representing the simulation data with the specified data.
         * 
         * @param data a reference to the new value of the object
         */
        virtual void initialize(ContainerType &data) = 0;
};
#endif
