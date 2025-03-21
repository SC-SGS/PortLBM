/**
 * @file        algorithm_handler.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of an abstract class for an algorithm handler. The GUI is 
 *              designed to be compatible with various backends. To ensure compatibility, it is widely isolated from 
 *              the implementation, and any interaction with the backend happens through specific handlers. They are 
 *              used to launch the simulation and to retrieve the results. In principle, an algorithm handler can be 
 *              used "synchronously" in the sense that iteration launches are synchronized with the GUI, or 
 *              "asynchronously", that is, the simulation can run independently from the GUI.
 * 
 *              For convenience reasons, algorithm handlers are also used in the non-GUI version.
 *               
 *              Algorithm handlers for all supported implementations are supposed to inherit from the abstract class
 *              declared here. Any additional values that may be necessary for custom executioners should be added in 
 *              their own classes.
 * 
 * @version     1.4
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) Marcel Graf
 * 
 */

#ifndef LBM_ALGORITHM_HANDLER_HPP
#define LBM_ALGORITHM_HANDLER_HPP

// LBM core functionality: Only required for real_type, may be outsourced for different backends
#include "constants.hpp"

// Standard library
#include <vector>
#include <concepts>

namespace lbm
{

    namespace execution
    {
        /**
         * @brief   This abstract class defines the minimum public interface of an LBM algorithm handler that isolates
         *          the GUI or any outside applications from the LBM implementation.
         */
        class AlgorithmHandler
        {
            public:

            /**
             * @brief   Allows to specify a constraint on the maximum number of processing elements. This variable is
             *          present here such that it is available to the GUI, where the number of processing elements is
             *          potentially adjusted in some way. 
             */
            size_t processing_element_constraint;

            /**
             * @brief   (Re-)Initializes the algorithm addressed by this algorithm handler.
             */
            virtual void initialize() = 0;

            /**
             * @brief   Starts or resumes the execution of the lattice Boltzmann simulation belonging to this algorithm 
             *          handler. It is executed until it is paused or until it reaches the final iteration. Algorithms 
             *          that reached their final iteration can be started but execution will conclude immediately since
             *          there are no more iterations to be executed.
             */
            virtual void start() = 0;

            /**
             * @brief   Pauses the execution of the algorithm addressed by this algorithm handler and waits for all 
             *          interactions regarding the backend to be completed. The intention is that this function can also
             *          be used to prepare the initialization of a new algorithm or a graceful exit of the GUI. 
             */
            virtual void pause() = 0;

            /**
             * @brief   Returns whether or not the addressed algorithm has completed its final iteration.
             */
            virtual bool is_finished() const = 0;

            /**
             * @brief   Blocks the thread on which this method is accessed until the addressed algorithm has reached
             *          its final iteration. Notice that implementations of this method may throw exceptions if the
             *          algorithm may not terminate for some reasons.
             */
            virtual void block_until_finished() = 0;

            /**
             * @brief   Returns a reference to the vector internally storing the density values.
             */
            virtual std::vector<real_type> &get_densities() const = 0;

            /**
             * @brief   Returns a reference to the vector internally storing the x-components of the velocities.
             */
            virtual std::vector<real_type> &get_x_velocities() const = 0;

            /**
             * @brief   Returns a reference to the vector internally storing the y-components of the velocities.
             */
            virtual std::vector<real_type> &get_y_velocities() const = 0;

            /**
             * @brief   Returns a reference to the vector internally storing the absolutes of the velocity values.
             */
            virtual std::vector<real_type> &get_absolute_velocities() const = 0;

            /**
             * @brief   Returns the number of horizontal nodes of the unexpanded simulated domain including ghost nodes.
             */
            virtual unsigned int get_horizontal_nodes() const = 0;

            /**
             * @brief   Returns the number of vertical nodes of the unexpanded simulated domain including ghost nodes.
             */
            virtual unsigned int get_vertical_nodes() const = 0;

            /**
             * @brief   Returns the density specified at the inlet of the simulation domain of the addressed algorithm.
             */
            virtual real_type get_inlet_density() const = 0;

            /**
             * @brief   Returns the density specified at the outlet of the simulation domain of the addressed algorithm.
             */
            virtual real_type get_outlet_density() const = 0;

            /**
             * @brief   Returns the current progress of the addressed algorithm, that is, which fraction of the total
             *          iteration count is already finished.
             */
            virtual real_type get_progress() const = 0;

            /**
             * @brief   Returns the last frame time of the addressed algorithm, that is, the duration of the last LBM
             *          iteration. Notice that the frametimes and the framerates of the backend do not necessarily 
             *          correspond to those of the GUI.
             */
            virtual real_type get_last_frametime() const = 0;

            protected:

            explicit AlgorithmHandler(size_t processing_element_constraint) : 
            processing_element_constraint(processing_element_constraint) {};
        };

        /**
         * @brief   This concept is intended for use with classes that possibly rely on multiple LBM Backends. All 
         *          classes that are derived from `lbm::execution::AlgorithmHandler` are considered algorithm handler 
         *          concepts.
         * 
         * @tparam  T   It is checked whether this class is an `AlgorithmHandlerConcept`
         */
        template <class T> 
        concept AlgorithmHandlerConcept = std::derived_from<T, AlgorithmHandler>;

    } // ! namespace execution

} // ! namespace lbm

#endif // ! LBM_ALGORITHM_HANDLER_HPP
