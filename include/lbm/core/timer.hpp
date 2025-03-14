/**
 * @file        timer.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of a class for a timer. It is based on elements from 
 *              std::chrono.
 * 
 * @version     1.2
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) Marcel Graf
 * 
 */

#ifndef LBM_TIMER_HPP
#define LBM_TIMER_HPP

// Standard library
#include <chrono>

namespace lbm
{

    namespace core
    {
     
    /**
     * @brief   This class defines a simple timer that can be reset and queried the elapsed time in seconds.
     */
    class Timer
    {
        public:

            /**
             * @brief   Constructs a new simple timer and initializes it with the current time.
             */
            Timer();

            /**
             * @brief   Resets the initial time of the timer to the current time.
             */
            void restart();

            /**
             * @brief   Returns the elapsed time in seconds since the last initialization.
             */
            double elapsed();

        private:

            /**
             * @brief   The time at which this timer was constructed or restarted the last time.
             */
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    };

    } // ! namespace core

} // ! namespace lbm

#endif // ! LBM_TIMER_HPP
