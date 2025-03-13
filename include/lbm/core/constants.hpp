/**
 * @file        constants.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains constants used throughout the project.
 * 
 * @version     1.4
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) Marcel Graf
 */

#ifndef LBM_CONSTANTS_HPP
#define LBM_CONSTANTS_HPP

#include <array>

namespace lbm
{
    // Set float data type throughout the entire lbm namespace
    #ifdef USE_FLOAT
        using real_type = float;    // Set `real_type` to `float`, i.e., single-precision
    #else
        using real_type = double;   // Set `real_type` to `float`, i.e., double-precision
    #endif

    namespace core
    {

        /**
         * @brief This namespace contains various `constexpr` used throughout the project.
         */
        namespace constants
        {
            constexpr unsigned int dimension_count = 2;
            constexpr unsigned int direction_count = 9;
            constexpr real_type boltzmann = 1.380649e-23;

            constexpr std::array<unsigned int, 8> streaming_directions = {0, 1, 2, 3, 5, 6, 7, 8};

            constexpr std::array<unsigned int, 9> all_directions = {0, 1, 2, 3, 4, 5, 6, 7, 8};

            constexpr std::array<real_type, 9> weights = 
                {1.0/36, 1.0/9, 1.0/36, 1.0/9, 4.0/9, 1.0/9, 1.0/36, 1.0/9, 1.0/36}; 

            constexpr real_type boltzmann_constant = 1.380649e-23;

        } // ! namespace constants

    } // ! namespace core

} // ! namespace lbm

#endif // ! LBM_CONSTANTS_HPP
