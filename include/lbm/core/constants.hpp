/**
 * @file        constants.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains constants used throughout the project.
 * 
 * @version     1.1
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 */

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <array>

namespace lbm
{

    namespace core
    {

        /**
         * @brief This namespace contains various `constexpr` used throughout the project.
         */
        namespace constants
        {
            constexpr unsigned int dimension_count = 2;
            constexpr unsigned int direction_count = 9;
            constexpr double boltzmann = 1.380649e-23;

            constexpr std::array<unsigned int, 8> streaming_directions = {0, 1, 2, 3, 5, 6, 7, 8};
            constexpr std::array<unsigned int, 9> all_directions = {0, 1, 2, 3, 4, 5, 6, 7, 8};
            constexpr std::array<double, 9> weights = {1.0/36, 1.0/9, 1.0/36, 1.0/9, 4.0/9, 1.0/9, 1.0/36, 1.0/9, 1.0/36}; 

            constexpr double boltzmann_constant = 1.380649e-23;

        } // ! namespace constants

    } // ! namespace core

} // ! namespace lbm

#endif