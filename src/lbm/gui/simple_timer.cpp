/**
 * @file        simple_timer.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This source file contains the implementation of the simple timer class defined in simple_timer.hpp.
 * 
 * @version     1.0
 * 
 * @date        2024-08-29
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/gui/simple_timer.hpp"

SimpleTimer::SimpleTimer() : 
    start_time_(std::chrono::high_resolution_clock::now()) {};

void SimpleTimer::restart()
{
    start_time_ = std::chrono::high_resolution_clock::now();
}

double SimpleTimer::elapsed()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> now = std::chrono::high_resolution_clock::now();
    return ((std::chrono::duration<double, std::ratio<1>>)(now - start_time_)).count();
}