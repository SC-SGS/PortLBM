/**
 * @brief       This source file contains the implementation of the timer class defined in timer.hpp.
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 *              Copyright (c) 2026 Alexander Strack
 */

#include "../../../include/lbm/core/timer.hpp"

lbm::core::Timer::Timer() :
    start_time_(std::chrono::high_resolution_clock::now()){};

void lbm::core::Timer::restart() { start_time_ = std::chrono::high_resolution_clock::now(); }

double lbm::core::Timer::elapsed()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> now = std::chrono::high_resolution_clock::now();
    return ((std::chrono::duration<double, std::ratio<1>>) (now - start_time_)).count();
}
