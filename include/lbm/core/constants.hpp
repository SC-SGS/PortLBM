/**
 * @brief       This header file contains constants used throughout the project.
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 *              Copyright (c) 2026 Alexander Strack
 */

#ifndef LBM_CONSTANTS_HPP
#define LBM_CONSTANTS_HPP

#include <array>
#include <string_view>

namespace lbm
{
// Floating-point precision throughout the entire lbm namespace
#ifdef USE_FLOAT
using real_type = float;  // single-precision (USE_FLOAT defined)
#else
using real_type = double;  // double-precision (default)
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

constexpr std::array<unsigned int, 8> streaming_directions = { 0, 1, 2, 3, 5, 6, 7, 8 };

constexpr std::array<unsigned int, 9> all_directions = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

constexpr std::array<real_type, 9> weights = { 1.0 / 36, 1.0 / 9,  1.0 / 36, 1.0 / 9, 4.0 / 9,
                                               1.0 / 9,  1.0 / 36, 1.0 / 9,  1.0 / 36 };

constexpr real_type boltzmann_constant = 1.380649e-23;

constexpr std::array<std::string_view, 4> algorithms = {
    "nptl" /* Non-linear Pull Two-Lattice */,
    "lptl" /* Linear Pull Two-Lattice */,
    "npol" /* Non-linear Pull One-Lattice */,
    "nsol" /* Non-linear Swap One-Lattice */
};

constexpr std::array<std::string_view, 3> data_layouts = { "collision", "stream", "bundle" };

constexpr std::array<std::string_view, 8> scenarios = { "Hagen-Poiseuille", "walls",  "circle", "square", "wing",
                                                        "skyscraper",       "porous", "plate" };

}  // namespace constants

}  // namespace core

}  // namespace lbm

#endif  // ! LBM_CONSTANTS_HPP
