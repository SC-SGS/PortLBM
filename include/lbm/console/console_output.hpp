/**
 * @file        console_output.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       This header file contains the declarations of various functions for console outputs.
 *
 * @version     1.6
 *
 * @date        March 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#ifndef LBM_CONSOLE_OUTPUT_HPP
#define LBM_CONSOLE_OUTPUT_HPP

// LBM core dependencies
#include "../core/access.hpp"
#include "../core/simulation.hpp"

// Format
#include <fmt/color.h>
#include <fmt/core.h>

// Standard library
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace lbm
{

/**
 * @brief This namespace contains various functions for printing information to the console.
 */
namespace console
{

// ANSI COLOR CODES ///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Prints the message to the console that mentions the usage of ANSI colors and the implications.
 */
inline void print_ansi_color_message()
{
    fmt::print(
        "This program utilizes ANSI color codes to output colored text. \n"
        "If your terminal does not support those codes, the output may be corrupted.\n"
        "Please also make sure that the color scheme of your terminal supports the used colors.\n"
        "The color names listed below should be colored.\n\n"
        "Used colors:\t\033[31mred\033[0m, "
        "\033[33myellow\033[0m, "
        "\033[32mgreen\033[0m, "
        "\033[36mcyan\033[0m, "
        "\033[34mblue\033[0m.\n\n"
        "Notice:\t\tSince the text is set to its default format (\\033[0m) multiple times in this program,\n"
        "\t\tformatted text, especially from exceptions or third-party code, may not have its typical color.\n\n");
}

/**
 * @brief   Prints the legend explaining the meaning of the colors used in the console output.
 */
inline void print_color_legend()
{
    fmt::print("Legend:\n"
               "-------------------------------------------------------------------------------\n"
               "\033[31mRED\033[0m:\tValues of the node in the origin\n"
               "\033[33mYELLOW\033[0m:\tPhase illustration: \033[33mx = 1\033[0m marks a solid node\n"
               "\033[32mGREEN\033[0m:\teverything related to buffer nodes or ghost nodes\n"
               "\033[36mCYAN\033[0m:\tMilestones and important events\n"
               "\033[34mBLUE\033[0m:\t1.) Distribution values of the \"outmost\" node\n"
               "\t2.) Phase illustration: \033[34m0\033[0m marks a fluid node\n\n");
}

// PRINT VECTORS //////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Allows to print out a vector representing a data layout whose column count is `horizontal_nodes`.
 *          Notice that the vector is assumed to represent a matrix.
 *
 * @tparam  T   the type of the objects the specified vector holds (must be numeric)
 *
 * @param[in]   vector              the contents of this vector are printed
 * @param[in]   horizontal_nodes    column count of the specified vector
 */
template <typename T>
void print_vector(const std::vector<T> &vector, const unsigned int horizontal_nodes)
{
    unsigned int vertical_nodes = vector.size() / horizontal_nodes;

    for (auto y = vertical_nodes; y-- > 0;)
    {
        for (auto x = 0; x < horizontal_nodes; ++x)
        {
            if (x == 0 && y == 0)
            {
                std::cout << "\033[31m";
            }
            else if (x == (horizontal_nodes - 1) && y == (vertical_nodes - 1))
            {
                std::cout << "\033[34m";
            }
            std::cout << vector.at(core::access::get_node_index(x, y, horizontal_nodes)) << "\t\033[0m";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// PRINT DISTRIBUTION VALUES OF ALL KINDS OF DOMAINS //////////////////////////////////////////////////////////////////

/**
 * @brief   Prints all distribution values to the console. They are displayed in the original order, i.e., the
 *          origin is located at the lower left corner of the printed distribution chart.
 *
 * @tparam  A   any `core::access::AccessorConcept` from access.hpp
 *
 * @param[in]   distribution_values   a vector containing the distribution values of all nodes
 * @param[in]   horizontal_nodes      total horizontal nodes in the subdomain
 * @param[in]   vertical_nodes        total vertical nodes in the subdomain
 */
template <core::access::AccessorConcept A>
void print_distribution_values(const std::vector<real_type> &distribution_values,
                               const std::vector<int8_t> &phase_information,
                               const core::Simulation &simulation)
{
    constexpr std::array<size_t, 9> print_dirs = { 6, 7, 8, 3, 4, 5, 0, 1, 2 };
    std::vector<real_type> current_values(9, 0);

    unsigned int current_node_index = 0;
    std::cout << std::setprecision(3) << std::fixed;

    for (unsigned int y = simulation.domain->vertical_nodes; y-- > 0;)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            for (unsigned int x = 0; x < simulation.domain->horizontal_nodes; ++x)
            {
                current_node_index = core::access::get_node_index(x, y, simulation.domain->horizontal_nodes);

                if (x == 0 && y == 0)
                {
                    std::cout << "\033[31m";  // Color origin values red
                }

                else if (x == (simulation.domain->horizontal_nodes - 1) && y == (simulation.domain->vertical_nodes - 1))
                {
                    std::cout << "\033[34m";  // Color values of outmost node blue
                }

                else if (phase_information[current_node_index] == -1)
                {
                    std::cout << "\033[32m";  // Color values of ghost nodes green
                }

                else if (phase_information[current_node_index] == 1)
                {
                    std::cout << "\033[33m";  // Color values of solid nodes yellow
                }

                for (auto direction = 0; direction < 9; ++direction)
                {
                    current_values[direction] =
                        distribution_values[A::at(current_node_index, direction, simulation.domain->total_node_count)];
                }

                for (unsigned int i = 0; i < 3; ++i)
                {
                    size_t direction = print_dirs.at(core::access::get_node_index(i, j, 3));
                    if (current_values[direction] >= 0)
                    {
                        std::cout << " ";
                    }
                    std::cout << current_values[direction] << " ";
                }
                std::cout << "\033[0m   ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

// PRINT PHASE INFORMATION ////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Adapted version of `print_vector` that prints out the phase vector of a lattice.
 *          Solid nodes have a phase value of `1` and are colored in yellow.
 *          Fluid nodes have a phase value of `0` and are colored in blue.
 *          Ghost nodes have a phase value of `-1` and are colored in green.
 *
 * @param[in]   vector              the phase vector
 * @param[in]   horizontal_nodes    total amount of horizontal nodes in the simulation domain
 */
void print_phase_vector(const std::vector<int8_t> &vector, const unsigned int horizontal_nodes);

// PRINT MACROSCOPIC OBSERVABLES //////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Prints all velocity values in the lattice to the console. All values are printed in order, i.e. the
 *          origin is located at the lower left corner of the output. This version of this function is supposed
 *          to operate on structures containing the values for multiple time steps, e.g. for debugging purposes.
 *
 * @param[in] properties    a struct containing the extents of the simulation domain
 * @param[in] x_velocities  a vector containing all velocities in x-direction
 * @param[in] y_velocities  a vector containing all velocities in y-direction
 * @param[in] time_step     the time time step considered
 */
void print_velocities(const core::Properties &properties,
                      const std::vector<real_type> &x_velocities,
                      const std::vector<real_type> &y_velocities,
                      const unsigned int time_step = 0);

/**
 * @brief   Prints all density values in the lattice to the console. All values are printed in order, i.e. the
 *          origin is located at the lower left corner of the output. This version of this function is supposed
 *          to operate on structures containing the values for multiple time steps, e.g. for debugging purposes.
 *
 * @param[in] properties    a struct containing the extents of the simulation domain
 * @param[in] densities     a vector containing all density values
 * @param[in] time_step     the time time step considered
 */
void print_densities(const core::Properties &properties,
                     const std::vector<real_type> &densities,
                     const unsigned int time_step = 0);

/**
 * @brief   Prints all velocity values in the lattice to the console. All values are printed in order, i.e. the
 *          origin is located at the lower left corner of the output.
 *
 * @param[in] properties    a struct containing the extents of the simulation domain
 * @param[in] x_velocities  a vector containing all velocities in x-direction
 * @param[in] y_velocities  a vector containing all velocities in y-direction
 */
void print_velocities(const core::Properties &properties,
                      const std::vector<real_type> &x_velocities,
                      const std::vector<real_type> &y_velocities);

/**
 * @brief   Prints all density values in the lattice to the console. All values are printed in order, i.e. the
 *          origin is located at the lower left corner of the output.
 *
 * @param[in] properties    a struct containing the extents of the simulation domain
 * @param[in] densities     a vector containing all density values
 */
void print_densities(const core::Properties &properties, const std::vector<real_type> &densities);

/**
 * @brief   Prints the simulation results, i.e. the velocity vectors and density values, to the console.
 *
 * @param[in]   properties          a struct containing the extents of the simulation domain
 * @param[in]   simulation_results  the structure containing the simulation results
 *                                  (density and component-wise velocity values)
 */
void print_simulation_results(const core::Properties &properties, const core::Results &simulation_results);

/**
 * @brief   Prints the simulation results, i.e. the velocity vectors and density values.
 *
 * @param[in]   properties      a struct containing the extents of the simulation domain
 * @param[in]   densities       a vector containing the density values
 * @param[in]   x_velocities    a vector containing the velocity values in x-direction
 * @param[in]   y_velocities    a vector containing the velocity values in y-direction
 */
void print_simulation_results(const core::Properties &properties,
                              const std::vector<real_type> &densities,
                              const std::vector<real_type> &x_velocities,
                              const std::vector<real_type> &y_velocities);

}  // namespace console

}  // namespace lbm

#endif  // ! LBM_CONSOLE_OUTPUT_HPP
