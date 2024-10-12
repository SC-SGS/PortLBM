/**
 * @file        console_output.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations of various functions for console outputs.
 * 
 * @version     1.0
 * 
 * @date        2024-10-10
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef CONSOLE_OUTPUT_HPP
#define CONSOLE_OUTPUT_HPP

#include "../core/access.hpp"
#include "../core/simulation.hpp"

#include <fmt/core.h>
#include <fmt/color.h>

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace lbm
{

    /**
     * @brief This namespace contains various functions for printing information to the console.
     */
    namespace console
    {

        /**
         * @brief Prints the message to the console that mentions the usage of ANSI colors and the implications.
         */
        inline void print_ansi_color_message()
        {
            fmt::print
            (
                "This program utilizes ANSI color codes to output colored text. \n" 
                "If your terminal does not support those codes, your output may be corrupted.\n"
                "Please also make sure that the color scheme of your terminal supports the used colors.\n"
                "The color names listed below should be colored.\n\n" 
                "Used colors:\t\033[31mred\033[0m, \033[33myellow\033[0m, \033[32mgreen\033[0m, \033[34mblue\033[0m.\n\n"
                "Notice:\t\tSince the text is set to its default format (\\033[0m) multiple times in this program,\n"
                "\t\tformatted text, especially from exceptions or third party code, may not have its typical color.\n\n"
            );

            std::string note_to_myself = fmt::format(fg(fmt::rgb(255, 0, 0)), "Note to myself: the fmt library also supports colored output...\n\n");
            fmt::print("{}", note_to_myself);
        }

        /**
         * @brief Prints the legend explaining the meaning of the colors used in the console output.
         */
        inline void print_color_legend()
        {
            fmt::print
            (
                "Legend:\n"
                "-------------------------------------------------------------------------------\n"
                "\033[31mRED\033[0m:\tValues of the node in the origin\n"
                "\033[33mYELLOW\033[0m:\tMilestones and important events\n"
                "\033[32mGREEN\033[0m:\t1.) Distribution values of buffer nodes or ghost nodes\n"
                "\t2.) Phase illustration: \033[32m#\033[0m marks a solid node\n"
                "\033[34mBLUE\033[0m:\t1.) Distribution values of the \"outmost\" node\n"
                "\t2.) Phase illustration: \033[34m~\033[0m marks a fluid node\n\n"
            );
        }

        /**
         * @brief Allows to print out a vector representing a data layout whose column count is HORIZONTAL_NODES.
         *        Notice that the vector is assumed to represent a matrix.
         * 
         * @tparam T the type of the objects the specified vector holds (must be numeric)
         * @param vector the vector that is to be printed in the console
         */
        template <typename T> void print_vector
        (
            const std::vector<T> &vector,
            const unsigned int horizontal_nodes
        )
        {
            unsigned int vertical_nodes = vector.size() / horizontal_nodes;

            for(auto y = vertical_nodes; y-- > 0; )
            {
                for(auto x = 0; x < horizontal_nodes; ++x)
                {
                    if(x == 0 && y == 0) std::cout << "\033[31m";
                    else if(x == (horizontal_nodes - 1) && y == (vertical_nodes -1)) std::cout << "\033[34m";
                    std::cout << vector[lbm::access::get_node_index(x, y, horizontal_nodes)]; 
                    std::cout << "\t\033[0m";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        /**
         * @brief Prints all distribution values in to the console.
         *        They are displayed in the original order, i.e. the origin is located at the lower left corner of the printed distribution chart.
         * 
         * @param distribution_values a vector containing the distribution values of all nodes 
         * @param access_function the function used to access the distribution values
         */
        template <class T> inline void print_distribution_values
        (
            const std::vector<double> &distribution_values, 
            const T &lbm_accessor
        )
        {
            static_assert(
                std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                "Template class must have base class lbm::access::LBMAccessorObject."
            );

            std::vector<std::vector<unsigned int>> print_dirs = {{6,7,8}, {3,4,5}, {0,1,2}};
            std::vector<double> current_values(9,0);

            unsigned int current_node_index = 0;
            unsigned int vertical_nodes = distribution_values.size() / (9 * lbm_accessor.horizontal_nodes); 
            std::cout << std::setprecision(3) << std::fixed;

            for(auto y = vertical_nodes; y-- > 0; )
            {
                for(auto i = 0; i < 3; ++i)
                {
                    auto current_row = print_dirs[i];
                    for(auto x = 0; x < lbm_accessor.horizontal_nodes; ++x)
                    {
                        if(x == 0 && y == 0) std::cout << "\033[31m";
                        else if(x == (lbm_accessor.horizontal_nodes - 1) && y == (vertical_nodes -1)) std::cout << "\033[34m";
                        current_node_index = lbm::access::get_node_index(x, y, lbm_accessor.horizontal_nodes);
                        current_values = lbm::access::get_distribution_values_of(distribution_values, current_node_index, lbm_accessor);

                        for(auto j = 0; j < 3; ++j)
                        {
                            auto direction = current_row[j];
                            std::cout << current_values[direction] << "  ";
                        }
                        std::cout << "\t\033[0m";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n\n";
            }
        } 

        /**
         * @brief Adapted version of print_vector that prints out the phase vector of a lattice.
         *        If a node is solid (i.e. the entry is true), it is represented by #.
         *        If a node is fluid (i.e. the entry is false), it is represented by ~.
         * 
         * @param vector the phase vector
         */
        void print_phase_vector
        (
            const std::vector<bool> &vector,
            const unsigned int horizontal_nodes
        );

        /**
         * @brief Prints all velocity values in the lattice to the console.
         *        All values are printed in order, i.e. the origin is located at the lower left corner of the output.
         * 
         * @param vector a vector containing all velocity values
         */
        void print_velocities
        (
            const lbm::Properties &properties,
            const std::vector<double> &x_velocities, 
            const std::vector<double> &y_velocities,
            const unsigned int time_step
        );

        void print_densities
        (
            const lbm::Properties &properties,
            const std::vector<double> &densities,
            const unsigned int time_step
        );

        /**
         * @brief Prints the simulation results, i.e. the velocity vectors and density values, for all time steps.
         * 
         * @param results a vector containing the simulation data tuples.
         */
        void print_simulation_results
        (
            const lbm::Properties &properties,
            const lbm::SimulationResults &simulation_results
        );

    /**
     * @brief Prints various pieces of debug information to the console.
     *        Included information:
     *        
     *        - Enumeration of all nodes within the lattice
     * 
     *        - Phase information
     * 
     *        - Border swap information
     * 
     *        - Distribution values 0
     * 
     * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
     * 
     * @param[in] simulation_data   a structure of data on which the simulation operates
     * @param[in] swap_info         a lbm::border_swap_information
     */
    template <class T> void debug_prints
    (
        const SimulationData<T> &simulation_data,
        const lbm::border_swap_information &swap_info
    )
    {
        static_assert(
            std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
            "Template class must have base class lbm::access::LBMAccessorObject.");

        std::vector<unsigned int> nodes;
        for(auto i = 0; i < simulation_data.end_node_index_buffered; ++i)
        {
            nodes.push_back(i);
        }

        std::cout << "Enumeration of all nodes within the lattice: \n"
                << "-------------------------------------------------------------------------------\n";

        lbm::console::print_vector(nodes, simulation_data.lbm_accessor->horizontal_nodes);
        std::cout << "\n";

        std::cout << "Phases: \n"
                << "-------------------------------------------------------------------------------\n";
        lbm::console::print_phase_vector(*simulation_data.phase_information, simulation_data.lbm_accessor->horizontal_nodes);
        std::cout << "\n";

        std::cout << "Border swap information: \n"
                << "-------------------------------------------------------------------------------\n";
        for(const auto& current : swap_info)
            lbm::console::print_vector(current, current.size());
        std::cout << "\n";

        std::cout << "Distribution values: \n"
                << "-------------------------------------------------------------------------------\n";
        lbm::console::print_distribution_values(*simulation_data.distribution_values_0, *simulation_data.lbm_accessor);
        std::cout << "\n";
    }

    } // ! namespace console

} // ! namespace lbm

#endif // ! CONSOLE_OUTPUT_HPP
