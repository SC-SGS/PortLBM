/**
 * @file        console_output.cpp
 *
 * @author      Marcel Graf
 *
 * @brief       This source file contains the definitions of various functions for console outputs.
 *
 * @version     1.7
 *
 * @date        March 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#include "../../../include/lbm/console/console_output.hpp"

void lbm::console::print_phase_vector(const std::vector<int8_t> &vector, const unsigned int horizontal_nodes)
{
    unsigned int vertical_nodes = vector.size() / horizontal_nodes;
    int number = 0;

    for (auto y = vertical_nodes; y-- > 0;)
    {
        for (auto x = 0; x < horizontal_nodes; ++x)
        {
            number = static_cast<int>(vector[core::access::get_node_index(x, y, horizontal_nodes)]);
            std::cout << " ";
            if (number == 1)
            {
                std::cout << "\033[33m ";
            }
            else if (number == 0)
            {
                std::cout << "\033[34m ";
            }
            else
            {
                std::cout << "\033[32m";
            }

            std::cout << number << "\033[0m ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void lbm::console::print_velocities(const core::Properties &properties,
                                    const std::vector<real_type> &x_velocities,
                                    const std::vector<real_type> &y_velocities,
                                    const unsigned int time_step)
{
    unsigned int index = 0;

    for (auto y = properties.vertical_nodes - 1; y-- > 1;)
    {
        for (auto x = 1; x < properties.horizontal_nodes - 1; ++x)
        {
            index = core::access::get_result_index(
                x, y, properties.horizontal_nodes, properties.domain_node_count, time_step);

            std::cout << "(";
            if (x_velocities[index] >= 0)
            {
                std::cout << " ";
            }
            std::cout << x_velocities[index] << ", ";
            if (y_velocities[index] >= 0)
            {
                std::cout << " ";
            }
            std::cout << y_velocities[index] << ")";
            std::cout << "\033[0m    ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void lbm::console::print_velocities(const core::Properties &properties,
                                    const std::vector<real_type> &x_velocities,
                                    const std::vector<real_type> &y_velocities)
{
    unsigned int index = 0;

    for (auto y = properties.vertical_nodes - 1; y-- > 1;)
    {
        for (auto x = 1; x < properties.horizontal_nodes - 1; ++x)
        {
            index = core::access::get_result_index(x, y, properties.horizontal_nodes);

            std::cout << "(";
            if (x_velocities[index] >= 0)
            {
                std::cout << " ";
            }
            std::cout << x_velocities[index] << ", ";
            if (y_velocities[index] >= 0)
            {
                std::cout << " ";
            }
            std::cout << y_velocities[index] << ")";
            std::cout << "\033[0m    ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void lbm::console::print_densities(
    const core::Properties &properties, const std::vector<real_type> &densities, const unsigned int time_step)
{
    unsigned int node_index = 0;
    real_type value = 0;

    for (auto y = properties.vertical_nodes - 1; y-- > 1;)
    {
        for (auto x = 1; x < properties.horizontal_nodes - 1; ++x)
        {
            if (x == 0 && y == 0)
            {
                std::cout << "\033[31m";
            }
            else if (x == (properties.horizontal_nodes - 1) && y == (properties.vertical_nodes - 1))
            {
                std::cout << "\033[34m";
            }

            value = densities[core::access::get_result_index(
                x, y, properties.horizontal_nodes, properties.domain_node_count, time_step)];

            if (value >= 0)
            {
                std::cout << " ";
            }
            std::cout << value;
            std::cout << "\033[0m  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void lbm::console::print_densities(const core::Properties &properties, const std::vector<real_type> &densities)
{
    unsigned int node_index = 0;
    real_type value = 0;

    for (auto y = properties.vertical_nodes - 1; y-- > 1;)
    {
        for (auto x = 1; x < properties.horizontal_nodes - 1; ++x)
        {
            if (x == 0 && y == 0)
            {
                std::cout << "\033[31m";
            }
            else if (x == (properties.horizontal_nodes - 1) && y == (properties.vertical_nodes - 1))
            {
                std::cout << "\033[34m";
            }

            value = densities[core::access::get_result_index(x, y, properties.horizontal_nodes)];
            if (value >= 0)
            {
                std::cout << " ";
            }
            std::cout << value;
            std::cout << "\033[0m  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void lbm::console::print_simulation_results(const core::Properties &properties, const core::Results &simulation_results)
{
    std::cout << "Velocity values: \n\n";

    for (auto i = 0; i < properties.time_steps; ++i)
    {
        std::cout << "t = " << i << "\n";
        std::cout << "-------------------------------------------------------------------------------- \n";
        lbm::console::print_velocities(
            properties, *simulation_results.x_velocities_cpu, *simulation_results.y_velocities_cpu, i);
        std::cout << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Density values: \n\n";

    for (auto i = 0; i < properties.time_steps; ++i)
    {
        std::cout << "t = " << i << "\n";
        std::cout << "-------------------------------------------------------------------------------- \n";
        lbm::console::print_densities(properties, *simulation_results.densities_cpu, i);
        std::cout << "\n";
    }
    std::cout << "\n";
}

void lbm::console::print_simulation_results(const core::Properties &properties,
                                            const std::vector<real_type> &densities,
                                            const std::vector<real_type> &x_velocities,
                                            const std::vector<real_type> &y_velocities)
{
    std::cout << "Velocity values: \n\n";

    for (auto i = 0; i < properties.time_steps; ++i)
    {
        std::cout << "t = " << i << "\n";
        std::cout << "-------------------------------------------------------------------------------- \n";
        lbm::console::print_velocities(properties, x_velocities, y_velocities, i);
        std::cout << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Density values: \n\n";

    for (auto i = 0; i < properties.time_steps; ++i)
    {
        std::cout << "t = " << i << "\n";
        std::cout << "-------------------------------------------------------------------------------- \n";
        lbm::console::print_densities(properties, densities, i);
        std::cout << "\n";
    }
    std::cout << "\n";
}
