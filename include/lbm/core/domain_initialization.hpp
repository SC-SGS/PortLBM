/**
 * @file        domain_initialization.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, functionality for setting up the simulation domain is declared and defined.
 * 
 * @version     1.1
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef DOMAIN_INITIALIZATION_HPP
#define DOMAIN_INITIALIZATION_HPP

// Dependencies on other LBM core features
#include "boundaries.hpp"
#include "simulation.hpp"

namespace lbm
{

    namespace core
    {
        enum Obstacle {NONE, WALLS, CIRCLE, SQUARE, WING, SKYSCRAPER, POROUS, PLATE};

        /**
         * @brief Sets the up pipe flow environment object with a fluid in an equilibrium non-moving state.
         *        The domain consists of solid nodes on the upper and lower boundary and fluid nodes otherwise.
         * 
         * @tparam A any `core::access::experimental::AccessorConcept` from access.hpp
         * 
         * @param[in, out]  simulation  reference to the structure containing all simulation data 
         * @param[in]       obstacle    what kind of obstacle is added to the domain
         */
        template<access::experimental::AccessorConcept A> 
        void setup_pipe_flow_environment(const Simulation &simulation, const Obstacle obstacle = NONE)
        {
            std::vector<double> values = maxwell_boltzmann_distribution(0, 0, 1);
            for(auto i = 0; i < simulation.properties->buffered_node_count; ++i)
            {
                access::set_distribution_values_of<A>(values, i, simulation.properties->buffered_node_count, *simulation.data->distribution_values_0);
            }

            boundary_conditions::update_velocity_input_density_output<A>(simulation);

            /* Phase information vector */
            for(auto x = 1; x < simulation.properties->horizontal_nodes - 1; ++x)
            {
                (*simulation.data->phase_information)[access::get_node_index(x, 1, simulation.properties->horizontal_nodes)] = 1;
                (*simulation.data->phase_information)[access::get_node_index(x, simulation.properties->vertical_nodes - 2, simulation.properties->horizontal_nodes)] = 1;
            }

            if(obstacle == WALLS)
            {
                for(auto y = 2; y < 0.5 * (simulation.properties->vertical_nodes - 1); ++y)
                {
                    (*simulation.data->phase_information)[access::get_node_index(simulation.properties->horizontal_nodes / 4,y, simulation.properties->horizontal_nodes)] = 1;
                    (*simulation.data->phase_information)[access::get_node_index(simulation.properties->horizontal_nodes / 4,y, simulation.properties->horizontal_nodes)] = 1;
                }

                for(auto y = simulation.properties->vertical_nodes - 1; y >= 0.5 * (simulation.properties->vertical_nodes - 1); --y)
                {
                    (*simulation.data->phase_information)[access::get_node_index(simulation.properties->horizontal_nodes / 2,y, simulation.properties->horizontal_nodes)] = 1;
                    (*simulation.data->phase_information)[access::get_node_index(simulation.properties->horizontal_nodes / 2,y, simulation.properties->horizontal_nodes)] = 1;
                }

                for(auto y = 2; y < 0.5 * (simulation.properties->vertical_nodes - 1); ++y)
                {
                    (*simulation.data->phase_information)[access::get_node_index(3 * simulation.properties->horizontal_nodes / 4, y, simulation.properties->horizontal_nodes)] = 1;
                    (*simulation.data->phase_information)[access::get_node_index(3 * simulation.properties->horizontal_nodes / 4, y, simulation.properties->horizontal_nodes)] = 1;
                }
            }
            else if(obstacle == CIRCLE)
            {
                int middle_x =  2 * simulation.properties->horizontal_nodes / 10;
                int middle_y = simulation.properties->vertical_nodes / 2;
                int radius = simulation.properties->vertical_nodes / 10;

                for(auto y = middle_y - 2 * radius; y <= middle_y + 2 * radius; ++y)
                {
                    for(auto x = middle_x - 2 * radius; x <= middle_x + 2 * radius; ++x)
                    {
                        if(sqrt(abs(middle_x-x) * abs(middle_x-x) + abs(middle_y-y) * abs(middle_y-y) ) <= sqrt(radius * radius))
                            (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }
            }
            else if(obstacle == SQUARE)
            {
                int middle_x =  2 * simulation.properties->horizontal_nodes / 10;
                int middle_y = simulation.properties->vertical_nodes / 2;
                int length = simulation.properties->vertical_nodes / 5;

                for(auto y = middle_y - 0.5 * length; y <= middle_y + 0.5 * length; ++y)
                {
                    for(auto x = middle_x - 0.5 * length; x <= middle_x + 0.5 * length; ++x)
                    {
                        (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }
            }
            else if(obstacle == PLATE)
            {
                int middle_x =  2 * simulation.properties->horizontal_nodes / 10;
                int middle_y = simulation.properties->vertical_nodes / 2;
                int length = simulation.properties->vertical_nodes / 5;

                for(auto y = middle_y - 0.5 * length; y <= middle_y + 0.5 * length; ++y)
                {
                    for(auto x = middle_x - 0.05 * length; x <= middle_x + 0.05 * length; ++x)
                    {
                        (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }
            }
            else if(obstacle == SKYSCRAPER)
            {
                int height_antenna = simulation.properties->vertical_nodes / 2;
                int width_antenna = 2;

                int height_second = 2 * simulation.properties->vertical_nodes / 5;
                int width_second = 0.5 * simulation.properties->vertical_nodes / 6;

                int height_first = simulation.properties->vertical_nodes / 4;
                int width_first = 0.5 * simulation.properties->vertical_nodes / 4;

                int middle_x =  3 * simulation.properties->horizontal_nodes / 10;

                for(auto y = 0; y <= height_first; ++y)
                {
                    for(auto x = middle_x - 0.5 * width_first; x <= middle_x + 0.5 * width_first; ++x)
                    {
                        (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }

                for(auto y = height_first + 1; y <= height_second; ++y)
                {
                    for(auto x = middle_x - 0.5 * width_second; x <= middle_x + 0.5 * width_second; ++x)
                    {
                        (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }

                for(auto y = height_second + 1; y <= height_antenna; ++y)
                {
                    for(auto x = middle_x - 0.5 * width_antenna; x <= middle_x + 0.5 * width_antenna; ++x)
                    {
                        (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }
            }
            else if(obstacle == WING)
            {
                int middle_x =  2 * simulation.properties->horizontal_nodes / 10;
                int middle_y = simulation.properties->vertical_nodes / 2;
                int radius_max = simulation.properties->vertical_nodes / 5;
                int scale_x = 1;
                int scale_y = 5;

                for(int y = middle_y - radius_max; y <= middle_y + radius_max; ++y)
                {
                    for(int x = middle_x - radius_max; x <= middle_x; ++x)
                    {
                        if(0.2 * abs(x - middle_x) * abs(x - middle_x) + abs(y - middle_y) * abs(y - middle_y) <= radius_max)
                        {
                            (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                        }
                            
                    }
                }

                for(auto y = middle_y - 2 * radius_max; y <= middle_y + 2 * radius_max; ++y)
                {
                    for(auto x = middle_x; x <= simulation.properties->horizontal_nodes - 1; ++x)
                    {
                        if(std::pow(abs(middle_x - x), 1.4) + scale_y * scale_y * abs(middle_y - y) * abs(middle_y - y) <= scale_y * scale_y * radius_max)
                        {
                            (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                        }
                            
                    }
                }
            }
            else if(obstacle == POROUS)
            {
                for(int x = 0; x < simulation.properties->horizontal_nodes; ++x)
                {
                    
                    for(int y = 0; y < simulation.properties->vertical_nodes; ++y)
                    {
                        int one = 1 + ( std::rand() % 20 );
                        int two = 1 + ( std::rand() % 20 );
                        if(x % one == one-1 && y % two == two-1)
                            (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }
            }
        }

    } // ! namespace core

} // ! namespace lbm

#endif // ! DOMAIN_INITIALIZATION_HPP