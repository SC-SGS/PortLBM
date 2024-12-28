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
        enum Obstacle {NONE, WALLS, CIRCLE};

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

            boundary_conditions::update_velocity_input_density_output<A>(*simulation.properties, *simulation.data->distribution_values_0);

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
                int radius = simulation.properties->vertical_nodes / 6;

                for(auto y = middle_y - 2*radius; y <= middle_y + 2*radius; ++y)
                {
                    for(auto x = middle_x - 2*radius; x <= middle_x + 2*radius; ++x)
                    {
                        if(sqrt(abs(middle_x-x) * abs(middle_x-x) + abs(middle_y-y) * abs(middle_y-y) ) <= sqrt(radius * radius))
                            (*simulation.data->phase_information)[access::get_node_index(x, y, simulation.properties->horizontal_nodes)] = 1;
                    }
                }
            }
        }

    } // ! namespace core

} // ! namespace lbm

#endif // ! DOMAIN_INITIALIZATION_HPP