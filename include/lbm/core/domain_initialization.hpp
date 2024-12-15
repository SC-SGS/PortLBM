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

        /**
         * @brief Sets the up pipe flow environment object with a fluid in an equilibrium non-moving state.
         *        The domain consists of solid nodes on the upper and lower boundary and fluid nodes otherwise.
         * 
         * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
         * 
         * @param[in]       properties      a struct containing the densities and velocities of the inlets and outlets
         * @param[in, out]  simulation_data a structure of data on which the simulation operates
         */
        template<access::experimental::AccessorConcept A> 
        void setup_pipe_flow_environment(const Simulation &simulation)
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
                (*simulation.data->phase_information)[access::get_node_index(x, 1, simulation.properties->horizontal_nodes)] = true;
                (*simulation.data->phase_information)[access::get_node_index(x, simulation.properties->vertical_nodes - 2, simulation.properties->horizontal_nodes)] = true;
            }

            // /* Add wall */
            // for(auto y = 2; y < 0.5 * (properties.vertical_nodes - 1); ++y)
            // {
            //     (*simulation_data.phase_information)[lbm::access::get_node_index(properties.horizontal_nodes / 2,y, properties.horizontal_nodes)] = true;
            //     (*simulation_data.phase_information)[lbm::access::get_node_index(properties.horizontal_nodes / 2,y, properties.horizontal_nodes)] = true;
            // }

        }

    } // ! namespace core

} // ! namespace lbm

#endif // ! DOMAIN_INITIALIZATION_HPP