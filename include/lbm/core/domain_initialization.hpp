/**
 * @file        domain_initialization.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, functionality for setting up the simulation domain is declared and defined.
 * 
 * @version     1.0
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef DOMAIN_INITIALIZATION_HPP
#define DOMAIN_INITIALIZATION_HPP

#include "simulation.hpp"
#include "boundaries.hpp"

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
        template<access::experimental::LBMAccessor T> void setup_pipe_flow_environment
        (
            const Properties &properties,
            Data &data
        )
        {
            std::vector<double> values = maxwell_boltzmann_distribution(0, 0, 1);
            for(auto i = 0; i < properties.end_node_index_buffered; ++i)
            {
                access::set_distribution_values_of<T>(values, i, *simulation_data.distribution_values_0);
            }    

            boundary_conditions::update_velocity_input_density_output(properties, *simulation_data.distribution_values_0);

            /* Phase information vector */
            for(auto x = 1; x < properties.horizontal_nodes - 1; ++x)
            {
                (*simulation_data.phase_information)[access::get_node_index(x,1, properties.horizontal_nodes)] = true;
                (*simulation_data.phase_information)[access::get_node_index(x,properties.vertical_nodes - 2, properties.horizontal_nodes)] = true;
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