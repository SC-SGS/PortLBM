// FULLY DEPRECATED

/**
 * @file        collision.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, functionality for the CPU-based collision step is declared and defined.
 * 
 * @version     1.1
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef COLLISION_HPP
#define COLLISION_HPP

// Dependencies on other LBM core features
#include "access.hpp"
#include "simulation.hpp"

namespace lbm
{

    namespace core
    {
        /**
         * @brief This namespace contains the CPU-based implementations for all functions necessary for the collision step.
         */
        namespace collision
        {

            /**
             * @brief   Performs the collision step for the specified node according to the BGK model using its density and velocity value.
             * 
             * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp
             * 
             * @param[in]   simulation_data             a structure containing all data related to the simulation
             * @param[in]   relaxation_time             the relaxation time
             * @param[in]   simulation_results_index    this index is used to access the velocity and density values
             * @param[in]   node                        the index of the node in question
             */
            template <access::experimental::AccessorConcept A> inline void collide_bgk
            (
                const Simulation &simulation,
                const double relaxation_time,
                const unsigned int simulation_results_index,
                const unsigned int node
            )
            {
                std::vector<double> values = access::get_distribution_values_of<A>(*simulation.data->distribution_values_1, node); //careful: explicitly using distribution_values_1 here!
                
                std::vector<double> result = maxwell_boltzmann_distribution
                (
                    (*simulation.results->x_velocities)[simulation_results_index], 
                    (*simulation.results->y_velocities)[simulation_results_index],
                    (*simulation.results->densities)[simulation_results_index]
                );

                for(auto i = 0; i < 9; ++i)
                {
                    result[i] = -(1/relaxation_time) * (values[i] - result[i]) + values[i];
                }

                access::set_distribution_values_of<A>(result, node, *simulation.data->distribution_values_1);
            }

        } // ! namespace collision

    } // ! namespace core

} // ! namespace lbm

#endif // ! COLLISION_HPP