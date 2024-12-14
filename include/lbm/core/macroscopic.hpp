/**
 * @file        macroscopic.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations and definitions of functions for calculating the velocity, 
 *              density and the pressure of a node following the D2Q9I model.
 * 
 * @version     1.2
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef MACROSCOPIC_HPP
#define MACROSCOPIC_HPP

#include "defines.hpp"

#include <list>
#include <numeric>

namespace lbm
{

    namespace core
    {

        /**
         * @brief This namespace includes functions for calculating macroscopic observables of non-boundary nodes 
         *        for the D2Q9I model.
         */
        namespace macroscopic
        {

            /**
             * @brief Calculates the density of a fluid node.
             * 
             * @param[in] distribution_values a vector containing all distribution functions of the respective fluid node
             * 
             * @return the density calculated from the provided distribution_values
             */
            inline double density(const std::vector<double> &distribution_values)
            {
                return std::accumulate(distribution_values.begin(), distribution_values.end(), 0.0);
            }

            /**
             * @brief Calculates the flow velocity of a fluid node.
             * 
             * @param[in] distribution_values a vector containing all distribution functions of the respective fluid node
             * 
             * @return a lbm::velocity representing the flow velocity
             */
            inline std::array<double, 2> flow_velocity(const std::vector<double> &distribution_values)
            {
                std::array<double, 2> flow_velocity{0,0};
                std::array<double, 2> velocity_vector{0,0};

                int velocity_x_component = 0; 
                int velocity_y_component = 0; 
                
                for(int i = 0; i < 9; ++i)
                {
                    velocity_x_component = i % 3 - 1; 
                    velocity_y_component = i / 3 - 1; 
                    flow_velocity[0] += distribution_values.at(i) * velocity_x_component;
                    flow_velocity[1] += distribution_values.at(i) * velocity_y_component;
                }
                return flow_velocity;
            }

        } // ! namespace macroscopic

    } // ! namespace core

} // ! namespace lbm

#endif // ! MACROSCOPIC_HPP