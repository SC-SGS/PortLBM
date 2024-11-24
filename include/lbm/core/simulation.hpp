/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Updated version of the lattice Boltzmann simulation.hpp first introduced in my SimTech project work:
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann.
 *              Initially, this file only contained operations to set up an example domain.
 *              Additional functionality was added to support the updated structure that suits the GPU implementation better. 
 * 
 * @version     2.1
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "../exceptions/exceptions.hpp"

#include "access.hpp"
#include "constants.hpp"
#include "defines.hpp"

#include <fmt/core.h>
#include <complex>

namespace lbm
{

    namespace core
    {

        /**
         * @brief Returns the Maxwell-Boltzmann-Distribution for all directions in the order proposed by Mattila et al.
         * 
         * @param[in] x_velocity    x component of the velocity of the node in question
         * @param[in] y_velocity    y component of the velocity of the node in question
         * @param[in] density       density of the node in question
         * 
         * @return a vector containing the distribution values
         */
        inline std::vector<double> maxwell_boltzmann_distribution
        (
            const double x_velocity, 
            const double y_velocity, 
            const double density
        )
        {
            std::vector<double> result(9,0);
            int velocity_x_component = 0; 
            int velocity_y_component = 0; 

            for(auto direction = 0; direction < 9; ++direction)
            {
                velocity_x_component = (direction % 3) - 1; 
                velocity_y_component = (direction / 3) - 1; 

                result[direction] = constants::weights.at(direction) *
                    (
                        density + 3 * (velocity_x_component * x_velocity + velocity_y_component * y_velocity)
                        + 9.0/2 * std::pow(velocity_x_component * x_velocity + velocity_y_component * y_velocity, 2)
                        - 3.0/2 * (x_velocity * x_velocity + y_velocity * y_velocity)
                    );
            }
            return result;
        }

        /**
         * @brief Returns the inverse direction of that specified.
         */
        inline constexpr unsigned int invert_direction(const unsigned int dir)
        {
            return 8 - dir;
        }

        /**
         * @brief This structure contains all important properties of the simulation.
         *        It is a replacement of the Settings structure used in the project work.
         *        It was replaced owing to the introduced support of a grid decomposition
         *        and a structure-of-array representation of the simulation results.
         */
        struct Properties
        {
            /* Algorithmic options */
            std::string data_layout;
            std::string algorithm;
            bool debug_mode;
            double relaxation_time;
            bool results_to_csv;
            unsigned int time_steps;

            /* Extents of the simulation domain */
            unsigned int vertical_nodes;
            unsigned int horizontal_nodes;
            
            // Total amount of nodes including ghost nodes but excluding buffer nodes
            unsigned int non_buffered_node_count;

            // Total amount of nodes including ghost nodes and buffer nodes
            unsigned int buffered_node_count;

            // Total amount of node within the actual simulation domain, excluding ghost nodes
            unsigned int domain_node_count;

            /* Inlets */
            double inlet_velocity_x;
            double inlet_velocity_y;
            double inlet_density;

            /* Outlets */
            double outlet_velocity_x;
            double outlet_velocity_y;
            double outlet_density;

            /**
             * @brief Constructs a new properties object with the specified parameters.
             */
            explicit Properties
            (
                // Algorithmic properties
                const std::string &&algorithm,
                const std::string &&data_layout,
                const bool debug_mode,
                const bool results_to_csv,
                const double relaxation_time,
                const unsigned int time_steps,
                // Domain properties
                const unsigned int vertical_nodes,
                const unsigned int horizontal_nodes,
                // Inlets
                const double inlet_velocity_x,
                const double inlet_velocity_y,
                const double inlet_density,
                // Outlets
                const double outlet_velocity_x,
                const double outlet_velocity_y,
                const double outlet_density
            ); 

            std::string to_string() const;
        };

        /**
         * TODO: Work in progress.
         * 
         * @brief This structure contains the extents of the expanded simulation domain.
         *        In every case, the ghost node borders are added.
         *        For algorithms with domain decomposition, that is, for all algorithms except the
         *        two-lattice algorithm, parameters regarding the subdomains are stored here.
         * 
         */
        struct ExpandedDomainData
        {
            // Total amount of nodes including ghost nodes and buffer nodes
            unsigned int buffered_node_count;

            /* Extents and amount of subdomains */
            unsigned int subdomain_height;
            unsigned int subdomain_width;
            unsigned int subdomain_count_vertical;
            unsigned int subdomain_count_horizontal;

            explicit ExpandedDomainData
            (
                const unsigned int buffered_node_count,
                const unsigned int subdomain_height,
                const unsigned int subdomain_width,
                const unsigned int subdomain_count_vertical,
                const unsigned int subdomain_count_horizontal
            );
        };

        /**
         * @brief This structure contains the results of the simulation in a structure-of-arrays representation.
         *        It is a replacement of the sim_data_tuple used in the project work. 
         *        
         *        Notice that this structure stores unique pointers to vectors and not the vectors themselves.
         *        It is not recommended to resize the vectors as may require reallocation and may thus invalidate 
         *        the pointer. This can lead to poor predictability. If you want to resize the vectors, set the 
         *        pointers to new vectors of the desired size or construct a new SimulationResults object.
         *        
         *        Results are stored time-step-wise, that is, the values are stored according to the order of
         *        the node index for one time step and then for another. Hence, to properly access the results,
         *        the accessor must know the total amount of nodes including buffers.
         */
        struct SimulationResults
        {
            /**
             * @brief A unique pointer to a vector containing the densities of all nodes in the simulation domain.
             *        Solid nodes should always have the value '-1.0' for better distinction from the fluid nodes.
             *        Notice that in the case of an incompressible fluid, the density values still vary since these
             *        "virtual" densities are required by the simulation. Hence, in this case, these density values
             *        are not meaningful. However, they are meaningful for compressible fluids.
             */
            std::unique_ptr<std::vector<double>> densities;

            /**
             * @brief A unique pointer to a vector containing the pressure values of all nodes in the simulation domain.
             *        Solid nodes should always have the value '-1.0' for better distinction from the fluid nodes.
             *        Notice that unlike the density values, the pressure values are meaningful for both compressible
             *        and incompressible fluids.
             */
            std::unique_ptr<std::vector<double>> pressures;

            /**
             * @brief A unique pointer to a vector containing the x components of the velocity vectors of all nodes 
             *        in the simulation domain. Solid nodes should always have a zero component and are not differenciated
             *        further regarding their velocities. All differenciation between solid and fluid nodes is realized
             *        through the density values.
             */
            std::unique_ptr<std::vector<double>> x_velocities;

            /**
             * @brief A unique pointer to a vector containing the y components of the velocity vectors of all nodes 
             *        in the simulation domain. Solid nodes should always have a zero component and are not differenciated
             *        further regarding their velocities. All differenciation between solid and fluid nodes is realized
             *        through the density values.
             */
            std::unique_ptr<std::vector<double>> y_velocities;

            std::unique_ptr<std::vector<double>> absolute_velocities;

            /**
             * @brief Constructs a new simulation results structure based on the provided properties structure.
             *        The internal vectors are initialized with the correct size and filled up with values such as all nodes
             *        were solid.
             * 
             * @param properties this structure of properties defines the total buffered node count and the number of time steps.
             */
            explicit SimulationResults(const size_t &size);

            explicit SimulationResults
            (
                const std::vector<double> &densities,
                const std::vector<double> &pressures,
                const std::vector<double> &x_velocities,
                const std::vector<double> &y_velocities,
                const std::vector<double> &absolute_velocities
            );
        };

        /**
         * @brief This structure contains all data on which the simulation operates internally.
         * 
         * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::core::access::LBMAccessorObject`
         */
        template <class T> struct SimulationData
        {
            std::unique_ptr<std::vector<uint8_t>> phase_information;
            std::unique_ptr<std::vector<uint8_t>> is_buffer;
            std::unique_ptr<std::vector<double>> distribution_values_0;
            std::unique_ptr<std::vector<double>> distribution_values_1;
            std::unique_ptr<std::vector<unsigned int>> boundary_interactions;

            unsigned int end_node_index_non_buffered;
            unsigned int end_node_index_buffered;

            std::unique_ptr<T> lbm_accessor;

            /**
             * @brief Constructs a new SimulationData object with an accessor object of the specified type.
             * 
             * @param properties the extents of the lattice specified in this Properties structure are used for initialization
             */
            explicit SimulationData<T>
            (
                const Properties &properties
            )
            :
            phase_information(std::make_unique<std::vector<uint8_t>>(properties.buffered_node_count, 0)),
            is_buffer(std::make_unique<std::vector<uint8_t>>(properties.buffered_node_count, 0)),
            distribution_values_0(std::make_unique<std::vector<double>>(properties.buffered_node_count * 9, 0.0f)),
            distribution_values_1(std::make_unique<std::vector<double>>(properties.buffered_node_count * 9, 0.0f)),
            boundary_interactions(std::make_unique<std::vector<unsigned int>>(properties.buffered_node_count * 9, 0)),
            end_node_index_non_buffered(properties.non_buffered_node_count),
            end_node_index_buffered(properties.buffered_node_count),
            lbm_accessor(std::make_unique<T>(properties.horizontal_nodes, properties.buffered_node_count))
            {};
        };

        /**
         * @brief Specialized version of the SimulationData structure for use with a collision accessor object.
         *        The only difference is that the collision accessor object requires one less parameter for its constructor.
         */
        template<> struct SimulationData<access::LBMCollisionAccessor>
        {
            std::unique_ptr<std::vector<uint8_t>> phase_information;
            std::unique_ptr<std::vector<uint8_t>> is_buffer;
            std::unique_ptr<std::vector<double>> distribution_values_0;
            std::unique_ptr<std::vector<double>> distribution_values_1;
            std::unique_ptr<std::vector<unsigned int>> boundary_interactions;

            unsigned int end_node_index_non_buffered;
            unsigned int end_node_index_buffered;

            std::unique_ptr<access::LBMCollisionAccessor> lbm_accessor;

            /**
             * @brief Constructs a new SimulationData object with an accessor object of the specified type.
             * 
             * @param properties the extents of the lattice specified in this Properties structure are used for initialization
             */
            explicit SimulationData<access::LBMCollisionAccessor>
            (
                const Properties &properties
            )
            :
            phase_information(std::make_unique<std::vector<uint8_t>>(properties.buffered_node_count, 0)),
            is_buffer(std::make_unique<std::vector<uint8_t>>(properties.buffered_node_count, 0)),
            distribution_values_0(std::make_unique<std::vector<double>>(properties.buffered_node_count, 0.0f)),
            distribution_values_1(std::make_unique<std::vector<double>>(properties.buffered_node_count, 0.0f)),
            boundary_interactions(std::make_unique<std::vector<unsigned int>>(properties.buffered_node_count * 9, 0)),
            end_node_index_non_buffered(properties.non_buffered_node_count),
            end_node_index_buffered(properties.buffered_node_count),
            lbm_accessor(std::make_unique<access::LBMCollisionAccessor>(properties.horizontal_nodes))
            {};
        };

    } // ! namespace core

} // ! namespace lbm

#endif // ! SIMULATION_HPP
