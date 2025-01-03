/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of crucial functionality of the SYCL lattice Boltzmann simulations.
 * 
 * @version     4.0
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

// LBM exceptions
#include "../exceptions/exceptions.hpp"

// Dependencies on other LBM core features
#include "access.hpp"
#include "constants.hpp"

// Format
#include <fmt/core.h>

// Standary library
#include <complex>
#include <memory>

// SYCL
#include <sycl/sycl.hpp>

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
        inline 
        std::vector<double> maxwell_boltzmann_distribution
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
        inline constexpr unsigned int invert_direction(const unsigned int dir) { return 8 - dir; }

        /**
         * @brief   This enum is used to specify the obstacle that is included in the simulation domain.
         */
        enum Obstacle {NONE, WALLS, CIRCLE, SQUARE, WING, SKYSCRAPER, POROUS, PLATE};

        /**
         * @brief This structure contains all important properties of the simulation.
         */
        struct Properties
        {
            // Algorithmic options
            std::string data_layout;
            std::string algorithm;
            bool debug_mode;
            double relaxation_time;
            bool results_to_csv;
            unsigned int time_steps;
            Obstacle obstacle;

            // Extents of the simulation domain
            unsigned int vertical_nodes;
            unsigned int horizontal_nodes;
            
            /** @brief Total amount of nodes including ghost nodes but excluding buffer nodes */
            unsigned int non_buffered_node_count;

            /** @brief Total amount of nodes including ghost nodes and buffer nodes */
            unsigned int buffered_node_count;

            /** @brief Total amount of node within the actual simulation domain, excluding ghost nodes */
            unsigned int domain_node_count;

            // Inlets
            double inlet_velocity_x;
            double inlet_velocity_y;
            double inlet_density;

            // Outlets
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
                const Obstacle obstacle,
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
         */
        class Results
        {
            private:

            std::shared_ptr<sycl::queue> queue;

            public:

            /**
             * @brief   A unique pointer to a vector containing the densities of all nodes in the simulation domain.
             *          Solid nodes should always have the value '-1.0' for better distinction from the fluid nodes.
             *          Notice that in the case of an incompressible fluid, the density values still vary since these
             *          "virtual" densities are required by the simulation. However, in this case, these density values
             *          are not meaningful, and have to be scaled by a factor to retrieve pressure values. 
             *          However, they are meaningful for compressible fluids.
             */
            std::unique_ptr<std::vector<double>> densities_cpu;

            /** @brief  GPU allocated densities */
            double *densities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the x components of the velocity vectors of all nodes 
             *          in the simulation domain. Solid nodes should always have a zero component and are not differenciated
             *          further regarding their velocities. All differenciation between solid and fluid nodes is realized
             *          through the density values.
             */
            std::unique_ptr<std::vector<double>> x_velocities_cpu;

            /** @brief  GPU allocated x components of the velocity vectors */
            double *x_velocities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the y components of the velocity vectors of all nodes 
             *          in the simulation domain. Solid nodes should always have a zero component and are not differenciated
             *          further regarding their velocities. All differenciation between solid and fluid nodes is realized
             *          through the density values.
             */
            std::unique_ptr<std::vector<double>> y_velocities_cpu;

            /** @brief  GPU allocated y components of the velocity vectors */
            double *y_velocities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the absolutes of the velocity vectors of each node.
             *          It is required for visualization purposes only and remains unused otherwise.
             */
            std::unique_ptr<std::vector<double>> absolute_velocities_cpu;

            /** @brief  GPU allocated absolutes of the velocity vectors */
            double *absolute_velocities_gpu;

            /**
             * @brief   Constructs a new simulation results object based on the provided properties structure.
             *          The internal vectors are initialized with the correct size and filled up with values such as all nodes
             *          were solid.
             * 
             * @param[in] size the size of each vector should be set to the amount of actual nodes (neither ghost nor buffer)
             */
            explicit Results(const size_t &size, sycl::queue &queue);

            ~Results()
            {
                sycl::free(densities_gpu, *queue);
                sycl::free(x_velocities_gpu, *queue);
                sycl::free(y_velocities_gpu, *queue);
                sycl::free(absolute_velocities_gpu, *queue);
            }
        };

        /**
         * @brief This structure contains all data on which the simulation operates internally.
         */
        class Data
        {
            private:

            std::shared_ptr<sycl::queue> queue;

            public:

            int8_t *phase_information;

            double *distribution_values_0;

            double *distribution_values_1;

            /**
             * @brief Constructs a new Data object with an accessor object of the specified type.
             * 
             * @param[in] buffered_node_count the amount of nodes in the lattice including ghosts and buffers.
             */
            explicit Data(const size_t &total_node_count, const sycl::queue &queue, bool two_lattice);

            ~Data()
            {
                sycl::free(phase_information, *queue);
                sycl::free(distribution_values_0, *queue);
                sycl::free(distribution_values_1, *queue);
            }
        };

        /**
         * @brief This structure contains all data that is related to the simulation.
         */
        struct Simulation
        {
            std::unique_ptr<Properties> properties;
            std::unique_ptr<Data> data;
            std::unique_ptr<Results> results;

            /**
             * @brief Constructor for the Simulation struct.
             * @throws `lbm::exceptions::json::PropertyArgumentException` if an unknown data layout is read from the JSON file
             */
            explicit Simulation(sycl::queue &queue);
        };

    } // ! namespace core

} // ! namespace lbm

#endif // ! SIMULATION_HPP
