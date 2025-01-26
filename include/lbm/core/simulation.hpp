/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of crucial functionality of the SYCL lattice Boltzmann 
 *              simulations.
 * 
 * @version     4.1
 * 
 * @date        January 2025
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

// HPX
#include <hpx/chrono.hpp>

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
         * @brief This structure contains all important properties of the simulation.
         */
        struct Properties
        {
            // Algorithmic options

            std::string data_layout;
            std::string algorithm;
            std::string scenario;
            bool debug_mode;
            double relaxation_time;
            bool results_to_csv;
            unsigned int time_steps;

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
                // Domain properties
                const std::string &&obstacle,
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
         * @brief   This class offers options to control an algorithm and to query its progress and performance data.
         */
        class Control
        {
            private:

            bool stopped;

            unsigned int current_iteration;
            unsigned int max_iterations;
            double progress;

            std::unique_ptr<hpx::chrono::high_resolution_timer> timer;
            double last_frametime;

            public:

            /**
             * @brief Constructs a Control object that covers the specified maximum number of iterations.
             * 
             * @param[in]   max_iterations  the maximum iteration executed by the algorithm 
             */
            explicit Control(const unsigned int max_iterations);

            /**
             * @brief   Forbids the execution of further iteration even if the maximum iteration count is not reached.
             *          The current iteration is always finished.
             */
            inline void forbid_execution() { stopped = true; }

            /**
             * @brief   Allows the algorithm to be executed if the maximum iteration count is not reached.
             */
            inline void allow_execution() { stopped = false; }

            /**
             * @brief   Checks whether the controlled algorithm is allowed to be executed.
             * 
             * @return  true    if the algorithm is neither stopped manually nor at its final iteration
             * @return  false   if the algorithm is stopped manually or at its final iteration
             */
            inline bool is_execution_allowed() const 
            { return (!stopped) && (current_iteration < max_iterations); }

            /**
             * @brief   Returns whether or not the controlled algorithm has reached is final iteration.
             */
            inline bool is_finished() const { return current_iteration == max_iterations; }

            /**
             * @brief   Returns whether the execution of the controlled algorithm has been forbidden.
             */
            inline bool is_paused() const { return stopped; }

            /**
             * @brief   Calculates the progress and performance metrics as preparation for the next iteration.
             */
            inline void finalize_iteration() 
            { 
                current_iteration++; 
                progress = (double)current_iteration / (double)max_iterations;
                last_frametime = timer->elapsed_microseconds() * 0.001;
            }

            /**
             * @brief   Resets the timer that tracks the frametimes.
             */
            inline void reset_timer() { timer->restart(); }

            /**
             * @brief   Returns the number of the currently processed iteration.
             */
            inline unsigned int get_current_iteration() const { return current_iteration; }

            /**
             * @brief   Returns the progress of the controlled algorithm, that is, which fraction of the total amount 
             *          of iterations it has already completed.
             */
            inline double get_progress() const { return progress; }

            /**
             * @brief   Returns the last frametime of the controlled algorithm, that is, how long the completion of
             *          the last iteration took.
             */
            inline double get_last_frametime() const { return last_frametime; }
        };

        /**
         * @brief   This namespace contains convenience functions for determining which powers of certain bases
         *          specified numbers are.
         */
        namespace power_functions
        {
            /**
             * @brief   Returns whether the specified number is a power of four.
             * 
             * @param[in]   x   the number in question 
             */
            inline bool is_power_of_4(size_t x)
            {
                size_t checkbit = 3;
                if (x & 0x1) return false;
                while (!(checkbit & x)) x >>= 2;
                return (!(x & 2));
            }

            /**
             * @brief   Checks whether the specified number is a power of four other than one, and returns the power.
             * 
             * @param[in]   x   the number in question 
             * @return      the power if `x` is actually a power of 4 greater than `1`, or `0` otherwise
             */
            inline size_t which_power_of_4(size_t x)
            {
                size_t checkbit = 3;
                size_t pow = 0;

                if (x & 0x1) return 0;

                while (!(checkbit & x)) 
                {
                    x >>= 2;
                    pow++;
                }
            
                return !(x & 2) ? pow : 0;
            }

            /**
             * @brief   Checks whether the specified number is a power of two other than one, and returns the power.
             * 
             * @param[in]   x   the number in question 
             * @return      the power if `x` is actually a power of 2 greater than `1`, or `0` otherwise
             */
            inline size_t which_power_of_2(size_t x)
            {
                size_t checkbit = 1;
                size_t pow = 0;

                if (x & 0x1) return 0;

                while (!(checkbit & x)) 
                {
                    x >>= 1;
                    pow++;
                }
            
                return !(x >> 1 | 0) ? pow : 0;
            }
        }

        /**
         * @brief   This structure stores data describing a simulation domain that is decomposed into equally sized 
         *          subdomains to match their processing by GPUs. Decomposition is performed into a grid of subdomains.
         *          The quantity and size of the subdomains is specified both vertically and horizontally. Furthermore,
         *          the expanded extents of the entire domain are stored. If subdomains are to be separated by a buffer
         *          grid, additional nodes are planned in when calculating the new extents of the total domain.
         *          The algorithm utilizing the buffers is responsible for addressing it properly. This structure is
         *          not intended to be used to govern buffer interactions.
         */
        struct DecomposedDomain
        {
            unsigned int expanded_node_count;
            unsigned int expanded_horizontal_nodes;
            unsigned int expanded_vertical_nodes;

            unsigned int subdomain_height;
            unsigned int subdomain_width;
            unsigned int subdomain_count_vertical;
            unsigned int subdomain_count_horizontal;

            /**
             * @brief   Constructs a decomposed domain structure where the new total dimensions, the dimensions of
             *          subdomains and their quantity per dimension is calculated based on the input parameters 
             *          describing what the original domain looks like.
             * 
             * @param[in]   unexpanded_horizontal_nodes the amount of horizontal nodes in the original domain
             *                                          including ghost nodes
             * @param[in]   unexpanded_vertical_nodes   the amount of horizontal nodes in the original domain
             *                                          including ghost nodes
             * @param[in]   max_work_group_size         the maximum work group size of the target device
             * @param[in]   buffered                    whether or not buffer nodes are to be planned in or not
             */
            explicit DecomposedDomain
            (
                const unsigned int unexpanded_horizontal_nodes,
                const unsigned int unexpanded_vertical_nodes,
                const size_t max_work_group_size,
                const bool buffered = false 
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

            /** @brief  GPU-allocated densities */
            double *densities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the x components of the velocity vectors of all nodes 
             *          in the simulation domain. Solid nodes should always have a zero component and are not differenciated
             *          further regarding their velocities. All differenciation between solid and fluid nodes is realized
             *          through the density values.
             */
            std::unique_ptr<std::vector<double>> x_velocities_cpu;

            /** @brief  GPU-allocated x components of the velocity vectors */
            double *x_velocities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the y components of the velocity vectors of all nodes 
             *          in the simulation domain. Solid nodes should always have a zero component and are not differenciated
             *          further regarding their velocities. All differenciation between solid and fluid nodes is realized
             *          through the density values.
             */
            std::unique_ptr<std::vector<double>> y_velocities_cpu;

            /** @brief  GPU-allocated y components of the velocity vectors */
            double *y_velocities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the absolutes of the velocity vectors of each node.
             *          It is required for visualization purposes only and remains unused otherwise.
             */
            std::unique_ptr<std::vector<double>> absolute_velocities_cpu;

            /** @brief  GPU-allocated absolutes of the velocity vectors */
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
            std::unique_ptr<Control> control;
            std::unique_ptr<DecomposedDomain> decomposed_domain;

            /**
             * @brief Constructor for the Simulation struct.
             * @throws `lbm::exceptions::json::PropertyArgumentException` if an unknown data layout is read from the JSON file
             */
            explicit Simulation(sycl::queue &queue);
        };

    } // ! namespace core

} // ! namespace lbm

#endif // ! SIMULATION_HPP
