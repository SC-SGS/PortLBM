/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declaration of crucial functionality of the SYCL lattice Boltzmann 
 *              simulations.
 * 
 * @version     4.2
 * 
 * @date        February 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_SIMULATION_HPP
#define LBM_SIMULATION_HPP

// Includes ///////////////////////////////////////////////////////////////////////////////////////////////////////////

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

// Core functionality /////////////////////////////////////////////////////////////////////////////////////////////////

namespace lbm
{

    namespace core
    {

        /**
         * @brief   Returns the Maxwell-Boltzmann-Distribution for all directions in the order proposed by Mattila et al.
         * 
         * @param[in]   x_velocity    x component of the velocity of the node in question
         * @param[in]   y_velocity    y component of the velocity of the node in question
         * @param[in]   density       density of the node in question
         * 
         * @return  a vector containing the distribution values
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
         * @brief  Returns the inverse direction of that specified. 
         */
        inline constexpr unsigned int invert_direction(const unsigned int dir) { return 8 - dir; }

        /**
         * @brief This structure contains all important properties of the simulation.
         */
        struct Properties
        {
            /* Algorithmic options */

            /**
             * @brief   Possible values for the algorithm string are:
             *        
             *          - `"gpu-two-lattice"`
             * 
             *          - `"gpu-two-lattice-linear"`
             * 
             *          - `"gpu-swap"`
             */
            std::string algorithm;

            /**
             * @brief   Possible values for the data layout string are:
             *        
             *          - `"stream"`
             * 
             *          - `"collide"`
             * 
             *          - `"bundle"`
             */
            std::string data_layout;

            // If set, a mode with thorough console debug prints is entered.
            bool debug_mode;

            unsigned int work_group_size;

            // This many time steps will be executed in total by the algorithm.
            unsigned int time_steps;

            // Every `frame_update_interval` iterations, macroscopic observables are copied to the host
            unsigned int frame_update_interval;

            /* Simulation domain */

            /**
             * @brief   Possible values for the scenario string are:
             * 
             *          - `"Hagen-Poiseuille"`
             * 
             *          - `"walls"`
             * 
             *          - `"circle"`
             *  
             *          - `"square"`
             * 
             *          - `"plate"`
             * 
             *          - `"skyscraper"`
             * 
             *          - `"wing"`
             * 
             *          - `"porous"`
             * 
             * @see     core/domain_initialization.hpp
             */
            std::string scenario;

            // Unbuffered amount of vertical nodes including an outer layer of ghost nodes
            unsigned int vertical_nodes;

            // Unbuffered amount of vertical nodes including an outer layer of ghost nodes
            unsigned int horizontal_nodes;

            // Total amount of nodes including ghost nodes and buffer nodes
            unsigned int total_unexpanded_node_count;

            // Total amount of node within the actual simulation domain, excluding ghost nodes
            unsigned int domain_node_count;

            /* Physical */

            double inlet_velocity_x;
            double inlet_velocity_y;
            double inlet_density;

            double outlet_velocity_x;
            double outlet_velocity_y;
            double outlet_density;

            double relaxation_time;

            /**
             * @brief Constructs a new properties object with the specified parameters.
             */
            explicit Properties
            (
                // Algorithmic properties
                const std::string &&algorithm,
                const std::string &&data_layout,
                const bool debug_mode,
                const unsigned int work_group_size,
                const unsigned int time_steps,
                const unsigned int frame_update_interval,
                // Domain properties
                const std::string &&scenario,
                const unsigned int vertical_nodes,
                const unsigned int horizontal_nodes,
                // Physical
                const double inlet_velocity_x,
                const double inlet_velocity_y,
                const double inlet_density,
                const double outlet_velocity_x,
                const double outlet_velocity_y,
                const double outlet_density,
                const double relaxation_time
            ); 

            /**
             * @brief   Returns a string where the data stored by this object is prepared for console output.
             */
            std::string to_string() const;
        };

        /**
         * @brief   This class offers options to control an algorithm and to query its progress and performance data.
         */
        class Control
        {
            private:

            // Whether execution has been manually disallowed for this control object or not
            bool stopped;

            // The current iteration of the targeted algorithm
            unsigned int current_iteration;

            // The maximum amount of iterations allowed for the targeted algorithm
            unsigned int max_iterations;

            // The current progress of the targeted algorithm in range [0,1]
            double progress;

            // Pointer to a high resolution HPX timer that measures the time spent for performing one iteration
            std::unique_ptr<hpx::chrono::high_resolution_timer> timer;

            // The last measured calculation time for one full lattice Boltzmann iteration
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
             * 
             * @return  the power if `x` is actually a power of 4 greater than `1`, or `0` otherwise
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
             * 
             * @return  the power if `x` is actually a power of 2 greater than `1`, or `0` otherwise
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
         * @brief   This structure stores data describing the simulation domain on which an algorithm operates. It
         *          prepares a suitable description of a domain for particular algorithms based on constructors 
         *          accepting inputs from the `core::Properties` structure. The `Domain` object specifies the domain
         *          on which the algorithm actually operates, which is only equivalent to what the `core::Properties`
         *          object contains when no buffering and no decomposition are involved, that is, for the linear two-
         *          lattice algorithm.
         * 
         *          In the case of a non-buffered and non-decomposed domain, the grid extents are just taken over from
         *          the according `core::Properties` object. There is one subdomain, and its extents match that of the
         *          entire domain. In this case, this object is used for convenience to facilitate defining methods
         *          that work for multiple algorithms.
         *  
         *          In the case of a non-buffered and decomposed domain, as it is used for the non-linear two-lattice
         *          algorithm, the domain is expanded to match the extents of the work groups. The decomposition is
         *          performed depending on the maximum work group size of the device.
         * 
         *          In the case of a buffered and decomposed domain, as it is used for the swap algorithm, the behavior
         *          is generally the same as for the non-buffered decomposed domain with the difference that nodes for
         *          a buffer grid are added that are not considered part of any particular subdomain. The domain object
         *          offers no functionality to deal with these buffers; this is considered the responsibility of the
         *          algorithm.
         * 
         */
        struct Domain
        {
            // The total amount of nodes in this domain. Buffer nodes are included for buffered domains.
            unsigned int total_node_count;

            // The total amount of horizontal nodes in this domain. Buffer nodes are included for buffered domains.
            unsigned int horizontal_nodes;

            // The total amount of vertical nodes in this domain. Buffer nodes are included for buffered domains.
            unsigned int vertical_nodes;

            // The amount of vertical nodes per subdomain
            unsigned int subdomain_vertical_nodes;

            // The amount of horizontal nodes per subdomain
            unsigned int subdomain_horizontal_nodes;

            // The amount of subdomains in vertical direction
            unsigned int subdomain_count_vertical;

            // The amount of subdomains in horizontal direction
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
            explicit Domain
            (
                const unsigned int unexpanded_horizontal_nodes,
                const unsigned int unexpanded_vertical_nodes,
                const size_t max_work_group_size,
                const bool buffered = false 
            );

            /**
             * @brief   Constructs a non-decomposed domain structure where the new total dimensions correspond to those
             *          specified in the `core::Properties` object. There is only one subdomain that has the extents of
             *          the entire grid.
             * 
             * @param[in]   horizontal_nodes    the amount of horizontal nodes in the original domain including ghost 
             *                                  nodes
             * @param[in]   vertical_nodes      the amount of vertical nodes in the original domain including ghost
             *                                  nodes
             */
            explicit Domain
            (
                const unsigned int horizontal_nodes,
                const unsigned int vertical_nodes
            );
        };

        struct SwapDomain
        {
            // The total amount of nodes in this domain. Buffer nodes are included for buffered domains.
            unsigned int total_node_count;

            // The total amount of horizontal nodes in this domain. Buffer nodes are included for buffered domains.
            unsigned int horizontal_nodes;

            // The total amount of vertical nodes in this domain. Buffer nodes are included for buffered domains.
            unsigned int vertical_nodes;

            // The amount of vertical nodes per subdomain
            unsigned int subdomain_vertical_nodes;

            // The amount of horizontal nodes per subdomain
            unsigned int subdomain_horizontal_nodes;

            // The amount of subdomains in vertical direction
            unsigned int subdomain_count_vertical;

            // The amount of subdomains in horizontal direction
            unsigned int subdomain_count_horizontal;

            explicit SwapDomain
            (
                const unsigned int unexpanded_horizontal_nodes,
                const unsigned int unexpanded_vertical_nodes,
                const size_t max_work_group_size
            );
        };

        /**
         * @brief This structure contains the results of the simulation in a structure-of-arrays representation.
         */
        class Results
        {
            private:

            // This SYCL queue is used to allocate the device memory and to transfer results to the host.
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

            // GPU-allocated densities
            double *densities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the x components of the velocity vectors of all nodes 
             *          in the simulation domain. Solid nodes should always have a zero component and are not 
             *          differenciated further regarding their velocities. All differenciation between solid and fluid
             *          nodes is realized through the density values.
             */
            std::unique_ptr<std::vector<double>> x_velocities_cpu;

            // GPU-allocated x components of the velocity vectors
            double *x_velocities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the y components of the velocity vectors of all nodes 
             *          in the simulation domain. Solid nodes should always have a zero component and are not 
             *          differenciated further regarding their velocities. All differenciation between solid and fluid 
             *          nodes is realized through the density values.
             */
            std::unique_ptr<std::vector<double>> y_velocities_cpu;

            // GPU-allocated y components of the velocity vectors
            double *y_velocities_gpu;

            /**
             * @brief   A unique pointer to a vector containing the absolutes of the velocity vectors of each node. It
             *          is required for visualization purposes only and remains unused otherwise.
             */
            std::unique_ptr<std::vector<double>> absolute_velocities_cpu;

            // GPU-allocated absolutes of the velocity vectors
            double *absolute_velocities_gpu;

            /**
             * @brief   Constructs a new simulation results object based on the provided properties structure. The 
             *          internal vectors are initialized with the correct size and filled up with values such as all
             *          nodes were solid.
             * 
             * @param[in]   size    the size of each vector should be set to the amount of actual nodes (neither ghost
             *                      nor buffer)
             * @param[in]   queue   the SYCL queue used to allocate the result arrays on the device
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

            // This SYCL queue is used to allocate the device memory.
            std::shared_ptr<sycl::queue> queue;

            public:

            /**
             * @brief   This GPU-allocated array stores the phase information of every node in the simulation domain.
             *          The following values are used:
             *          
             *          - `1` for solid nodes; these are only considered when preparing bounce-back
             *          
             *          - `0` for fluid nodes; these are considered for streaming and inout handling
             * 
             *          - `-1` for ghost and buffer nodes; ghost nodes are never explicitly considered
             */
            int8_t *phase_information;

            /**
             * @brief   This GPU-allocated array stores the distribution values of all nodes in the simulation domain.
             *          The storage pattern is defined by the data layout on which the algorithm operates. All 
             *          algorithms use this array.
             */
            double *distribution_values_0;

            /**
             * @brief   This GPU-allocated array is only used for the linear and non-linear two-lattice algorithm,
             *          in which case it also contains the distribution values. For the swap algorithm, it remains
             *          uninitialized, that is, a `nullptr`.
             */
            double *distribution_values_1;

            /**
             * @brief Constructs a new Data object with an accessor object of the specified type.
             * 
             * @param[in] total_unexpanded_node_count the amount of nodes in the lattice including ghosts and buffers.
             */

            /**
             * @brief   Constructs a new `Data` object that contains the phase information and the distribution values.
             *          The array `distribution_values_1` is only initialized for the linear and non-linear two-lattice
             *          algorithm.
             * 
             * @param[in]   total_node_count    the total amount of nodes in the domain including buffer and ghost 
             *                                  nodes
             * @param[in]   queue               the SYCL queue used to allocate the data on the device
             * @param[in]   two_lattice         whether or not the data object is intended for use with a two-lattice
             *                                  algorithm
             */
            explicit Data(const size_t total_node_count, sycl::queue &queue, const bool two_lattice);

            ~Data()
            {
                sycl::free(phase_information, *queue);
                sycl::free(distribution_values_0, *queue);
                if(!(distribution_values_1 == nullptr))
                    sycl::free(distribution_values_1, *queue);
            }
        };

        /**
         * @brief   This structure contains all data that is related to the simulation.
         */
        struct Simulation
        {
            std::unique_ptr<Properties> properties;
            std::unique_ptr<Data> data;
            std::unique_ptr<Results> results;
            std::unique_ptr<Control> control;
            std::unique_ptr<SwapDomain> domain;

            /**
             * @brief   This struct contains all data required for a GPU lattice Boltzmann implementation.
             * 
             * @param[in]   queue   the SYCL queue used for interactions with the device
             * 
             * @throws  `lbm::exceptions::json::PropertyArgumentException` if an unknown data layout is read from the 
             *          JSON file
             */
            explicit Simulation(sycl::queue &queue);
        };

    } // ! namespace core

} // ! namespace lbm

#endif // ! SIMULATION_HPP
