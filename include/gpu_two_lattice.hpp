/**
 * @file        gpu_two_lattice.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header, all functionality of the GPU-based two lattice algorithm is declared.
 *              At the moment, the name is misleading since the functionality is not yet realized on the GPU.
 *              In this early version, all simulations are still run on the CPU variant as defined in my SimTech project work
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann.
 *              Currently, the objective is to prepare the data structures required for the GPU variant.
 *              Compatiblity to the CPU implementation is achieved by additional measures.
 *              Eventually, all functionality is realized on the GPU, and any deprecated features will be removed.
 * 
 * @version     0.1
 * 
 * @date        2024-10-03
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef GPU_TWO_LATTICE_HPP
#define GPU_TWO_LATTICE_HPP

#include "access.hpp"
#include "collision.hpp"
#include "defines.hpp"
#include "utils.hpp"
#include "simulation.hpp"

#include <vector>
#include <set>
//Not yet: #include <sycl/sycl.hpp>

/**
 * @brief This namespace contains all methods for the two-lattice algorithm on the GPU.
 */
namespace gpu_two_lattice
{

    /**
     * @brief The non-debug variant of the stream and collide algorithm is not yet implemented since 
     *        the development is too early to create a non-debug variant.
     *        But there will be such a variant in the future.
     * 
     * @tparam T any child class of lbm_access::LBMAccessorObject
     * @param simulation_data a structure of data on which the simulation operates
     */
    template <class T> void stream_and_collide
    (
        const SimulationData<T> &simulation_data
    );

    /**
     * @brief This is the debug and development variant of the stream and collide method on the GPU.
     *        It performs the streaming and the collision step.
     *        Various debug comments are printed in the console.
     * 
     * @tparam T any child class of lbm_access::LBMAccessorObject
     * 
     * @param simulation_data           a structure containing the data on which the algorithm operates
     * @param sim_results               a structure containing the macroscopic simulation results produced by the algorithm
     * @param iteration_node_offset     the offset for reading the macroscopic observables based on the current iteration
     */
    template <class T> void stream_and_collide_debug
    (
        const SimulationData<T> &simulation_data,
        const SimulationResults &sim_results,
        const unsigned int iteration_node_offset
    )
    {
        std::cout << "Got iteration node offset " << iteration_node_offset << ", expected " << TOTAL_NODE_COUNT << "\n";
        std::cout << "\t SOURCE before stream and collide: " << std::endl;
        to_console::print_distribution_values(*simulation_data.distribution_values_0, *simulation_data.lbm_accessor);

        std::cout << "\t DESTINATION before stream and collide: " << std::endl;
        to_console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);

        // Stream
        for(unsigned int node = 0; node <= simulation_data.end_node_index_buffered; ++node)
        {   
            if(!is_ghost_node(node, *simulation_data.phase_information))
            {
                for (const auto direction : ALL_DIRECTIONS)
                {
                    (*simulation_data.distribution_values_1)[simulation_data.lbm_accessor->get_index(node, direction)] =
                        (*simulation_data.distribution_values_0)[
                            simulation_data.lbm_accessor->get_index(
                                lbm_access::get_neighbor(node, invert_direction(direction)), 
                                direction)];
                }
            }
        }

        std::cout << "DESTINATION after streaming: " << std::endl;
        to_console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);

        int size = simulation_data.distribution_values_0->size() / DIRECTION_COUNT;
        std::vector<double> dbg_densities(size, -1);
        std::vector<velocity> dbg_velocities(size, velocity{0,0});

        // Calculate density and velocity
        for(unsigned int node = 0; node < simulation_data.end_node_index_buffered; ++node)
        {
            if(!is_ghost_node(node, *simulation_data.phase_information))
            {
                std::vector<double> current_distributions = 
                    lbm_access::get_distribution_values_of<T>(*simulation_data.distribution_values_1, node, *simulation_data.lbm_accessor);
                
                // Density
                (*sim_results.densities)[node + iteration_node_offset] = macroscopic::density(current_distributions);
                dbg_densities[node] =  (*sim_results.densities)[node + iteration_node_offset];

                // Velocity
                velocity v = macroscopic::flow_velocity(current_distributions); 
                (*sim_results.x_velocities)[node + iteration_node_offset] = v[0];
                (*sim_results.y_velocities)[node + iteration_node_offset] = v[1];
                dbg_velocities[node] = velocity{(*sim_results.x_velocities)[node + iteration_node_offset], (*sim_results.y_velocities)[node + iteration_node_offset]};
            }
        }

        std::cout << "Velocities: " << std::endl;
        to_console::print_velocity_vector(dbg_velocities);

        std::cout << "Densities: " << std::endl;
        to_console::print_vector(dbg_densities, HORIZONTAL_NODES);

        to_console::print_phase_vector(*simulation_data.phase_information);

        // Collide
        for(unsigned int node = 0; node < simulation_data.end_node_index_buffered; ++node)
        {   
            if(!is_ghost_node(node, *simulation_data.phase_information))
            {
                std::cout << "Node " << node << " is not a ghost node. Calculating values. \n"; 
                std::vector<double> current_distributions = lbm_access::get_distribution_values_of(*simulation_data.distribution_values_1, node, *simulation_data.lbm_accessor);
                to_console::print_vector(current_distributions, current_distributions.size());
                current_distributions = collision::collide_bgk(current_distributions, dbg_velocities[node], dbg_densities[node]);
                std::cout << "After collision, node " << node << " is found to have distribution values \n\n"; 
                to_console::print_vector(current_distributions, current_distributions.size());
                lbm_access::set_distribution_values_of(current_distributions, *simulation_data.distribution_values_1, node, *simulation_data.lbm_accessor);
            }

        }

        std::cout << "DESTINATION after collision: " << std::endl;
        to_console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param properties 
     * @param simulation_data 
     * @param legacy_data 
     * @param debug_mode 
     */
    template <class T> void run
    (  
        const Properties &properties,
        SimulationData<T> &simulation_data,
        const LegacyData &legacy_data,
        const bool debug_mode = true
    )
    {
        to_console::print_run_greeting("GPU two-lattice algorithm", properties.time_steps);
        SimulationResults sim_results(properties);
        std::cout << "Length of simulation result is " << sim_results.x_velocities->size() << "\n";
        std::shared_ptr<std::vector<double>> temp;
        std::vector<velocity> velocities(properties.non_buffered_node_count, velocity{0,0});
        std::vector<double> densities(properties.non_buffered_node_count, 0);
        unsigned int iteration_node_offset = 0;

        for(auto step = 0; step < properties.time_steps; ++step)
        {
            std::cout << "\033[33mIteration " << step << ":\033[0m \n";

            // Bounce-back
            bounce_back::emplace_bounce_back_values(*legacy_data.bsi, *simulation_data.distribution_values_0, *simulation_data.lbm_accessor);

            if(debug_mode) 
            {
                std::cout << "SOURCE after emplace bounce-back values: " << std::endl;
                to_console::print_distribution_values(*simulation_data.distribution_values_0, *simulation_data.lbm_accessor);
            }

            // Stream and collide
            if(!debug_mode) {/*stream_and_collide(simulation_data);*/}
            else {stream_and_collide_debug(simulation_data, sim_results, iteration_node_offset);}

            // Inout update
            boundary_conditions::update_velocity_input_density_output(*simulation_data.distribution_values_1, velocities, densities, *simulation_data.lbm_accessor);
            if(debug_mode)
            {
                std::cout << "Updated inlet and outlet ghost nodes." <<std::endl;
                to_console::print_distribution_values(*simulation_data.distribution_values_1, *simulation_data.lbm_accessor);
            }

            // Done. Next.
            std::cout << "\tFinished iteration " << step << std::endl;
            simulation_data.distribution_values_1.swap(simulation_data.distribution_values_0);

            iteration_node_offset += properties.non_buffered_node_count;
        }

        std::cout << "All done, exiting simulation. " << std::endl;
        std::vector<sim_data_tuple>result(
            properties.time_steps, 
        std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT, 0)));

        for(auto time = 0; time < properties.time_steps; ++time)
        {
            for(auto node = 0; node < simulation_data.end_node_index_buffered; ++node)
            {
                std::get<0>(result[time])[node][0] = (*sim_results.x_velocities)[node + time * properties.non_buffered_node_count];
                std::get<0>(result[time])[node][1] = (*sim_results.y_velocities)[node + time * properties.non_buffered_node_count];
                std::get<1>(result[time])[node] = (*sim_results.densities)[node + time * properties.non_buffered_node_count];
            }
        }

        to_console::print_simulation_results(result);
    }
}

#endif