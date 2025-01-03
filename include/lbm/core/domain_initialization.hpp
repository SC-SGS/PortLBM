/**
 * @file        domain_initialization.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, functionality for setting up the simulation domain is declared and defined.
 * 
 * @version     2.1
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef DOMAIN_INITIALIZATION_HPP
#define DOMAIN_INITIALIZATION_HPP

// Dependencies on other LBM core features
#include "boundaries.hpp"
#include "simulation.hpp"
#include "../console/console_output.hpp"

namespace lbm
{

    namespace core
    {

        /**
         * @brief   This namespace contains various methods that are used during the initialization of the simulation domain,
         *          They are not meant to be accessed outside of certain methods that are responsible for domain initialization.
         *          Use the methods of the parent namespace `lbm::core` instead.
         */
        namespace domain_initialization
        {
            /**
             * @brief   Sets the standstill distribution values of the simulation domain on the GPU.
             * 
             * @tparam  A   any `core::access::AccessorConcept` from access.hpp 
             * 
             * @param[in]   queue                   the SYCL queue to which the distribution values residing on the GPU belong
             * @param[in]   buffered_node_count     the total amount of nodes within the simulation domain including buffers and ghost nodes
             * @param[in]   distribution_values     a pointer to GPU-allocated memory at which the distribution values reside
             */
            template<access::AccessorConcept A> 
            void set_standstill_values
            (
                sycl::queue &queue, 
                unsigned int buffered_node_count, 
                double *distribution_values
            )
            {
                std::vector<double> distribution_values_standstill = maxwell_boltzmann_distribution(0, 0, 1);
                double *distribution_values_standstill_gpu = sycl::malloc_device<double>(distribution_values_standstill.size(), queue);
                queue.copy(
                    distribution_values_standstill.data(), distribution_values_standstill_gpu, distribution_values_standstill.size()
                ).wait();

                auto event_fill_dist_vals = queue.parallel_for
                (
                    buffered_node_count,                           
                    [=](const sycl::id<1> &idx)     
                    {
                        for(int direction = 0; direction < 9; ++direction)
                        {
                            distribution_values[A::at(idx, direction, buffered_node_count)] = distribution_values_standstill_gpu[direction];
                        }
                    }  
                );

                event_fill_dist_vals.wait();
                sycl::free(distribution_values_standstill_gpu, queue);
            }

            /**
             * @brief   Sets the inlet and outlet properties as specified within the simulation object.
             * 
             * @tparam  A any `core::access::AccessorConcept` from access.hpp
             * 
             * @param[in]   queue               the SYCL queue on which the distribution values that are to be set reside
             * @param[in]   simulation          the simulation object contains a `core::Properties` object 
             *                                  storing the inlet and outlet density and velocity    
             * @param[in]   distribution_values a pointer to GPU-allocated memory at which the distribution values reside
             */
            template <access::AccessorConcept A>
            void set_inout_distribution_values
            (
                sycl::queue &queue, 
                const Simulation &simulation,
                double *distribution_values
            )
            {
                unsigned int vertical_nodes =  simulation.properties->vertical_nodes;
                unsigned int horizontal_nodes =  simulation.properties->horizontal_nodes;
                unsigned int total_node_count = simulation.properties->buffered_node_count;

                std::vector<double> distribution_inlet = 
                    maxwell_boltzmann_distribution(
                        simulation.properties->inlet_velocity_x, simulation.properties->inlet_velocity_y, simulation.properties->inlet_density
                    );
                
                std::vector<double> distribution_outlet = 
                    maxwell_boltzmann_distribution(
                        simulation.properties->outlet_velocity_x, simulation.properties->outlet_velocity_y, simulation.properties->outlet_density
                    );

                double *distribution_inlet_gpu = sycl::malloc_device<double>(distribution_inlet.size(), queue);
                double *distribution_outlet_gpu = sycl::malloc_device<double>(distribution_outlet.size(), queue);

                queue.copy(distribution_inlet.data(), distribution_inlet_gpu, distribution_inlet.size()).wait();
                queue.copy(distribution_outlet.data(), distribution_outlet_gpu, distribution_outlet.size()).wait();

                auto event_update_inlet = queue.submit(
                    [=](sycl::handler &cgh)
                    {
                        cgh.parallel_for(
                            sycl::range<2>(9, vertical_nodes), 
                            [=](const sycl::id<2> &id)     
                            {
                                unsigned int node = access::get_node_index(0, id.get(1), horizontal_nodes);
                                distribution_values[A::at(node, id.get(0), total_node_count)] = distribution_inlet_gpu[id.get(0)];
                            }  
                        );
                    }
                );

                auto event_update_outlet = queue.submit(
                    [=](sycl::handler &cgh)
                    {
                        cgh.parallel_for(
                            sycl::range<2>(9, vertical_nodes), 
                            [=](const sycl::id<2> &id)     
                            {
                                unsigned int node = access::get_node_index(horizontal_nodes - 1, id.get(1), horizontal_nodes);
                                distribution_values[A::at(node, id.get(0), total_node_count)] = distribution_outlet_gpu[id.get(0)];
                            }  
                        );
                    }
                );

                event_update_inlet.wait(); 
                event_update_outlet.wait(); 

                sycl::free(distribution_inlet_gpu, queue);
                sycl::free(distribution_outlet_gpu, queue);
            }

            /**
             * @brief   Adds the phase information for the `core::WALLS` scenario to the specified vector.
             *          The content of the vector is to be copied to the GPU at a later stage.
             * 
             * @param[in]       properties              the structure containing the extents of the simulation domain
             * @param[in, out]  phase_information_cpu   a vector containing the phase information
             */
            inline 
            void add_walls(const Properties &properties, std::vector<int8_t> &phase_information_cpu)
            {
                for(auto y = 2; y < (properties.vertical_nodes - 1) / 2; ++y)
                {
                    phase_information_cpu[access::get_node_index(properties.horizontal_nodes / 4, y, properties.horizontal_nodes)] = 1;
                    phase_information_cpu[access::get_node_index(properties.horizontal_nodes / 4, y, properties.horizontal_nodes)] = 1;
                }

                for(auto y = properties.vertical_nodes - 1; y >= (properties.vertical_nodes - 1) / 2; --y)
                {
                    phase_information_cpu[access::get_node_index(properties.horizontal_nodes / 2, y, properties.horizontal_nodes)] = 1;
                    phase_information_cpu[access::get_node_index(properties.horizontal_nodes / 2, y, properties.horizontal_nodes)] = 1;
                }

                for(auto y = 2; y < (properties.vertical_nodes - 1) / 2; ++y)
                {
                    phase_information_cpu[access::get_node_index(3 * properties.horizontal_nodes / 4, y, properties.horizontal_nodes)] = 1;
                    phase_information_cpu[access::get_node_index(3 * properties.horizontal_nodes / 4, y, properties.horizontal_nodes)] = 1;
                }
            }

            /**
             * @brief   Adds the phase information for the `core::CIRCLE` scenario to the specified vector.
             *          The content of the vector is to be copied to the GPU at a later stage.
             * 
             * @param[in]       properties              the structure containing the extents of the simulation domain
             * @param[in, out]  phase_information_cpu   a vector containing the phase information
             */
            inline
            void add_circle(const Properties &properties, std::vector<int8_t> &phase_information_cpu)
            {
                int middle_x =  2 * properties.horizontal_nodes / 10;
                int middle_y = properties.vertical_nodes / 2;
                int radius = properties.vertical_nodes / 5;

                for(auto y = middle_y - radius; y <= middle_y + radius; ++y)
                {
                    for(auto x = middle_x - radius; x <= middle_x + radius; ++x)
                    {
                        if(abs(middle_x-x) * abs(middle_x-x) + abs(middle_y-y) * abs(middle_y-y) <= radius * radius)
                            phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                    }
                }
            }

            /**
             * @brief   Adds the phase information for the `core::SQUARE` scenario to the specified vector.
             *          The content of the vector is to be copied to the GPU at a later stage.
             * 
             * @param[in]       properties              the structure containing the extents of the simulation domain
             * @param[in, out]  phase_information_cpu   a vector containing the phase information
             */
            inline
            void add_square(const Properties &properties, std::vector<int8_t> &phase_information_cpu)
            {
                int middle_x =  2 * properties.horizontal_nodes / 10;
                int middle_y = properties.vertical_nodes / 2;
                int length = properties.vertical_nodes / 5;

                for(auto y = middle_y - 0.5 * length; y <= middle_y + 0.5 * length; ++y)
                {
                    for(auto x = middle_x - 0.5 * length; x <= middle_x + 0.5 * length; ++x)
                    {
                        phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                    }
                }
            }

            /**
             * @brief   Adds the phase information for the `core::PLATE` scenario to the specified vector.
             *          The content of the vector is to be copied to the GPU at a later stage.
             * 
             * @param[in]       properties              the structure containing the extents of the simulation domain
             * @param[in, out]  phase_information_cpu   a vector containing the phase information
             */
            inline
            void add_plate(const Properties &properties, std::vector<int8_t> &phase_information_cpu)
            {
                int middle_x =  properties.horizontal_nodes / 5;
                int middle_y = properties.vertical_nodes / 2;
                int length = properties.vertical_nodes / 5;

                for(auto y = middle_y - length / 2; y <= middle_y + length / 2; ++y)
                {
                    for(auto x = middle_x - length / 20; x <= middle_x + length / 20; ++x)
                    {
                        phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                    }
                }
            }

            /**
             * @brief   Adds the phase information for the `core::SKYSCRAPER` scenario to the specified vector.
             *          The content of the vector is to be copied to the GPU at a later stage.
             * 
             * @param[in]       properties              the structure containing the extents of the simulation domain
             * @param[in, out]  phase_information_cpu   a vector containing the phase information
             */
            inline
            void add_skyscraper(const Properties &properties, std::vector<int8_t> &phase_information_cpu)
            {
                int height_antenna = properties.vertical_nodes / 2;
                int width_antenna = 2;

                int height_second = 2 * properties.vertical_nodes / 5;
                int width_second = properties.vertical_nodes / 12;

                int height_first = properties.vertical_nodes / 4;
                int width_first = properties.vertical_nodes / 8;

                int middle_x =  3 * properties.horizontal_nodes / 10;

                for(auto y = 0; y <= height_first; ++y)
                {
                    for(auto x = middle_x - width_first / 2; x <= middle_x + width_first / 2; ++x)
                    {
                        phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                    }
                }

                for(auto y = height_first + 1; y <= height_second; ++y)
                {
                    for(auto x = middle_x - width_second / 2; x <= middle_x + width_second / 2; ++x)
                    {
                        phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                    }
                }

                for(auto y = height_second + 1; y <= height_antenna; ++y)
                {
                    for(auto x = middle_x - width_antenna / 2; x <= middle_x + width_antenna / 2; ++x)
                    {
                        phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                    }
                }
            }

            /**
             * @brief   Adds the phase information for the `core::WING` scenario to the specified vector.
             *          The content of the vector is to be copied to the GPU at a later stage.
             * 
             * @param[in]       properties              the structure containing the extents of the simulation domain
             * @param[in, out]  phase_information_cpu   a vector containing the phase information
             */
            inline
            void add_wing(const Properties &properties, std::vector<int8_t> &phase_information_cpu)
            {
                int middle_x =  2 * properties.horizontal_nodes / 10;
                int middle_y = properties.vertical_nodes / 2;
                int radius_max = properties.vertical_nodes / 5;
                int scale_x = 1;
                int scale_y = 5;

                for(int y = middle_y - radius_max; y <= middle_y + radius_max; ++y)
                {
                    for(int x = middle_x - radius_max; x <= middle_x; ++x)
                    {
                        if(0.2 * abs(x - middle_x) * abs(x - middle_x) + abs(y - middle_y) * abs(y - middle_y) <= radius_max)
                        {
                            phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                        }  
                    }
                }

                for(auto y = middle_y - 2 * radius_max; y <= middle_y + 2 * radius_max; ++y)
                {
                    for(auto x = middle_x; x <= properties.horizontal_nodes - 1; ++x)
                    {
                        if(std::pow(abs(middle_x - x), 1.4) + scale_y * scale_y * abs(middle_y - y) * abs(middle_y - y) <= scale_y * scale_y * radius_max)
                        {
                            phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                        }
                            
                    }
                }
            }

            /**
             * @brief   Adds the phase information for the `core::POROUS` scenario to the specified vector.
             *          The content of the vector is to be copied to the GPU at a later stage.
             * 
             * @param[in]       properties              the structure containing the extents of the simulation domain
             * @param[in, out]  phase_information_cpu   a vector containing the phase information
             */
            inline
            void add_porous_medium(const Properties &properties, std::vector<int8_t> &phase_information_cpu)
            {
                for(int x = 0; x < properties.horizontal_nodes; ++x)
                {
                    
                    for(int y = 0; y < properties.vertical_nodes; ++y)
                    {
                        int one = 1 + (std::rand() % 20);
                        int two = 1 + (std::rand() % 20);
                        if(x % one == one - 1 && y % two == two - 1)
                            phase_information_cpu[access::get_node_index(x, y, properties.horizontal_nodes)] = 1;
                    }
                }
            }

        }   // ! namespace domain_initialization
        
        /**
         * @brief   Sets the up pipe flow environment object with a fluid in an equilibrium non-moving state.
         *          The domain consists of solid nodes on the upper and lower boundary, and further solid obstacles
         *          depending on the scenario specified within the simulation object.
         * 
         * @tparam A any `core::access::AccessorConcept` from access.hpp
         * 
         * @param[in]       queue       the SYCL queue to which the values residing on the GPU belong
         * @param[in, out]  simulation  reference to the structure containing all simulation data 
         * @param[in]       obstacle    what kind of obstacle is added to the domain
         */
        template<access::AccessorConcept A> 
        void setup_pipe_flow_environment
        (
            Simulation &simulation, 
            sycl::queue &queue, 
            Obstacle obstacle = NONE
        )
        {
            domain_initialization::set_standstill_values<A>(
                queue, simulation.properties->buffered_node_count, simulation.data->distribution_values_0
            );
            domain_initialization::set_inout_distribution_values<A>(
                queue, simulation, simulation.data->distribution_values_0
            );

            if
            (
                simulation.properties->algorithm == "gpu-two-lattice-linear" || 
                simulation.properties->algorithm == "gpu-two-lattice"
            )
            {
                domain_initialization::set_standstill_values<A>(
                    queue, simulation.properties->buffered_node_count, simulation.data->distribution_values_1
                );
                domain_initialization::set_inout_distribution_values<A>(
                    queue, simulation, simulation.data->distribution_values_1
                );
            }

            if(simulation.properties->debug_mode)
            {
                std::vector<double> dist_vals(9 * simulation.properties->buffered_node_count, 0);
                queue.copy(simulation.data->distribution_values_0, dist_vals.data(), 9 * simulation.properties->buffered_node_count).wait();
                console::print_distribution_values<A>(dist_vals, simulation.properties->horizontal_nodes, simulation.properties->vertical_nodes);
            }

            std::vector<int8_t> phase_information_cpu(simulation.properties->buffered_node_count, 0);

            /* Phase information vector */
            for(auto x = 1; x < simulation.properties->horizontal_nodes - 1; ++x)
            {
                phase_information_cpu[access::get_node_index(x, 1, simulation.properties->horizontal_nodes)] = 1;
                phase_information_cpu[access::get_node_index(x, simulation.properties->vertical_nodes - 2, simulation.properties->horizontal_nodes)] = 1;
            }
            for(int y = 0; y < simulation.properties->vertical_nodes - 1; ++y)
            {
                phase_information_cpu[access::get_node_index(0, y, simulation.properties->horizontal_nodes)] = -1;
                phase_information_cpu[access::get_node_index(simulation.properties->horizontal_nodes - 1, y, simulation.properties->horizontal_nodes)] = -1;
            }
            for(int x = 0; x < simulation.properties->horizontal_nodes - 1; ++x)
            {
                phase_information_cpu[access::get_node_index(x, simulation.properties->vertical_nodes - 1, simulation.properties->horizontal_nodes)] = -1;
                phase_information_cpu[access::get_node_index(x, 0, simulation.properties->horizontal_nodes)] = -1;
            }   

            switch(obstacle)
            {
                case WALLS:
                    domain_initialization::add_walls(*simulation.properties, phase_information_cpu); 
                    break;
                case CIRCLE:
                    domain_initialization::add_circle(*simulation.properties, phase_information_cpu); 
                    break;
                case SQUARE:
                    domain_initialization::add_square(*simulation.properties, phase_information_cpu); 
                    break;
                case PLATE:
                    domain_initialization::add_plate(*simulation.properties, phase_information_cpu); 
                    break;
                case SKYSCRAPER:
                    domain_initialization::add_skyscraper(*simulation.properties, phase_information_cpu); 
                    break;
                case WING:
                    domain_initialization::add_wing(*simulation.properties, phase_information_cpu); 
                    break;
                case POROUS:
                    domain_initialization::add_porous_medium(*simulation.properties, phase_information_cpu); 
                    break;
                default:
                    break;
            }
            
            queue.copy(phase_information_cpu.data(), simulation.data->phase_information, simulation.properties->buffered_node_count).wait();

            if(simulation.properties->debug_mode)
            {
                std::vector<int8_t> phase_information_check(simulation.properties->buffered_node_count, 0);
                queue.copy(simulation.data->phase_information, phase_information_check.data(), simulation.properties->buffered_node_count).wait();
                console::print_phase_vector(phase_information_check, simulation.properties->horizontal_nodes);
            }
        }

    } // ! namespace core

} // ! namespace lbm

#endif // ! DOMAIN_INITIALIZATION_HPP
