/**
 * @file        domain_initialization.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, functionality for setting up the simulation domain is declared and defined.
 * 
 * @version     3.0
 * 
 * @date        February 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_DOMAIN_INITIALIZATION_HPP
#define LBM_DOMAIN_INITIALIZATION_HPP

// Dependencies on other LBM features
#include "simulation.hpp"
// #include "../console/console_output.hpp" // for debug purposes only

namespace lbm
{

    namespace core
    {

        /**
         * @brief   This namespace contains various methods and structs that are used during the initialization of the 
         *          simulation domain. 
         */
        namespace domain_initialization
        {

// GPU STENCILS ///////////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief   This namespace contains structures that are used for parameterizing certain phase information
             *          stencils using templates. These stencil structures all implement an `evaluate` method that
             *          defines a condition for when a node is solid (`1`) or fluid (`0`).
             * 
             */
            namespace gpu_stencils
            {

                /**
                 * @brief   This stencil defines the evaluation condition for a Hagen-Poiseuille environment.
                 */
                struct HagenPoiseuilleStencil
                {
                    static inline
                    unsigned int evaluate
                    (
                        const unsigned int x, 
                        const unsigned int y,
                        const unsigned int vertical_nodes_org, 
                        const unsigned int horizontal_nodes_org
                    )
                    {
                        return 0;
                    }
                };

                /**
                 * @brief   This stencil defines the evaluation condition for a scenario with three walls pointing 
                 *          upward from the lower pipe boundary and two walls pointing downward from the upper pipe
                 *          boundary. All walls are half the vertical size of the domain.
                 */
                struct WallStencil
                {
                    static inline
                    unsigned int evaluate
                    (
                        const unsigned int x, 
                        const unsigned int y,
                        const unsigned int vertical_nodes_org, 
                        const unsigned int horizontal_nodes_org
                    )
                    {
                        unsigned int cond = 
                            (
                                x == (1 * horizontal_nodes_org) / 6 || 
                                x == (3 * horizontal_nodes_org) / 6 || 
                                x == (5 * horizontal_nodes_org) / 6
                            ) 
                            && (y > 1) && (y < vertical_nodes_org / 2);

                        cond +=
                            (
                                x == (2 * horizontal_nodes_org) / 6 || 
                                x == (4 * horizontal_nodes_org) / 6
                            ) 
                            && (y < vertical_nodes_org - 2) && (y > ((vertical_nodes_org - 1) / 2));

                        return cond;
                    }
                };

                /**
                 * @brief   This stencil defines the evaluation condition for a scenario with a circle obstacle.
                 */
                struct CircleStencil
                {
                    static inline
                    unsigned int evaluate
                    (
                        const unsigned int x, 
                        const unsigned int y,
                        const unsigned int vertical_nodes_org, 
                        const unsigned int horizontal_nodes_org
                    )
                    {
                        unsigned int middle_x =  2 * horizontal_nodes_org / 10;
                        unsigned int middle_y = vertical_nodes_org / 2;
                        unsigned int radius = vertical_nodes_org / 5;

                        unsigned int cond = 
                        (y >= middle_y - radius) &&
                        (y <= middle_y + radius) &&
                        (x >= middle_x - radius) &&
                        (x <= middle_x + radius) &&
                        (sycl::abs(middle_x-x) * sycl::abs(middle_x-x) 
                            + sycl::abs(middle_y-y) * sycl::abs(middle_y-y) <= radius * radius);
                        return cond;
                    }
                };

                /**
                 * @brief   This stencil defines the evaluation condition for a scenario with a square obstacle.
                 */
                struct SquareStencil
                {
                    static inline
                    unsigned int evaluate
                    (
                        const unsigned int x, 
                        const unsigned int y,
                        const unsigned int vertical_nodes_org, 
                        const unsigned int horizontal_nodes_org
                    )
                    {
                        unsigned int middle_x =  2 * horizontal_nodes_org / 10;
                        unsigned int middle_y = vertical_nodes_org / 2;
                        unsigned int half_length = vertical_nodes_org / 10;

                        unsigned int cond = 
                        (y >= middle_y - half_length) &&
                        (y <= middle_y + half_length) &&
                        (x >= middle_x - half_length) &&
                        (x <= middle_x + half_length);
                        return cond;
                    }
                };

                /**
                 * @brief   This stencil defines the evaluation condition for a scenario with a plate obstacle.
                 */
                struct PlateStencil
                {
                    static inline
                    unsigned int evaluate
                    (
                        const unsigned int x, 
                        const unsigned int y,
                        const unsigned int vertical_nodes_org, 
                        const unsigned int horizontal_nodes_org
                    )
                    {
                        unsigned int middle_x =  horizontal_nodes_org / 10;
                        unsigned int middle_y = vertical_nodes_org / 2;
                        unsigned int half_height = 7 * vertical_nodes_org / 40;
                        unsigned int half_width = half_height / 60;

                        unsigned int cond = 
                        (y >= middle_y - half_height) &&
                        (y <= middle_y + half_height) &&
                        (x >= middle_x - half_width) &&
                        (x <= middle_x + half_width);
                        return cond;
                    }
                };

                /**
                 * @brief   This stencil defines the evaluation condition for a scenario with a skyscraper obstacle.
                 *          The skyscraper has three storeys of decreasing width.
                 */
                struct SkyscraperStencil
                {
                    static inline
                    unsigned int evaluate
                    (
                        const unsigned int x, 
                        const unsigned int y,
                        const unsigned int vertical_nodes_org, 
                        const unsigned int horizontal_nodes_org
                    )
                    {
                        unsigned int height_antenna = vertical_nodes_org / 2;
                        unsigned int half_width_antenna = 1;

                        unsigned int height_second = 2 * vertical_nodes_org / 5;
                        unsigned int half_width_second = vertical_nodes_org / 24;

                        unsigned int height_first = vertical_nodes_org / 4;
                        unsigned int half_width_first = vertical_nodes_org / 16;

                        unsigned int middle_x =  3 * horizontal_nodes_org / 10;

                        unsigned int cond = 
                        (y >= 1) &&
                        (y <= height_first && x >= middle_x - half_width_first && x <= middle_x + half_width_first) ||
                        (y <= height_second && x >= middle_x - half_width_second && x <= middle_x + half_width_second) 
                        || (y <= height_antenna && x >= middle_x - half_width_antenna && x <= middle_x + 
                        half_width_antenna);
                        return cond;
                    }
                };

                /**
                 * @brief   This stencil defines the evaluation condition for a scenario with a wing-shaped obstacle.
                 */
                struct WingStencil
                {
                    static inline
                    unsigned int evaluate
                    (
                        const unsigned int x, 
                        const unsigned int y,
                        const unsigned int vertical_nodes_org, 
                        const unsigned int horizontal_nodes_org
                    )
                    {
                        int middle_x = horizontal_nodes_org / 5;
                        int middle_y = vertical_nodes_org / 2;
                        int radius_max = vertical_nodes_org / 5;
                        int scale_x = 1;
                        int scale_y = 5;

                        unsigned int cond = 
                        (
                            (
                                0.2 * sycl::abs((x - middle_x) * (x - middle_x)) 
                                + sycl::abs((y - middle_y) * (y - middle_y))
                            ) 
                            <= radius_max
                        ) ||
                        (
                            (
                                sycl::pow(sycl::abs(x - middle_x), 1.4) 
                                + scale_y * scale_y * sycl::abs((y - middle_y) * (y - middle_y))
                            ) 
                            <= 
                            scale_y * scale_y * radius_max
                        );
                        return cond;
                    }
                };

                /**
                 * @brief   This concept allows to specify a phase stencil for assignment on the GPU through a template
                 *          parameter.
                 * 
                 * @tparam  T   this class is tested for membership in the set of accessors during compile time
                 */
                template <class T>
                concept GPUStencil = 
                std::same_as<T, HagenPoiseuilleStencil> || 
                std::same_as<T, WallStencil> || 
                std::same_as<T, CircleStencil> || 
                std::same_as<T, SquareStencil> || 
                std::same_as<T, PlateStencil> || 
                std::same_as<T, SkyscraperStencil> || 
                std::same_as<T, WingStencil>;

            /**
             * @brief   Adds the pipe boundaries to the phase information vector belonging to the specified simulation.
             * 
             * @tparam  N   any `lbm::core::access::decomposed::NodeAccessor` from access.hpp
             * 
             * @param[in, out]  simulation  this object contains a pointer to the concerned phase information vector 
             * @param[in, out]  queue       the SYCL queue that is to be used
             */
            template<core::access::decomposed::NodeAccessor N>
            void set_pipe_boundaries
            (
                Simulation &simulation, 
                sycl::queue &queue
            )
            {
                auto event = queue.submit(
                    [&](sycl::handler &cgh)
                    {
                        unsigned int extended_horizontal_nodes = 
                            simulation.domain->horizontal_nodes;
                        unsigned int horizontal_nodes_org = simulation.properties->horizontal_nodes;
                        unsigned int vertical_nodes_org = simulation.properties->vertical_nodes;
                        unsigned int subdomain_horizontal_nodes = simulation.domain->subdomain_horizontal_nodes;
                        unsigned int subdomain_vertical_nodes = simulation.domain->subdomain_vertical_nodes;

                        int8_t *phase_information = simulation.data->phase_information;

                        cgh.parallel_for(
                            sycl::range<2>(2, simulation.properties->horizontal_nodes - 2), 
                            [=](const sycl::id<2> &id)     
                            {
                                phase_information[
                                    N::get_index(
                                        id.get(1) + 1, id.get(0) * vertical_nodes_org + 1 - 3 * id.get(0), 
                                        subdomain_vertical_nodes, 
                                        subdomain_horizontal_nodes, 
                                        extended_horizontal_nodes
                                    )
                                ] = 1;
                            }  
                        );
                    }
                );
                event.wait();
            }

            } // ! namespace gpu_stencils

// CPU PHASE MASKS ////////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief   This namespace contains methods to assign phase masks to specified vectors on the CPU. The
             *          intention is that this phase mask is then copied to the GPU.
             */
            namespace cpu_phase_masks
            {
                /**
                 * @brief   Assigns a phase mask for a pseudo-randomly generated porous medium to the specified vector.
                 *          The remaining parameters are used to respect a potential decomposition of the domain.
                 * 
                 * @tparam  N   any `lbm::core::access::decomposed::NodeAccessor` from access.hpp
                 * 
                 * @param[in]       vertical_nodes_org          the amount of vertical nodes of the unextended domain
                 * @param[in]       horizontal_nodes_org        the amount of horizontal nodes of the unextended domain
                 * @param[in]       subdomain_vertical_nodes    the amount of vertical nodes per subdomain
                 * @param[in]       subdomain_horizontal_nodes  the amount of horizontal nodes per subdomain
                 * @param[in]       extended_horizontal_nodes   the amount of total horizontal nodes in the extended
                 *                                              simulation domain
                 * @param[in, out]  phase_information_cpu       the vector in which the phase mask will be stored
                 */
                template<core::access::decomposed::NodeAccessor N>
                void porous_medium_mask
                (
                    const unsigned int vertical_nodes_org, 
                    const unsigned int horizontal_nodes_org,
                    const unsigned int subdomain_vertical_nodes,
                    const unsigned int subdomain_horizontal_nodes,
                    const unsigned int extended_horizontal_nodes,
                    std::vector<int8_t> &phase_information_cpu
                )
                {
                    for(int y = 2; y < vertical_nodes_org - 2; ++y)
                    {
                        for(int x = 1; x <= horizontal_nodes_org - 2; ++x)
                        {
                            int one = 1 + (std::rand() % 10);
                            int two = 1 + (std::rand() % 10);
                            int value = 0;
                            if(x % one == one - 1 && y % two == two - 1) value = 1;
                                phase_information_cpu[
                                    N::get_index(
                                        x, 
                                        y, 
                                        subdomain_vertical_nodes, 
                                        subdomain_horizontal_nodes, 
                                        extended_horizontal_nodes
                                    )
                                ] = value;
                        }
                    }

                    for(int y = 1; y < vertical_nodes_org; y += vertical_nodes_org - 3)
                    {
                        for(int x = 1; x <= horizontal_nodes_org - 2; ++x)
                        {
                            phase_information_cpu[
                                N::get_index(
                                    x, 
                                    y, 
                                    subdomain_vertical_nodes, 
                                    subdomain_horizontal_nodes, 
                                    extended_horizontal_nodes
                                )
                            ] = 1;
                        }
                    }
                }

            } // ! cpu_phase_masks

// FUNCTIONS TO INIZIALIZE DISTRIBUTION VALUES ////////////////////////////////////////////////////////////////////////

            /**
             * @brief   Sets the standstill distribution values of the simulation domain on the GPU.
             * 
             * @tparam  A   any `core::access::AccessorConcept` from access.hpp 
             * 
             * @param[in, out]  simulation  this object contains the relevant pointers to device memory
             * @param[in]       queue       the SYCL queue that is to be used
             */
            template<access::AccessorConcept A> 
            void set_standstill_values
            (
                Simulation &simulation, 
                sycl::queue &queue
            )
            {
                std::vector<double> distribution_values_standstill = maxwell_boltzmann_distribution(0, 0, 1);
                double test [9];
                for(int i = 0; i < 9; ++i) test[i] = distribution_values_standstill[i];

                auto event = queue.submit(
                    [&](sycl::handler &cgh)
                    {
                        unsigned int total_node_count = simulation.domain->total_node_count;
                        double *distribution_values = simulation.data->distribution_values_0;

                        cgh.parallel_for(
                            total_node_count,                           
                            [=](const sycl::id<1> &idx)     
                            {
                                for(int direction = 0; direction < 9; ++direction)
                                {
                                    distribution_values[A::at(idx, direction, total_node_count)] = test[direction];
                                }
                            }  
                        );
                    }
                );
                event.wait(); 
            }

            /**
             * @brief   Sets the inlet and outlet properties as specified within the simulation object.
             * 
             * @tparam  A   any `core::access::AccessorConcept` from access.hpp
             * @tparam  N   any `lbm::core::access::decomposed::NodeAccessor` from access.hpp
             * 
             * @param[in, out]  simulation  this object contains the relevant pointers to device memory
             * @param[in]       queue       the SYCL queue that is to be used
             */
            template <access::AccessorConcept A, core::access::decomposed::NodeAccessor N>
            void set_inout_distribution_values
            (
                Simulation &simulation,
                sycl::queue &queue
            )
            {
                std::vector<double> distribution_values_cpu = 
                    maxwell_boltzmann_distribution(
                        simulation.properties->inlet_velocity_x, simulation.properties->inlet_velocity_y, simulation.properties->inlet_density
                    );
                
                std::vector<double> distribution_outlet = 
                    maxwell_boltzmann_distribution(
                        simulation.properties->outlet_velocity_x, simulation.properties->outlet_velocity_y, simulation.properties->outlet_density
                    );

                distribution_values_cpu.insert(distribution_values_cpu.end(), distribution_outlet.begin(), distribution_outlet.end());
                double test [18];
                for(int i = 0; i < 18; ++i) test[i] = distribution_values_cpu[i];

                // std::cout << "Inout distribution values: \n";
                // console::print_vector<double>(distribution_values_cpu, distribution_values_cpu.size());

                // double *distribution_values_buf_gpu = sycl::malloc_device<double>(distribution_values_cpu.size(), queue);

                // queue.copy(distribution_values_cpu.data(), distribution_values_buf_gpu, distribution_values_cpu.size()).wait();

                // std::cout << "Vertical range: [0, " <<  simulation.properties->vertical_nodes << "]\n";

                auto event = queue.submit(
                    [&](sycl::handler &cgh)
                    {
                        unsigned int horizontal_nodes_org =  simulation.properties->horizontal_nodes;
                        unsigned int total_node_count = simulation.domain->total_node_count;
                        unsigned int vertical_nodes = simulation.domain->vertical_nodes;
                        unsigned int horizontal_nodes = simulation.domain->horizontal_nodes;
                        unsigned int subdomain_horizontal_nodes = simulation.domain->subdomain_horizontal_nodes;
                        unsigned int subdomain_vertical_nodes = simulation.domain->subdomain_vertical_nodes;
                        double *distribution_values = simulation.data->distribution_values_0;

                        cgh.parallel_for(
                            sycl::range<2>(2, simulation.properties->vertical_nodes),
                            [=](const sycl::id<2> &id)     
                            {
                                unsigned int node = N::get_index(
                                    id.get(0) * (horizontal_nodes_org - 1), 
                                    id.get(1), 
                                    subdomain_vertical_nodes, 
                                    subdomain_horizontal_nodes, 
                                    horizontal_nodes
                                );

                                for(int i = 0; i < 9; ++i)
                                {
                                    distribution_values[A::at(node, i, total_node_count)] = test[i + 9 * id.get(0)];
                                }
                            }  
                        );
                    }
                );

                event.wait(); 
            }

            /**
             * @brief   Adds a the specified phase information stencil to a domain using the specified node accessor.
             * 
             * @tparam  N   any `lbm::core::access::decomposed::NodeAccessor` from access.hpp   
             * @tparam  S   any `lbm::core::domain_initialization::gpu_stencils::GPUStencil` 
             * 
             * @param[in]   simulation  this object contains a pointer to the phase information vector on the GPU
             * @param[in]   queue       the SYCL queue that is to be used
             */
            template<core::access::decomposed::NodeAccessor N, gpu_stencils::GPUStencil S> 
            void add_stencil(Simulation &simulation, sycl::queue &queue)
            {
                auto event = queue.submit(
                    [&](sycl::handler &cgh)
                    {
                        unsigned int vertical_nodes_org = simulation.properties->vertical_nodes;
                        unsigned int horizontal_nodes_org = simulation.properties->horizontal_nodes;
                        unsigned int extended_horizontal_nodes = 
                            simulation.domain->horizontal_nodes;
                        unsigned int subdomain_vertical_nodes = simulation.domain->subdomain_vertical_nodes;
                        unsigned int subdomain_horizontal_nodes = simulation.domain->subdomain_horizontal_nodes;
                        int8_t *phase_information = simulation.data->phase_information;

                        cgh.parallel_for(
                            sycl::range<2>(vertical_nodes_org - 4, horizontal_nodes_org - 2), 
                            [=](const sycl::id<2> &id)     
                            {
                                unsigned int x = id.get(1) + 1;
                                unsigned int y = id.get(0) + 2;
                                phase_information[
                                    N::get_index(
                                        x, 
                                        y, 
                                        subdomain_vertical_nodes, 
                                        subdomain_horizontal_nodes, 
                                        extended_horizontal_nodes
                                    )
                                ] = S::evaluate(x, y, vertical_nodes_org, horizontal_nodes_org);
                            }  
                        );
                    }
                );

                event.wait();
            }

            /**
             * @brief   Sets the up pipe flow environment object with a fluid in an equilibrium non-moving state.
             *          The domain consists of solid nodes on the upper and lower boundary, and further solid obstacles
             *          depending on the scenario specified within the simulation object.
             * 
             * @tparam  A   any `core::access::AccessorConcept` from access.hpp
             * @tparam  N   any `lbm::core::access::decomposed::NodeAccessor` from access.hpp   

             * @param[in, out]  simulation  reference to the structure containing all simulation data 
             * @param[in]       queue       the SYCL queue to which the values residing on the GPU belong
             * @param[in]       obstacle    what kind of obstacle is added to the domain
             */
            template<core::access::AccessorConcept A, core::access::decomposed::NodeAccessor N> 
            void setup_domain
            (
                Simulation &simulation, 
                sycl::queue &queue, 
                std::string obstacle = "Hagen-Poiseuille"
            )
            {
                domain_initialization::set_standstill_values<A>(simulation, queue);

                domain_initialization::set_inout_distribution_values<A, N>(simulation, queue);

                if
                (
                    simulation.properties->algorithm == "gpu-two-lattice" || 
                    simulation.properties->algorithm == "gpu-two-lattice-linear"
                )
                {
                    queue.copy(
                        simulation.data->distribution_values_0, 
                        simulation.data->distribution_values_1, 
                        9 * simulation.domain->total_node_count
                    ).wait();
                }

                // Set pipe boundaries
                if(!(obstacle == "porous")) {gpu_stencils::set_pipe_boundaries<N>(simulation, queue); }

                // std::vector<int8_t> phase_information(simulation.domain->total_node_count, 0);
                // queue.copy(
                //     simulation.data->phase_information, 
                //     phase_information.data(), 
                //     simulation.domain->total_node_count
                // ).wait();

                // std::cout << "Phase information after setting pipe boundaries: \n";
                // console::print_phase_vector(phase_information, simulation.domain->horizontal_nodes);

                if(obstacle == "Hagen-Poiseuille")  
                { add_stencil<N, gpu_stencils::HagenPoiseuilleStencil>(simulation, queue); }
                else if(obstacle == "walls")        
                { add_stencil<N, gpu_stencils::WallStencil>(simulation, queue); }
                else if(obstacle == "circle")       
                { add_stencil<N, gpu_stencils::CircleStencil>(simulation, queue); }
                else if(obstacle == "square")       
                { add_stencil<N, gpu_stencils::SquareStencil>(simulation, queue); }
                else if(obstacle == "plate")        
                { add_stencil<N, gpu_stencils::PlateStencil>(simulation, queue); }
                else if(obstacle == "skyscraper")   
                { add_stencil<N, gpu_stencils::SkyscraperStencil>(simulation, queue); }
                else if(obstacle == "wing")         
                { add_stencil<N, gpu_stencils::WingStencil>(simulation, queue); }
                else if(obstacle == "porous")       
                {
                    std::vector<int8_t> phase_information_cpu(simulation.domain->total_node_count, -1);
                    cpu_phase_masks::porous_medium_mask<N>
                    (
                        simulation.properties->vertical_nodes,
                        simulation.properties->horizontal_nodes,
                        simulation.domain->subdomain_vertical_nodes,
                        simulation.domain->subdomain_horizontal_nodes,
                        simulation.domain->horizontal_nodes,
                        phase_information_cpu
                    );

                    queue.copy(
                        phase_information_cpu.data(), 
                        simulation.data->phase_information, 
                        simulation.domain->total_node_count
                    ).wait();
                }

                // queue.copy(
                //     simulation.data->phase_information, 
                //     phase_information.data(), 
                //     simulation.decomposed_domain->total_node_count
                // ).wait();

                // std::cout << "Phase information after setting inner phases: \n";
                // console::print_phase_vector(phase_information, simulation.decomposed_domain->horizontal_nodes);
            }

        } // ! namespace domain_initialization

    } // ! namespace core

} // ! namespace lbm

#endif // ! DOMAIN_INITIALIZATION_HPP
