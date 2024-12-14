/**
 * @file        linear_gpu_two_lattice.hpp
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
 * @version     3.0
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LINEAR_GPU_TWO_LATTICE_HPP
#define LINEAR_GPU_TWO_LATTICE_HPP

// Console output
#include "../../../console/console_output.hpp"

// LBM core features
#include "../../../core/access.hpp"
#include "../../../core/boundaries.hpp"
#include "../../../core/collision.hpp"
#include "../../../core/domain_initialization.hpp"
#include "../../../core/simulation.hpp"

// Linear GPU boundary handling
//#include "../../boundaries/linear/linear_gpu_boundaries.hpp"

// Algorithm
#include "../../../execution/algorithm.hpp"

// SYCL constants
#include "../../sycl_constants.hpp"

// Kernels
#include "linear_two_lattice_kernels.hpp"

namespace lbm
{

    /**
     * @brief This namespace contains the GPU implementations of the two-lattice and the swap algorithm.
     */
    namespace gpu
    {

        /**
         * @brief This namespace contains the GPU implementation of the two-lattice algorithm.
         */
        namespace two_lattice
        {

            /**
             * @brief This namespace contains implementations using linear kernels.
             */
            namespace linear
            {

                /**
                 * @brief Class representation of the linear SYCL implementation of the two-lattice algorithm.
                 * 
                 * @tparam A any `core::access::experimental::AccessorConcept` from access.hpp
                 */
                template <core::access::experimental::AccessorConcept A>
                class LinearGpuTwoLattice : public execution::Algorithm
                {
                    private:

                    /**
                     * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                     *          the macroscopic observables in the process.
                     */
                    inline void stream_and_collide()
                    {
                        {
                            sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                            sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));
                            sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));
                            sycl::buffer<double, 1> densities_sycl(simulation->results->densities->data(), sycl::range<1>(simulation->results->densities->size()));
                            sycl::buffer<double, 1> x_velocities_sycl(simulation->results->x_velocities->data(), sycl::range<1>(simulation->results->x_velocities->size()));
                            sycl::buffer<double, 1> y_velocities_sycl(simulation->results->y_velocities->data(), sycl::range<1>(simulation->results->y_velocities->size()));

                            queue->submit
                            (
                                [&](sycl::handler &cgh)
                                {
                                    sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                    sycl::accessor<double, 1, constants::read> src_acc = src_sycl.get_access<constants::read>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> dst_acc = dst_sycl.get_access<constants::read_write>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> densities_acc = densities_sycl.get_access<constants::read_write>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> x_velocities_acc = x_velocities_sycl.get_access<constants::read_write>(cgh);
                                    sycl::accessor<double, 1, constants::read_write> y_velocities_acc = y_velocities_sycl.get_access<constants::read_write>(cgh);
                                    
                                    auto kernel = kernels::StreamCollideKernel<A>
                                    (
                                        phase_info_acc,
                                        src_acc,
                                        dst_acc,
                                        densities_acc,
                                        x_velocities_acc,
                                        y_velocities_acc,
                                        simulation->properties
                                    );
                                    cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                                }
                            );
                        }
                    }

                    /**
                     * @brief   Emplaces the values as preparation for the bounce-back scheme for the two-lattice algorithm.
                     */
                    void emplace_bounce_back()
                    {
                        sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                        sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                                sycl::accessor<double, 1, constants::read_write> src_acc = src_sycl.get_access<constants::read_write>(cgh);
                                
                                auto kernel = kernels::boundaries::EmplaceBounceBackKernel<A>(phase_info_acc, src_acc);
                                cgh.parallel_for(sycl::range<1>(simulation->properties->vertical_nodes * simulation->properties->horizontal_nodes), kernel);
                            }
                        );
                    }

                    /**
                     * @brief   Updates the inlets and outlets as preparation for the two-laatice algorithm.
                     */
                    void perform_inout_update()
                    {
                        sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));

                        queue->submit
                        (
                            [&](sycl::handler &cgh)
                            {
                                sycl::accessor<double, 1, constants::read_write> src_acc = src_sycl.get_access<constants::read_write>(cgh);
                                auto kernel = kernels::boundaries::InoutUpdateKernel<A>(src_acc, *simulation->properties);
                                cgh.parallel_for(sycl::range<2>(simulation->properties->vertical_nodes - 4, 2), kernel);
                            }
                        );
                    }

                    public:

                    /**
                     * @brief   Performs one iteration of the linear two-lattice algorithm. 
                     *          The instructions are enqueued in the specified queue.
                     */
                    inline void execute() override
                    {
                        future = hpx::async
                        (
                            [&]
                            {
                                emplace_bounce_back();
                                perform_inout_update();
                                stream_and_collide();
                                simulation->data->distribution_values_1.swap(simulation->data->distribution_values_0);
                            }
                        );
                    }

                    /**
                     * @brief   Runs the two-lattice algorithm with the specified properties.
                     *          It operates on the provided data.
                     */
                    void run()
                    {
                        for(auto step = 0; step < simulation->properties->time_steps; ++step)
                        {
                            emplace_bounce_back();
                            perform_inout_update();
                            stream_and_collide();

                            simulation->data->distribution_values_1.swap(simulation->data->distribution_values_0);
                        }
                    }

                    explicit LinearGpuTwoLattice(const sycl::queue &queue) : Algorithm(queue) {}; 
                };

                // /**
                //  * @brief This namespace contains the debug variants of all linear GPU two-lattice methods.
                //  * 
                //  */
                // namespace debug
                // {
                //     /**
                //      * @brief   Performs the streaming step of the two-lattice algorithm.
                //      * 
                //      * @tparam  LBMAccessor any child class of `core::access::LBMAccessorObject`
                //      * 
                //      * @param[in]       properties          the structure containing all relevant simulation properties
                //      * @param[in, out]  simulation->data     the data on which the algorithm operates
                //      * @param[in, out]  queue               the SYCL queue used for scheduling the corresponding kernel
                //      */
                //     template <class LBMAccessor> void stream
                //     (
                //         const core::Properties &properties,
                //         const core::SimulationData<LBMAccessor> &simulation->data,
                //         sycl::queue &queue  
                //     )
                //     {
                //         static_assert
                //         (
                //             std::is_base_of<core::access::LBMAccessorObject, LBMAccessor>::value, 
                //             "Template class must have base class core::access::LBMAccessorObject."
                //         );

                //         sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                //         sycl::buffer<double, 1> src_sycl(simulation->data->distribution_values_0->data(), sycl::range<1>(simulation->data->distribution_values_0->size()));
                //         sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));

                //         queue.submit
                //         (
                //             [&](sycl::handler &cgh)
                //             {
                //                 sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                //                 sycl::accessor<double, 1, constants::read> src_acc = src_sycl.get_access<constants::read>(cgh);
                //                 sycl::accessor<double, 1, constants::write> dst_acc = dst_sycl.get_access<constants::write>(cgh);
                                
                //                 auto kernel = linear::kernels::StreamKernel<LBMAccessor>
                //                 (
                //                     phase_info_acc,
                //                     src_acc,
                //                     dst_acc,
                //                     *simulation->data->lbm_accessor,
                //                     properties
                //                 );

                //                 cgh.parallel_for(sycl::range<1>(properties.vertical_nodes * properties.horizontal_nodes), kernel);
                //             }
                //         );
                //     }

                //     /**
                //      * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                //      *          the macroscopic observables in the process.
                //      * 
                //      * @tparam  LBMAccessor any child class of `core::access::LBMAccessorObject`
                //      * 
                //      * @param[in]       properties          the structure containing all relevant simulation properties
                //      * @param[in, out]  simulation->data     the data on which the algorithm operates
                //      * @param[in, out]  simulation->results  the macroscopic simulation results produced by the algorithm
                //      * @param[in, out]  queue               the SYCL queue used for scheduling the corresponding kernel
                //      */
                //     template <class LBMAccessor> void update_macroscopic_observables
                //     (
                //         const core::Properties &properties,
                //         const core::SimulationData<LBMAccessor> &simulation->data,
                //         const core::SimulationResults &simulation->results,
                //         sycl::queue &queue  
                //     )
                //     {
                //         sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                //         sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));
                //         sycl::buffer<double, 1> densities_sycl(simulation.results->densities->data(), sycl::range<1>(simulation.results->densities->size()));
                //         sycl::buffer<double, 1> x_velocities_sycl(simulation.results->x_velocities->data(), sycl::range<1>(simulation.results->x_velocities->size()));
                //         sycl::buffer<double, 1> y_velocities_sycl(simulation.results->y_velocities->data(), sycl::range<1>(simulation.results->y_velocities->size()));

                //         queue.submit
                //         (
                //             [&](sycl::handler &cgh)
                //             {
                //                 sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                //                 sycl::accessor<double, 1, constants::read> dst_acc = dst_sycl.get_access<constants::read>(cgh);

                //                 sycl::accessor<double, 1, constants::write> densities_acc = densities_sycl.get_access<constants::write>(cgh);
                //                 sycl::accessor<double, 1, constants::write> x_velocities_acc = x_velocities_sycl.get_access<constants::write>(cgh);
                //                 sycl::accessor<double, 1, constants::write> y_velocities_acc = y_velocities_sycl.get_access<constants::write>(cgh);
                                
                //                 auto kernel = linear::kernels::MacroscopicObservablesKernel<LBMAccessor>
                //                 (
                //                     phase_info_acc,
                //                     dst_acc,
                //                     densities_acc,
                //                     x_velocities_acc,
                //                     y_velocities_acc,
                //                     *simulation->data->lbm_accessor,
                //                     properties
                //                 );
                //                 cgh.parallel_for(sycl::range<1>(properties.vertical_nodes * properties.horizontal_nodes), kernel);
                //             }
                //         );
                //     }

                //     /**
                //      * @brief   Performs the collision step of the two-lattice algorithm.
                //      * 
                //      * @tparam  LBMAccessor any child class of `core::access::LBMAccessorObject`
                //      * 
                //      * @param[in]       properties          the structure containing all relevant simulation properties
                //      * @param[in, out]  simulation->data     the data on which the algorithm operates
                //      * @param[in, out]  simulation->results  the macroscopic simulation results produced by the algorithm
                //      * @param[in, out]  queue               the SYCL queue used for scheduling the corresponding kernel
                //      */
                //     template <class LBMAccessor> void collide
                //     (
                //         const core::Properties &properties,
                //         const core::SimulationData<LBMAccessor> &simulation->data,
                //         const core::SimulationResults &simulation->results,
                //         sycl::queue &queue  
                //     )
                //     {
                //         sycl::buffer<uint8_t, 1> phase_info_sycl(simulation->data->phase_information->begin(), simulation->data->phase_information->end());
                //         sycl::buffer<double, 1> dst_sycl(simulation->data->distribution_values_1->data(), sycl::range<1>(simulation->data->distribution_values_1->size()));
                //         sycl::buffer<double, 1> densities_sycl(simulation.results->densities->data(), sycl::range<1>(simulation.results->densities->size()));
                //         sycl::buffer<double, 1> x_velocities_sycl(simulation.results->x_velocities->data(), sycl::range<1>(simulation.results->x_velocities->size()));
                //         sycl::buffer<double, 1> y_velocities_sycl(simulation.results->y_velocities->data(), sycl::range<1>(simulation.results->y_velocities->size()));

                //         queue.submit
                //         (
                //             [&](sycl::handler &cgh)
                //             {
                //                 sycl::accessor<uint8_t, 1, constants::read> phase_info_acc = phase_info_sycl.get_access<constants::read>(cgh);
                //                 sycl::accessor<double, 1, constants::read_write> dst_acc = dst_sycl.get_access<constants::read_write>(cgh);

                //                 sycl::accessor<double, 1, constants::write> densities_acc = densities_sycl.get_access<constants::write>(cgh);
                //                 sycl::accessor<double, 1, constants::write> x_velocities_acc = x_velocities_sycl.get_access<constants::write>(cgh);
                //                 sycl::accessor<double, 1, constants::write> y_velocities_acc = y_velocities_sycl.get_access<constants::write>(cgh);
                                
                //                 auto kernel = linear::kernels::CollideKernel<LBMAccessor>
                //                 (
                //                     phase_info_acc,
                //                     dst_acc,
                //                     densities_acc,
                //                     x_velocities_acc,
                //                     y_velocities_acc,
                //                     *simulation->data->lbm_accessor,
                //                     properties
                //                 );
                //                 cgh.parallel_for(sycl::range<1>(properties.vertical_nodes * properties.horizontal_nodes), kernel);
                //             }
                //         );
                //     }

                //     /**
                //      * @brief   Performs the streaming and collision step of the two-lattice algorithm and updates
                //      *          the macroscopic observables in the process.
                //      *          This debug variant prints several pieces of information to the console.
                //      * 
                //      * @tparam  LBMAccessor any child class of `core::access::LBMAccessorObject`
                //      * 
                //      * @param[in]       properties          the structure containing all relevant simulation properties
                //      * @param[in]       iteration           the current time step
                //      * @param[in, out]  simulation->data     the data on which the algorithm operates
                //      * @param[in, out]  simulation->results  the macroscopic simulation results produced by the algorithm
                //      * @param[in, out]  queue               the SYCL queue used for scheduling the corresponding kernel
                //      */
                //     template <class LBMAccessor> void stream_and_collide
                //     (
                //         const core::Properties &properties,
                //         const core::SimulationData<LBMAccessor> &simulation->data,
                //         const core::SimulationResults &simulation->results,
                //         sycl::queue &queue  
                //     )
                //     {
                //         static_assert
                //         (
                //             std::is_base_of<core::access::LBMAccessorObject, LBMAccessor>::value, 
                //             "Template class must have base class core::access::LBMAccessorObject."
                //         );

                //         stream(properties, simulation->data, queue);

                //         std::cout << "\033[36mDestination lattice after streaming: \n"
                //                 << "-------------------------------------------------------------------------------\n\033[0m";
                //         lbm::console::print_distribution_values(*simulation->data->distribution_values_1, *simulation->data->lbm_accessor);
                    
                //         update_macroscopic_observables(properties, simulation->data, simulation->results, queue);

                //         std::cout << "Velocities: \n"
                //                 << "-------------------------------------------------------------------------------\n";
                //         lbm::console::print_velocities(properties, *simulation.results->x_velocities, *simulation.results->y_velocities, 0);

                //         std::cout << "Densities: \n"
                //                 << "-------------------------------------------------------------------------------\n";
                //         lbm::console::print_densities(properties, *simulation.results->densities, 0);

                //         collide(properties, simulation->data, simulation->results, queue);

                //         std::cout << "\033[36mDestination lattice after collision: \n"
                //                 << "-------------------------------------------------------------------------------\n\033[0m";
                //         lbm::console::print_distribution_values(*simulation->data->distribution_values_1, *simulation->data->lbm_accessor);
                //     }

                //     /**
                //      * @brief   Runs the two-lattice algorithm with the specified properties.
                //      *          It operates on the provided data.
                //      *          This debug variant prints several pieces of information to the console.
                //      * 
                //      * @tparam  LBMAccessor any child class of `core::access::LBMAccessorObject`
                //      * 
                //      * @param[in]       properties          the structure containing all relevant simulation properties
                //      * @param[in, out]  simulation->data     the data on which the algorithm operates
                //      */
                //     template <class LBMAccessor> void run
                //     (  
                //         const core::Properties &properties,
                //         core::SimulationData<LBMAccessor> &simulation->data
                //     )
                //     {
                //         static_assert
                //         (
                //             std::is_base_of<core::access::LBMAccessorObject, LBMAccessor>::value, 
                //             "Template class must have base class core::access::LBMAccessorObject."
                //         );

                //         std::cout << "\033[36mNow running GPU two-lattice for " << properties.time_steps << " iterations.\033[0m\n\n";
                //         std::cout.flush();
                        
                //         lbm::core::SimulationResults current_simulation_results(properties.domain_node_count);

                //         lbm::core::SimulationResults all_simulation_results({}, {}, {}, {}, {});

                //         all_simulation.results->densities->reserve(properties.time_steps * properties.domain_node_count);
                //         all_simulation.results->pressures->reserve(properties.time_steps * properties.domain_node_count);
                //         all_simulation.results->x_velocities->reserve(properties.time_steps * properties.domain_node_count);
                //         all_simulation.results->y_velocities->reserve(properties.time_steps * properties.domain_node_count);

                //         sycl::default_selector device_selector; 
                //         sycl::queue queue(device_selector);

                //         std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << "\n";   

                //         for(auto step = 0; step < properties.time_steps; ++step)
                //         {
                //             std::cout << "\033[36m===============================================================================\n"
                //                     << "Iteration " << step << "\n"
                //                     << "===============================================================================\033[0m\n\n";

                //             linear::emplace_bounce_back(properties, simulation->data, queue);

                //             std::cout << "\033[36mSource lattice after emplacing bounce-back values: \n"
                //                     << "-------------------------------------------------------------------------------\033[0m\n";
                //             lbm::console::print_distribution_values(*simulation->data->distribution_values_0, *simulation->data->lbm_accessor);

                //             linear::perform_inout_update(properties, simulation->data, queue);

                //             std::cout << "\033[36mDestination lattice after updating inlets and outlets: \n"
                //                     << "-------------------------------------------------------------------------------\033[0m\n";
                //             lbm::console::print_distribution_values(*simulation->data->distribution_values_0, *simulation->data->lbm_accessor);

                //             stream_and_collide(properties, simulation->data, current_simulation_results, queue);

                //             std::cout << "\033[36mFinished iteration " << step << "\033[0m \n\n\n";

                //             // Swap source and destination lattice
                //             simulation->data->distribution_values_1.swap(simulation->data->distribution_values_0);

                //             all_simulation.results->densities->insert(all_simulation.results->densities->end(), current_simulation.results->densities->begin(), current_simulation.results->densities->end());
                //             all_simulation.results->pressures->insert(all_simulation.results->pressures->end(), current_simulation.results->pressures->begin(), current_simulation.results->pressures->end());
                //             all_simulation.results->x_velocities->insert(all_simulation.results->x_velocities->end(), current_simulation.results->x_velocities->begin(), current_simulation.results->x_velocities->end());
                //             all_simulation.results->y_velocities->insert(all_simulation.results->y_velocities->end(), current_simulation.results->y_velocities->begin(), current_simulation.results->y_velocities->end());
                //         }

                //         std::cout << "\033[36mAll done, exiting simulation. \033[0m\n";
                //         lbm::console::print_simulation_results(properties, all_simulation_results);
                //     }

                // } // ! namespace debug

            } // ! namespace linear

        } // ! namespace two_lattice

    } // ! namespace gpu

} // ! namespace lbm

#endif // ! GPU_TWO_LATTICE_HPP
