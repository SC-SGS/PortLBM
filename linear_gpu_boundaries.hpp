/**
 * @file        linear_gpu_boundaries.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       In this header file, linear kernels for treating boundary conditions are declared and defined.
 * 
 * @version     1.0
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef BOUNDARIES_GPU_HPP
#define BOUNDARIES_GPU_HPP

// Dependencies on other LBM core features
#include "../../../core/access.hpp"
#include "../../../core/boundaries.hpp"
#include "../../../core/collision.hpp"
#include "../../../core/domain_initialization.hpp"
#include "../../../core/simulation.hpp"

// LBM console output
#include "../../../console/console_output.hpp"

// SYCL constants
#include "../../sycl_constants.hpp"

#include <sycl/sycl.hpp>

namespace lbm
{
    namespace gpu
    {

        /**
         * @brief This namespace contains GPU-based implementations of various methods concerning boundaries.
         */
        namespace boundaries
        {

            /**
             * @brief This namespace contains linear GPU kernels for boundary treatment.
             */
            namespace linear
            {

                /**
                 * @brief   Kernel for emplacing the bounce-back values.
                 * 
                 * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                 */
                template<access::experimental::AccessorConcept A>
                class EmplaceBounceBackKernel
                {
                    private:

                    sycl::accessor<uint8_t, 1, constants::read> phase_info_accessor;
                    sycl::accessor<double, 1, constants::read_write> distribution_values_accessor;
                    unsigned int horizontal_nodes;

                    public:

                    EmplaceBounceBackKernel
                    (
                        const sycl::accessor<uint8_t, 1, constants::read> &phase_info_accessor,
                        const sycl::accessor<double, 1, constants::read_write> &distribution_values_accessor,
                        const unsigned int horizontal_nodes
                    )
                    : 
                    phase_info_accessor(phase_info_accessor), 
                    distribution_values_accessor(distribution_values_accessor),
                    horizontal_nodes(horizontal_nodes)
                    {}

                    void operator()(const sycl::item<1> &item) const
                    {
                        auto id = item.get_linear_id();

                        if(phase_info_accessor[id])
                        {
                            for(const auto& dir : core::constants::all_directions)
                            {
                                distribution_values_accessor[A.at(id, dir)] =
                                distribution_values_accessor[A.at(core::access::get_neighbor(id, dir, horizontal_nodes), core::invert_direction(dir))];            
                            }
                        }
                    }
                };

                /**
                 * @brief Kernel for updating the inlets and outlets.
                 * 
                 * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                 */
                template<access::experimental::AccessorConcept A>
                class InoutUpdateKernel
                {

                    private:

                    sycl::accessor<double, 1, constants::read_write> distribution_values_accessor;
                    double densities[2];

                    public:

                    InoutUpdateKernel
                    (
                        const sycl::accessor<double, 1, constants::read_write> &distribution_values_accessor,
                        const core::Properties &properties
                    )
                    : 
                    distribution_values_accessor(distribution_values_accessor),
                    densities(properties.inlet_density, properties.outlet_density)
                    {}

                    void operator()(const sycl::id<2> &id) const
                    {
                        auto id_y = id[0]; // in {0,1, ... , horizontal_nodes - 5}
                        auto id_x = id[1]; // in {0,1}

                        unsigned int missing[3];

                        double f_inverse[3];
                        double f_1 = 0;
                        double f_4 = 0;
                        double f_7 = 0;

                        double density = densities[id_x];

                        for(int i = 0; i < 3; ++i)
                        {
                            // x = 0: inlet boundary, missing directions: 2, 5, 8
                            // x = 1: outlet boundary, missing directions: 0, 3, 6
                            missing[i] = id_x * 3 * i + (1 - id_x) * (2 + 3 * i);
                        }

                        // Setup actual coordinates
                        id_y = id_y + 2;                                         // y coordinate: offset by two
                        id_x = 1 + id_x * (properties.horizontal_nodes - 3);   // x coordinate: 0 -> 1, 1 -> horizontal_nodes - 2

                        unsigned int current_border_node = core::access::get_node_index(id_x, id_y, properties.horizontal_nodes);

                        f_1 = distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 7, properties.horizontal_nodes), 1)];
                        f_4 = distribution_values_accessor[A.at(current_border_node, 4)];
                        f_7 = distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 1, properties.horizontal_nodes), 7)];

                        for(int i = 0; i < 3; ++i)
                        {
                            f_inverse[i] = distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, missing[i], properties.horizontal_nodes), 8 - missing[i])];
                        }

                        double x_velocity = 1 - (1 / density) * (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

                        distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 8 - missing[0], properties.horizontal_nodes), missing[0])]
                            = f_inverse[0] - 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;
                        distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 8 - missing[1], properties.horizontal_nodes), missing[1])]
                            = f_inverse[1] + (2.0/3) * density * x_velocity;
                        distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 8 - missing[2], properties.horizontal_nodes), missing[2])]
                            = f_inverse[2] + 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;
                    }
                };

                /**
                 * @brief   Debug method that enacts a CPU-based computation that is semantically identical to `lbm::gpu::boundaries::linear::InoutUpdateKernel::operator()`.
                 * 
                 * @tparam  A any `core::access::experimental::AccessorConcept` from access.hpp 
                 * 
                 * @param densities 
                 * @param properties 
                 * @param distribution_values_accessor 
                 */
                template<access::experimental::AccessorConcept A> 
                void inout_update_debugger
                (
                    const double densities[2],
                    const core::Properties &properties,
                    std::vector<double> &distribution_values_accessor
                )
                {
                    std::cout << "Number of outer iterations (y): " << properties.vertical_nodes - 4 << "\n";
                    for(int outer = 0; outer < properties.vertical_nodes - 4; ++outer)
                    {
                        for(int inner = 0; inner < 2; ++inner)
                        {
                            int id[2] = {outer, inner};

                            auto id_y = id[0]; // in {0,1, ... , horizontal_nodes - 5}
                            auto id_x = id[1]; // in {0,1}

                            unsigned int missing[3];

                            double f_inverse[3];
                            double f_1 = 0;
                            double f_4 = 0;
                            double f_7 = 0;

                            double density = densities[id_x];
                            std::cout << "Density: " << density << "\n";
                            std::cout << std::setprecision(6) << std::fixed;
                            std::cout << "Missing variables: ";
                            for(int i = 0; i < 3; ++i)
                            {
                                // x = 0: inlet boundary, missing directions: 2, 5, 8
                                // x = 1: outlet boundary, missing directions: 0, 3, 6
                                missing[i] = id_x * 3 * i + (1 - id_x) * (2 + 3 * i);
                                std::cout << missing[i] << " ";
                            }
                            std::cout << "\n";

                            // Setup actual coordinates
                            id_y = id_y + 2;                                         // y coordinate: offset by two
                            id_x = 1 + id_x * (properties.horizontal_nodes - 3);   // x coordinate: 0 -> 1, 1 -> horizontal_nodes - 2
                            unsigned int current_border_node = core::access::get_node_index(id_x, id_y, properties.horizontal_nodes);
                            std::cout << "Treating node with coordinates (x, y) = (" << id_x << ", " << id_y << "), with node index " << current_border_node << "\n";
                            
                            unsigned int known_directions[6] = {1, 4, 7, 8 - missing[0], 8 - missing[1], 8 - missing[2]};
                            
                            std::cout << "Known directions: ";
                            for(int i = 0; i < 6; ++i)
                            {
                                std::cout << known_directions[i] << " ";
                            }
                            std::cout << "\n";

                            f_1 = distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 7, properties.horizontal_nodes), 1)];
                            f_4 = distribution_values_accessor[A.at(current_border_node, 4)];
                            f_7 = distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 1, properties.horizontal_nodes), 7)];

                            for(int i = 0; i < 3; ++i)
                            {
                                f_inverse[i] = distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, missing[i], properties.horizontal_nodes), 8 - missing[i])];
                                std::cout << "The corresponding neighbor of this node in direction " << missing[i] << " is believed to have index " << core::access::get_neighbor(current_border_node, missing[i], properties.horizontal_nodes) << "\n";
                            }

                            std::cout << "\n";

                            std::cout << "Known distribution values: \n";
                            std::cout << "f_1 = " << f_1 << "\n";
                            std::cout << "f_4 = " << f_4 << "\n";
                            std::cout << "f_7 = " << f_7 << "\n";

                            for(int i = 0; i < 3; ++i)
                            {
                                std::cout << "f_" << 8 - missing[i] << " = " << f_inverse[i] << "\n";
                            }

                            std::cout << "\n";

                            double x_velocity = 1 - (1 / density) * (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

                            std::cout << "Velocity: " << x_velocity << "\n";

                            distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 8 - missing[0], properties.horizontal_nodes), missing[0])]
                                = f_inverse[0] - 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;
                            distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 8 - missing[1], properties.horizontal_nodes), missing[1])]
                                = f_inverse[1] + (2.0/3) * density * x_velocity;
                            distribution_values_accessor[A.at(core::access::get_neighbor(current_border_node, 8 - missing[2], properties.horizontal_nodes), missing[2])]
                                = f_inverse[2] + 0.5 * (f_1 - f_7) + 1.0/6 * density * x_velocity;

                            std::cout << "Calculated distribution values: \n";

                            for(int i = 0; i < 3; ++i)
                            {
                                std::cout << "f[" << missing[i] << "] = " 
                                          << distribution_values_accessor[lbm_accessor(core::access::get_neighbor(current_border_node, 8 - missing[i], properties.horizontal_nodes), missing[i])] << "\n";
                            }
                            std::cout << "\n";
                            std::cout << std::setprecision(3) << std::fixed;
                        }
                    }
                }

            } // ! namespace linear

        } // ! namespace boundaries

    } // ! namespace gpu

} // ! namesapce lbm

#endif