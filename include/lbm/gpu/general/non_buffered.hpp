/**
 * @file        general_non_buffered.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       This namespace contains kernels that can be used in multiple non-buffered lattice Boltzmann
 *              implementations.
 *
 * @version     1.3
 *
 * @date        April 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#ifndef LBM_GENERAL_NON_BUFFERED_HPP
#define LBM_GENERAL_NON_BUFFERED_HPP

// Dependencies on other LBM core features
#include "../../core/access.hpp"
#include "../../core/constants.hpp"
#include "../../core/simulation.hpp"

// SYCL
#include <sycl/sycl.hpp>

namespace lbm
{

namespace gpu
{

namespace general
{

/**
 * @brief   This namespace contains kernels that can be used in multiple non-buffered lattice Boltzmann
 *          implementations.
 */
namespace non_buffered
{

/**
 * @brief Kernel for updating the outlets according to the boundary condition by Zou and He.
 *
 * @tparam  A any `core::access::AccessorConcept` from access.hpp
 */
template <core::access::AccessorConcept A>
class OutletUpdateKernel
{
  private:
    real_type *source;
    real_type *y_velocities;
    real_type outlet_density;
    real_type outlet_density_inverse;
    unsigned int horizontal_nodes;
    unsigned int horizontal_nodes_original;
    unsigned int vertical_nodes;

  public:
    /**
     * @brief   Constructor for a new `OutletUpdateKernel` object.
     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
     *
     * @param[in]   simulation  the structure containing all simulation data
     */
    OutletUpdateKernel(const core::Simulation &simulation) :
        source(simulation.data->distribution_values_0),
        y_velocities(simulation.results->y_velocities_gpu),
        outlet_density(simulation.properties->outlet_density),
        outlet_density_inverse(1 / simulation.properties->outlet_density),
        vertical_nodes(simulation.domain->vertical_nodes),
        horizontal_nodes(simulation.domain->horizontal_nodes),
        horizontal_nodes_original(simulation.properties->horizontal_nodes)
    { }

    /**
     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
     *
     * @param[in]   id  a one-dimensional ID of a SYCL work item processing this kernel
     */
    void operator()(const sycl::item<1> &id) const
    {
        unsigned int missing[3];

        real_type f_inverse[3];
        real_type f_1 = 0;
        real_type f_4 = 0;
        real_type f_7 = 0;

        for (int i = 0; i < 3; ++i)
        {
            missing[i] = 3 * i;
        }

        // Setup actual coordinate, y coordinate: offset by two
        int y = id + 2;

        unsigned int current_border_node =
            core::access::get_node_index(horizontal_nodes_original - 2, y, horizontal_nodes);

        f_1 = source[A::at(core::access::get_neighbor(current_border_node, 7, horizontal_nodes),
                           1,
                           horizontal_nodes * vertical_nodes)];

        f_4 = source[A::at(current_border_node, 4, horizontal_nodes * vertical_nodes)];

        f_7 = source[A::at(core::access::get_neighbor(current_border_node, 1, horizontal_nodes),
                           7,
                           horizontal_nodes * vertical_nodes)];

        for (int i = 0; i < 3; ++i)
        {
            f_inverse[i] = source[A::at(core::access::get_neighbor(current_border_node, missing[i], horizontal_nodes),
                                        8 - missing[i],
                                        horizontal_nodes * vertical_nodes)];
        }

        real_type x_velocity = -outlet_density + (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

        real_type y_velocity =
            y_velocities[core::access::get_result_index(horizontal_nodes_original - 3, y, horizontal_nodes_original)];

        source[A::at(core::access::get_neighbor(current_border_node, 8 - missing[0], horizontal_nodes),
                     missing[0],
                     horizontal_nodes * vertical_nodes)] =
            f_inverse[0] - 0.5 * (f_1 - f_7) - 1.0 / 6 * x_velocity - 0.5 * y_velocity;

        source[A::at(core::access::get_neighbor(current_border_node, 8 - missing[1], horizontal_nodes),
                     missing[1],
                     horizontal_nodes * vertical_nodes)] = f_inverse[1] - (2.0 / 3) * x_velocity;

        source[A::at(core::access::get_neighbor(current_border_node, 8 - missing[2], horizontal_nodes),
                     missing[2],
                     horizontal_nodes * vertical_nodes)] =
            f_inverse[2] + 0.5 * (f_1 - f_7) - 1.0 / 6 * x_velocity + 0.5 * y_velocity;
    }
};

}  // namespace non_buffered

}  // namespace general

}  // namespace gpu

}  // namespace lbm

#endif
