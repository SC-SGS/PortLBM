/**
 * @file        general_non_buffered.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       This namespace contains kernels that can be used in multiple buffered lattice Boltzmann implementations.
 *
 * @version     1.4
 *
 * @date        April 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 *
 */

#ifndef LBM_GENERAL_BUFFERED_HPP
#define LBM_GENERAL_BUFFERED_HPP

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
 * @brief   This namespace contains kernels that can be used in multiple buffered lattice Boltzmann
 *          implementations.
 */
namespace buffered
{

// Buffer updates /////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Updates all horizontal buffers by copying the bordering values from the nodes above and
 *          below facing the boundary.
 *
 * @tparam  A any `core::access::AccessorConcept` from access.hpp
 */
template <core::access::AccessorConcept A>
class HorizontalCopyToBufferKernel
{
  private:
    int8_t *phase_information;
    real_type *distribution_values;

    unsigned int vertical_nodes;
    unsigned int horizontal_nodes;
    unsigned int vertical_subdomain_size;

  public:
    /**
     * @brief   Constructor for a new `HorizontalCopyToBufferKernel` object.
     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
     *
     * @param[in]   simulation  the structure containing all simulation data
     */
    explicit HorizontalCopyToBufferKernel(const core::Simulation &simulation) :
        phase_information(simulation.data->phase_information),
        distribution_values(simulation.data->distribution_values_0),
        vertical_nodes(simulation.domain->vertical_nodes),
        horizontal_nodes(simulation.domain->horizontal_nodes),
        vertical_subdomain_size(simulation.domain->subdomain_vertical_nodes)
    { }

    /**
     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
     *
     * @param[in]   item    a two-dimensional SYCL work item processing this kernel
     */
    void operator()(const sycl::item<2> &item) const
    {
        auto global_id_x = item.get_id(1);
        auto global_id_y = (item.get_id(0) + 1) * (vertical_subdomain_size + 1);
        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);
        unsigned int neighbor_top = lbm::core::access::get_neighbor(linear_index, 7, horizontal_nodes);
        unsigned int neighbor_bottom = lbm::core::access::get_neighbor(linear_index, 1, horizontal_nodes);

        for (int direction = 0; direction < 3; ++direction)
        {
            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                distribution_values[A::at(neighbor_top, direction, horizontal_nodes * vertical_nodes)];
        }

        for (int direction = 6; direction < 9; ++direction)
        {
            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                distribution_values[A::at(neighbor_bottom, direction, horizontal_nodes * vertical_nodes)];
        }
    }
};

/**
 * @brief   Updates all vertical buffers by copying the bordering values from the nodes left and right
 *          of the buffer.
 *
 * @tparam  A any `core::access::AccessorConcept` from access.hpp
 */
template <core::access::AccessorConcept A>
class VerticalCopyToBufferKernel
{
  private:
    int8_t *phase_information;
    real_type *distribution_values;

    unsigned int vertical_nodes;
    unsigned int horizontal_nodes;
    unsigned int horizontal_subdomain_size;

  public:
    /**
     * @brief   Constructor for a new `VerticalCopyToBufferKernel` object.
     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
     *
     * @param[in]   simulation  the structure containing all simulation data
     */
    explicit VerticalCopyToBufferKernel(const core::Simulation &simulation) :
        phase_information(simulation.data->phase_information),
        distribution_values(simulation.data->distribution_values_0),
        vertical_nodes(simulation.domain->vertical_nodes),
        horizontal_nodes(simulation.domain->horizontal_nodes),
        horizontal_subdomain_size(simulation.domain->subdomain_horizontal_nodes)
    { }

    /**
     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
     *
     * @param[in]   item    a two-dimensional SYCL work item processing this kernel
     */
    void operator()(const sycl::item<2> &item) const
    {
        auto global_id_x = (item.get_id(1) + 1) * (horizontal_subdomain_size + 1);
        auto global_id_y = item.get_id(0);
        auto linear_index = core::access::get_node_index(global_id_x, global_id_y, horizontal_nodes);
        unsigned int neighbor_left = lbm::core::access::get_neighbor(linear_index, 3, horizontal_nodes);
        unsigned int neighbor_right = lbm::core::access::get_neighbor(linear_index, 5, horizontal_nodes);

        for (int direction = 2; direction < 9; direction += 3)  //(const auto& direction : {2, 5, 8})
        {
            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                distribution_values[A::at(neighbor_left, direction, horizontal_nodes * vertical_nodes)];
        }

        for (int direction = 0; direction < 9; direction += 3)  //(const auto& direction : {0, 3, 6})
        {
            distribution_values[A::at(linear_index, direction, horizontal_nodes * vertical_nodes)] =
                distribution_values[A::at(neighbor_right, direction, horizontal_nodes * vertical_nodes)];
        }
    }
};

// Inlets and outlets /////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Kernel for updating the inlets that are overwritten by the swap algorithm.
 *
 * @tparam  A any `core::access::AccessorConcept` from access.hpp
 */
template <core::access::AccessorConcept A>
class InletUpdateKernel
{
  private:
    real_type *distribution_values;
    real_type distribution[9];

    real_type inlet_density;
    real_type inlet_velocity_x;
    real_type inlet_velocity_y;

    unsigned int horizontal_nodes;
    unsigned int horizontal_nodes_org;
    unsigned int total_nodes;

    unsigned int subdomain_vertical_nodes;
    unsigned int subdomain_horizontal_nodes;

  public:
    /**
     * @brief   Constructor for a new `InletUpdateKernel` object.
     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
     *
     * @param[in]   simulation      the structure containing all simulation data
     * @param[in]   distribution    an array containing the distribution values of all inlet nodes
     */
    InletUpdateKernel(const core::Simulation &simulation, const real_type distribution[9]) :
        distribution_values(simulation.data->distribution_values_0),
        inlet_density(simulation.properties->inlet_density),
        inlet_velocity_x(simulation.properties->inlet_velocity_x),
        inlet_velocity_y(simulation.properties->inlet_velocity_y),
        horizontal_nodes(simulation.domain->horizontal_nodes),
        horizontal_nodes_org(simulation.properties->horizontal_nodes),
        total_nodes(simulation.domain->total_node_count),
        subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes),
        subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes)
    {
        memcpy(this->distribution, distribution, 9 * sizeof(real_type));
    }

    /**
     * @brief This overloaded operator is implicitly called to launch the kernel for various work items.
     *
     * @param[in]   id  a one-dimensional ID of a SYCL work item processing this kernel
     */
    void operator()(const sycl::item<1> &id) const
    {
        unsigned int current_node = core::access::decomposed::BufferedNodeAccess::get_index(
            1, id.get_id(0) + 1, subdomain_vertical_nodes, subdomain_horizontal_nodes, horizontal_nodes);

        current_node = core::access::get_neighbor(current_node, 3, horizontal_nodes);

        for (int i = 0; i < 9; ++i)
        {
            distribution_values[A::at(current_node, i, total_nodes)] = distribution[i];
        }
    }
};

/**
 * @brief Kernel for updating the outlets according to the boundary condition by Zou and He.
 *
 * @tparam  A any `core::access::AccessorConcept` from access.hpp
 */
template <core::access::AccessorConcept A>
class OutletUpdateKernel
{
  private:
    real_type *distribution_values;

    real_type *y_velocities;

    real_type outlet_density;
    real_type outlet_density_inverse;

    unsigned int horizontal_nodes;
    unsigned int horizontal_nodes_original;
    unsigned int vertical_nodes;

    unsigned int subdomain_vertical_nodes;
    unsigned int subdomain_horizontal_nodes;

  public:
    /**
     * @brief   Constructor for a new `OutletUpdateKernel` object.
     *          Create an instance of this kernel and pass it to `cgh.parallel_for(...)`.
     *
     * @param[in]   simulation  the structure containing all simulation data
     */
    OutletUpdateKernel(const core::Simulation &simulation) :
        distribution_values(simulation.data->distribution_values_0),
        y_velocities(simulation.results->y_velocities_gpu),
        outlet_density(simulation.properties->outlet_density),
        outlet_density_inverse(1 / simulation.properties->outlet_density),
        vertical_nodes(simulation.domain->vertical_nodes),
        horizontal_nodes(simulation.domain->horizontal_nodes),
        horizontal_nodes_original(simulation.properties->horizontal_nodes),
        subdomain_vertical_nodes(simulation.domain->subdomain_vertical_nodes),
        subdomain_horizontal_nodes(simulation.domain->subdomain_horizontal_nodes)
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

        unsigned int current_border_node = core::access::decomposed::BufferedNodeAccess::get_index(
            horizontal_nodes_original - 3, y, subdomain_vertical_nodes, subdomain_horizontal_nodes, horizontal_nodes);

        current_border_node = core::access::get_neighbor(current_border_node, 5, horizontal_nodes);

        f_1 = distribution_values[A::at(core::access::get_neighbor(current_border_node, 7, horizontal_nodes),
                                        1,
                                        horizontal_nodes * vertical_nodes)];

        f_4 = distribution_values[A::at(current_border_node, 4, horizontal_nodes * vertical_nodes)];

        f_7 = distribution_values[A::at(core::access::get_neighbor(current_border_node, 1, horizontal_nodes),
                                        7,
                                        horizontal_nodes * vertical_nodes)];

        for (int i = 0; i < 3; ++i)
        {
            f_inverse[i] =
                distribution_values[A::at(core::access::get_neighbor(current_border_node, missing[i], horizontal_nodes),
                                          8 - missing[i],
                                          horizontal_nodes * vertical_nodes)];
        }

        real_type x_velocity = -outlet_density + (f_1 + f_4 + f_7 + 2 * (f_inverse[0] + f_inverse[1] + f_inverse[2]));

        real_type y_velocity =
            y_velocities[core::access::get_result_index(horizontal_nodes_original - 3, y, horizontal_nodes_original)];

        distribution_values[A::at(core::access::get_neighbor(current_border_node, 8 - 0, horizontal_nodes),
                                  0,
                                  horizontal_nodes * vertical_nodes)] =
            f_inverse[0] - 0.5 * (f_1 - f_7) - 1.0 / 6 * x_velocity - 0.5 * y_velocity;

        distribution_values[A::at(core::access::get_neighbor(current_border_node, 8 - 3, horizontal_nodes),
                                  3,
                                  horizontal_nodes * vertical_nodes)] = f_inverse[1] - (2.0 / 3) * x_velocity;

        distribution_values[A::at(core::access::get_neighbor(current_border_node, 8 - 6, horizontal_nodes),
                                  6,
                                  horizontal_nodes * vertical_nodes)] =
            f_inverse[2] + 0.5 * (f_1 - f_7) - 1.0 / 6 * x_velocity + 0.5 * y_velocity;
    }
};

}  // namespace buffered

}  // namespace general

}  // namespace gpu

}  // namespace lbm

#endif
