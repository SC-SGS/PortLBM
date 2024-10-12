/**
 * @file domain_initialization.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef DOMAIN_INITIALIZATION_HPP
#define DOMAIN_INITIALIZATION_HPP

#include "simulation.hpp"

namespace lbm
{
    /**
     * @brief Create an example domain for testing purposes. The domain is a rectangle with
     *        dimensions specified in the defines file where the outermost nodes are ghost nodes.
     *        The upper and lower ghost nodes are solid whereas the leftmost and rightmost columns are fluid
     *        nodes that mark the inlet and outlet respectively.
     *        Notice that all data will be written to the parameters which are assumed to be empty initially.
     * 
     * @param distribution_values a vector containing all distribution values.
     * @param nodes a vector containing all node indices, including those of solid nodes and ghost nodes.
     * @param fluid_nodes a vector containing the indices of all fluid nodes.
     * @param phase_information a vector containing the phase information of all nodes where true means solid.
     * @param access_function the domain will be prepared for access with this access function.
     * @param enable_debug true if debug mode is to be enabled, and false otherwise
     */

    /**
     * @brief Sets the up pipe flow environment object with a fluid in an equilibrium non-moving state.
     *        The domain consists of solid nodes on the upper and lower boundary and fluid nodes otherwise.
     * 
     * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
     * 
     * @param[in]       properties      a struct containing the densities and velocities of the inlets and outlets
     * @param[in, out]  simulation_data a structure of data on which the simulation operates
     */
    template<class T> void setup_pipe_flow_environment
    (
        const lbm::Properties &properties,
        lbm::SimulationData<T> &simulation_data
    )
    {
        static_assert(
            std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
            "Template class must have base class lbm::access::LBMAccessorObject.");

        std::vector<double> values = maxwell_boltzmann_distribution(0, 0, 1);
        for(auto i = 0; i < simulation_data.end_node_index_buffered; ++i)
        {
            lbm::access::set_distribution_values_of(values, i, *simulation_data.lbm_accessor, *simulation_data.distribution_values_0);
        }    

        lbm::boundary_conditions::update_velocity_input_density_output(properties, *simulation_data.lbm_accessor, *simulation_data.distribution_values_0);

        /* Phase information vector */
        for(auto x = 0; x < properties.horizontal_nodes; ++x)
        {
            (*simulation_data.phase_information)[lbm::access::get_node_index(x,0, properties.horizontal_nodes)] = true;
            (*simulation_data.phase_information)[lbm::access::get_node_index(x,properties.vertical_nodes - 1, properties.horizontal_nodes)] = true;
        }
    }
}

#endif // ! DOMAIN_INITIALIZATION_HPP