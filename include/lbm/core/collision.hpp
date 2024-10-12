#ifndef COLLISION_HPP
#define COLLISION_HPP

#include "access.hpp"
#include "simulation.hpp"

namespace lbm
{
    /**
     * @brief This namespace contains all functions necessary for the collision step.
     */
    namespace collision
    {

        /**
         * @brief Performs the collision step for the specified node according to the BGK model using its density and velocity value.
         * 
         * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
         * 
         * @param[in, out]  simulation_data             a structure containing the accessor object and the distribution values
         * @param[in]       simulation_results          a structure containing the densities and velocities of all nodes
         * @param[in]       relaxation_time             the relaxation time
         * @param[in]       simulation_results_index    this index is used to access the velocity and density values
         * @param[in]       node                        the index of the node in question
         */
        template <class T> inline void collide_bgk
        (
            const lbm::SimulationData<T> &simulation_data,
            const lbm::SimulationResults &simulation_results,
            const double relaxation_time,
            const unsigned int simulation_results_index,
            const unsigned int node
        )
        {
            static_assert(
                std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                "Template class must have base class lbm::access::LBMAccessorObject.");

            std::vector<double> values = lbm::access::get_distribution_values_of(*simulation_data.distribution_values_1, node, *simulation_data.lbm_accessor); //careful: explicitly using distribution_values_1 here!
            std::vector<double> result = maxwell_boltzmann_distribution(
                (*simulation_results.x_velocities)[simulation_results_index], 
                (*simulation_results.y_velocities)[simulation_results_index],
                (*simulation_results.densities)[simulation_results_index]
            );

            for(auto i = 0; i < 9; ++i)
            {
                result[i] = -(1/relaxation_time) * (values[i] - result[i]) + values[i];
            }

            lbm::access::set_distribution_values_of(result, node, *simulation_data.lbm_accessor, *simulation_data.distribution_values_1);
        }
    } // ! namespace collision

} // ! namespace lbm

#endif // ! COLLISION_HPP