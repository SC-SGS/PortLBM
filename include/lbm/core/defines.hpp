#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <vector>
#include <complex>
#include <map>
#include <functional>

namespace lbm
{
    /**
     * @brief Representation of a velocity vector
     */
    //typedef std::array<double, 2> velocity; 

    /**
     * @deprecated Legacy typedef. Will be removed in the future.
     * 
     * @brief Convenience type definition that represents a vector from which the boundary treatment 
     *        of all nodes can be retrieved. Information is stored in the following way:
     *        Each entry of the outer vector represents a border node, called BORDER_NODE for explanation.
     *        Every BORDER_NODE is a vector with the following information:
     * 
     *        - 0th entry: The index of BORDER_NODE
     * 
     *        - Further entries: The directions pointing to non-inout ghost nodes (including solid nodes within the domain).
     * 
     *        This is used for the halfway bounce-back boundary treatment where BORDER_NODE will copy the 
     *        respective distribution values BEFORE streaming.
     */
    //typedef std::vector<std::vector<unsigned int>> border_swap_information;

    /**
     * @deprecated Legacy typedef. Will be removed in the future.
     * 
     * @brief Convenience type definition that describes a tuple containing vectors of all flow velocities 
     *        and density values for a fixed time step.
     */
    //typedef std::tuple<std::vector<velocity>, std::vector<double>> sim_data_tuple;

    /**
     * @deprecated Legacy typedef. Will be removed in the future.
     * @brief This type stands for an access function. Node values can be stored in different layout and 
     *        via this function, the corresponding access scheme can be specified.
     */
    //typedef std::function<unsigned int(unsigned int, unsigned int)> access_function;
}

#endif