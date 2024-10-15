#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include "access.hpp"
#include "macroscopic.hpp"
#include "simulation.hpp"
#include "../console/console_output.hpp"

namespace lbm
{
    /**
     * @brief Returns whether the node in question borders the ghost nodes outside the actual domain.
     * 
     * @param[in] node_index        the index of the node in question
     * @param[in] vertical_nodes    the amount of vertical nodes including ghost nodes and buffers
     * @param[in] horizontal_nodes  the amount of horizontal nodes including ghost nodes and buffers
     * 
     * @return  whether or not a node is an edge node
     */
    inline bool is_edge_node
    (
        const unsigned int node_index, 
        const unsigned int vertical_nodes, 
        const unsigned int horizontal_nodes
    )
    {
        std::tuple<unsigned int, unsigned int> coordinates = lbm::access::get_node_coordinates(node_index, horizontal_nodes);
        unsigned int x = std::get<0>(coordinates);
        unsigned int y = std::get<1>(coordinates);
        return ((x == 1) || (x == (horizontal_nodes - 2))) && ((y == 1) || (y == (vertical_nodes - 2)));
    }

    /**
     * @brief Returns whether the node with the specified index is a ghost node.
     *        This is the case when at least one of the node coordinates is zero or the maximum extent 
     *        in the according dimension, or if the node is solid.
     * 
     * @param[in] node_index        index of the node in question
     * @param[in] properties        the properties structure containing the domain extents
     * @param[in] phase_information whether the node is solid (true) or fluid (false)
     * 
     * @return whether or not the node is a ghost node
     */
    inline bool is_ghost_node
    (
        const unsigned int node_index, 
        const Properties &properties,
        const bool phase_information
    )
    {
        if(phase_information)
        {
            return true;
        }
        else
        {
            std::tuple<unsigned int, unsigned int> coordinates = lbm::access::get_node_coordinates(node_index, properties.horizontal_nodes);
            unsigned int x = std::get<0>(coordinates);
            unsigned int y = std::get<1>(coordinates);
            return ((x == 0) || (x == (properties.horizontal_nodes - 1))) || ((y == 0) || (y == (properties.vertical_nodes - 1)));
        }
    }

    /**
     * @deprecated This method was used to determine the bounce-back directions excluding inlet and outlet nodes in the CPU implementation.
     *             Since the bounce-back semantics are slightly different for the GPU implementation (bounce-back is processed per solid node), 
     *             this method will not be needed in the future.
     * 
     * @brief This convenience method behaves like `is_ghost_node(const unsigned int, const Properties&, const bool)`
     *        with the exception that inlet and outlet nodes are not considered.
     * 
     * @param[in] node_index        index of the node in question
     * @param[in] properties        the properties structure containing the domain extents
     * @param[in] phase_information whether the node is solid (true) or fluid (false)
     * 
     * @return true if the node is a non-inlet and non-outlet ghost node, and false otherwise
     */
    inline bool is_non_inout_ghost_node
    (
        unsigned int node_index, 
        const lbm::Properties &properties,
        const bool phase_information
    )
    {
        std::tuple<unsigned int, unsigned int> coordinates = lbm::access::get_node_coordinates(node_index, properties.horizontal_nodes);
        unsigned int x = std::get<0>(coordinates);
        unsigned int y = std::get<1>(coordinates);
        return ((x != 0) && (x != (properties.horizontal_nodes - 1))) && ((y == 0) || (y == (properties.vertical_nodes - 1)) || phase_information);
    }

    /**
     * @brief This namespace contains function representations of boundary conditions used in the lattice-Boltzmann model.
     *        Notice that parallel versions exist within "parallel_framework.hpp".
     */
    namespace bounce_back
    {

        /**
         * @brief Performs an outstream step for all border nodes in the directions where they border non-inout ghost nodes.
         *        The distribution values will be stored in the ghost nodes in inverted order such that
         *        after this method is executed, the border nodes can be treated like regular nodes when performing an instream.
         * 
         * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
         * 
         * @param[in]       bsi                 the border swap information
         * @param[in, out]  distribution_values a vector containing the distribution values of all nodes
         * @param[in]       lbm_accessor        the accessor object according to which distribution values are accessed
         */
        template <class T> void emplace_bounce_back_values
        (
            const lbm::border_swap_information &bsi,
            std::vector<double> &distribution_values,
            const T &lbm_accessor
        )
        {
            static_assert(
                std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                "Template class must be child of lbm::access::LBMAccessorObject.");
            
            for(auto bsi_iterator = bsi.begin(); bsi_iterator < bsi.end(); ++bsi_iterator)
            {
                for(auto direction_iterator = (*bsi_iterator).begin()+1; direction_iterator < (*bsi_iterator).end(); ++direction_iterator) 
                {
                    distribution_values[
                        lbm_accessor.get_index(
                            lbm::access::get_neighbor((*bsi_iterator)[0], *direction_iterator, lbm_accessor.horizontal_nodes), 
                            invert_direction(*direction_iterator))] 
                                = distribution_values[lbm_accessor.get_index((*bsi_iterator)[0], *direction_iterator)];
                }
            }
        }

        /**
         * @brief Retrieves the border swap information data structure.
         *        This method does not consider inlet and outlet ghost nodes when performing bounce-back
         *        as the inserted values will be overwritten by inflow and outflow values anyways.
         * 
         * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain
         * @param phase_information a vector containing the phase information for every vector (true means solid)
         * @return lbm::border_swap_information see documentation of lbm::border_swap_information
         */

        /**
         * @brief Retrieves the border swap information data structure.
         *        This method does not consider inlet and outlet ghost nodes when performing bounce-back
         *        as the inserted values will be overwritten by inflow and outflow values anyways.
         * 
         * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
         * 
         * @param[in]       properties      the properties structure containing the domain extents   
         * @param[in, out]  simulation_data a structure containing, among others, the phase information
         * 
         * @return an lbm::border_swap_information, i.e. a vector of vectors with the index of a fluid node and bounce-back directions
         */
        template <class T> lbm::border_swap_information retrieve_border_swap_info
        (
            const lbm::Properties &properties,
            lbm::SimulationData<T> &simulation_data
        )
        {
            static_assert(
                std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                "Template class must be child of lbm::access::LBMAccessorObject.");

            std::vector<unsigned int> current_adjacencies;
            lbm::border_swap_information result;

            for(unsigned int node = 0; node < simulation_data.end_node_index_buffered; ++node)
            {
                if(!lbm::is_ghost_node(node, properties, (*simulation_data.phase_information)[node]))
                {
                    std::cout << "Not ghost node: " << node << "\n";
                    current_adjacencies = {node};
                    for(const auto direction : lbm::constants::streaming_directions)
                    {
                        unsigned int current_neighbor = lbm::access::get_neighbor(node, direction, properties.horizontal_nodes);
                        if(is_non_inout_ghost_node(current_neighbor, properties, (*simulation_data.phase_information)[node]))
                        {
                            std::cout << "\tFound non inout ghost node: " << current_neighbor << "\n";
                            current_adjacencies.push_back(direction);
                        }
                    }
                    if(current_adjacencies.size() > 1) result.push_back(current_adjacencies);
                }
            }
            for(const auto& current : result)
                lbm::console::print_vector(current, current.size());
            return result;
        }

    } // ! namespace bounce_back

    /**
     * @brief This namespace contains all functions that are required for enforcing boundary conditions.
     *        It uses the ghost nodes for this purpose.
     */
    namespace boundary_conditions
    {

        /**
         * @brief Updates the ghost nodes that represent inlet and outlet edges.
         *        When updating, a velocity border condition will be considered for the input
         *        and a density border condition for the output.
         *        The inlet velocity is constant throughout all inlet nodes whereas the outlet nodes
         *        all have the specified density.
         *        The corresponding values are lbm::constants defined in `constants.hpp`.
         * 
         * @tparam T an lbm accessor object, that is, any object whose class inherits from `lbm::access::LBMAccessorObject`
         * 
         * @param[in]       properties          the properties structure containing the inlet and outlet density and velocity 
         * @param[in]       lbm_accessor        the accessor object according to which distribution values are accessed       
         * @param[in, out]  distribution_values a vector containing the distribution values of all nodes
         */
        template <class T>
        void update_velocity_input_density_output
        (
            const lbm::Properties &properties,
            const T &lbm_accessor,
            std::vector<double> &distribution_values
        )
        {
            static_assert(
                std::is_base_of<lbm::access::LBMAccessorObject, T>::value, 
                "Template class must be child of lbm::access::LBMAccessorObject.");

            std::vector<double> current_dist_vals(9, 0);
            unsigned int current_border_node = 0;
            lbm::velocity v = {properties.inlet_velocity_x, properties.inlet_velocity_y};
            //(auto y = 1; y < properties.vertical_nodes - 1; ++y)
            for(auto y = 2; y < properties.vertical_nodes - 2; ++y)
            {
                // Update inlets
                current_border_node = lbm::access::get_node_index(1,y,properties.horizontal_nodes);
                current_dist_vals = maxwell_boltzmann_distribution(
                    properties.inlet_velocity_x, properties.inlet_velocity_y, properties.inlet_density);

                lbm::access::set_distribution_values_of
                (
                    current_dist_vals,
                    current_border_node,
                    lbm_accessor,
                    distribution_values
                );

                // Update outlets
                current_border_node = lbm::access::get_node_index(properties.horizontal_nodes - 2,y,properties.horizontal_nodes);
                v = lbm::macroscopic::flow_velocity(
                    lbm::access::get_distribution_values_of(
                        distribution_values, lbm::access::get_neighbor(current_border_node, 3, properties.horizontal_nodes), lbm_accessor));
                current_dist_vals = maxwell_boltzmann_distribution(v[0], v[1], properties.outlet_density);
                lbm::access::set_distribution_values_of
                (
                    current_dist_vals,
                    current_border_node,
                    lbm_accessor,
                    distribution_values
                );
            }
        }

    } // ! namespace boundary_conditions

} // ! namespace lbm

#endif // ! BOUNCE_BACK_HPP