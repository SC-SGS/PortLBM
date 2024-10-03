#include "../include/parallel_framework.hpp"

#include <hpx/algorithm.hpp>

/**
 * @brief This function is used to determine the fluid nodes belonging to a certain subdomain.
 * 
 * @param subdomain the index of the subdomain (counting starts at the bottom)
 * @param fluid_nodes a vector containing all fluid nodes within the simulation domain
 * @return see documentation of start_end_it_tuple
 */
start_end_it_tuple parallel_framework::get_subdomain_fluid_node_pointers
(
    const unsigned int &subdomain,
    const std::vector<unsigned int> &fluid_nodes
)
{
    unsigned int min_node_in_question = (SUBDOMAIN_HEIGHT + 1) * HORIZONTAL_NODES * subdomain;
    unsigned int max_node_in_question = min_node_in_question + SUBDOMAIN_HEIGHT * HORIZONTAL_NODES - 1;

    std::vector<unsigned int>::const_iterator first = fluid_nodes.begin();
    std::vector<unsigned int>::const_iterator end = fluid_nodes.end() - 1;

    unsigned int current_value = *first;

    while (current_value < min_node_in_question)
    {
        ++first;
        current_value = *first;
    }
    
    current_value = *end;

    while (current_value > max_node_in_question)
    {
        --end;
        current_value = *end;
    }

    return std::make_tuple(first, end);
}

/**
 * @brief Sets up a suitable domain for parallel computation. The domain is a rectangle with
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
 */
void parallel_framework::setup_parallel_domain
(    
    std::vector<double> &distribution_values,
    std::vector<unsigned int> &nodes,
    std::vector<unsigned int> &fluid_nodes,
    std::vector<bool> &phase_information,
    access_function access_function
)
{
    distribution_values.assign(TOTAL_NODE_COUNT * DIRECTION_COUNT, 0); 
    std::vector<double> values = maxwell_boltzmann_distribution(VELOCITY_VECTORS.at(4), 1);

    for(auto i = 0; i < TOTAL_NODE_COUNT; ++i)
    {
        // Set up vector of all nodes for direct access
        nodes.push_back(i);

        // Set distribution values to equilibrium state
        lbm_access::set_distribution_values_of(values, distribution_values, i, access_function);
    }
    boundary_conditions::initialize_inout(distribution_values, access_function);

    /* Phase information vector */
    phase_information.assign(TOTAL_NODE_COUNT, false);
    for(auto x = 0; x < HORIZONTAL_NODES; ++x)
    {
        phase_information[lbm_access::get_node_index(x,0)] = true;
        phase_information[lbm_access::get_node_index(x,VERTICAL_NODES - 1)] = true;
    }

    /* Fluid nodes vector */
    for(auto y = 1; y < VERTICAL_NODES - 1; ++y)
    {
        for(auto x = 1; x < HORIZONTAL_NODES - 1; ++x)
        {
            if(!phase_information[lbm_access::get_node_index(x,y)])
                fluid_nodes.push_back(lbm_access::get_node_index(x,y));
        }
    }

}

/**
 * @brief Retrieves a version of the border swap information data structure that is suitable for the parallel framework.
 * 
 * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain
 * @param phase_information a vector containing the phase information for every vector (true means solid)
 * @return border_swap_information see documentation of border_swap_information
 */
border_swap_information parallel_framework::retrieve_border_swap_info
(
    const std::vector<start_end_it_tuple> &fluid_node_bounds,
    const std::vector<unsigned int> &fluid_nodes,  
    const std::vector<bool> &phase_information
)
{
    std::vector<unsigned int> current_adjacencies;
    border_swap_information result;
    std::vector<unsigned int>::const_iterator start;
    std::vector<unsigned int>::const_iterator end;

    for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
    {
        start = std::get<0>(fluid_node_bounds[subdomain]);
        end = std::get<1>(fluid_node_bounds[subdomain]);
        
        for(auto it = start; it <= end; ++it)
        {
            current_adjacencies = {*it};
            for(const auto direction : STREAMING_DIRECTIONS)
            {
                unsigned int current_neighbor = lbm_access::get_neighbor(*it, direction);
                if(is_non_inout_ghost_node(current_neighbor, phase_information))
                {
                    current_adjacencies.push_back(direction);
                }
            }
            if(current_adjacencies.size() > 1) result.push_back(current_adjacencies);
        }
    }
    return result;
}

/**
 * @brief Retrieves a version of the border swap information data structure that is suitable for the parallel framework.
 * 
 * @param fluid_nodes a vector containing the indices of all fluid nodes within the simulation domain
 * @param phase_information a vector containing the phase information for every vector (true means solid)
     * @return a vector of border_swap_information, one for each subdomain
 */
std::vector<border_swap_information> parallel_framework::subdomain_wise_border_swap_info
(
    const std::vector<start_end_it_tuple> &fluid_node_bounds,
    const std::vector<unsigned int> &fluid_nodes,  
    const std::vector<bool> &phase_information
)
{
    std::vector<unsigned int> current_adjacencies;
    border_swap_information current_bsi;
    std::vector<unsigned int>::const_iterator start;
    std::vector<unsigned int>::const_iterator end;
    std::vector<border_swap_information> result;

    for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
    {
        start = std::get<0>(fluid_node_bounds[subdomain]);
        end = std::get<1>(fluid_node_bounds[subdomain]);
        current_bsi = {};
        for(auto it = start; it <= end; ++it)
        {
            current_adjacencies = {*it};
            for(const auto direction : STREAMING_DIRECTIONS)
            {
                unsigned int current_neighbor = lbm_access::get_neighbor(*it, direction);
                if(is_non_inout_ghost_node(current_neighbor, phase_information))
                {
                    current_adjacencies.push_back(direction);
                }
            }
            if(current_adjacencies.size() > 1) current_bsi.push_back(current_adjacencies);
        }
        result.push_back(current_bsi);
    }
    return result;
}

/**
 * @brief Returns a tuple specifying the inclusive range boundaries for the specified buffer index.
 * 
 * @return a tuple, 0th entry: start node of buffer, 1st entry: end note of buffer
 */
std::tuple<unsigned int, unsigned int> parallel_framework::get_buffer_node_range
(
    const unsigned int &buffer_index
)
{
    unsigned int start = SUBDOMAIN_HEIGHT * HORIZONTAL_NODES + buffer_index * (SUBDOMAIN_HEIGHT + 1) * HORIZONTAL_NODES;
    return std::make_tuple(start,start + HORIZONTAL_NODES - 1); 
}

/**
 * @brief Performs the pre-iteration buffer initialization for the buffer with the specified boundaries.
 *        For every buffer node, the directions pointing up will be copied from the nodes below and the
 *        directions pointing down will be copied from the nodes above.
 * 
 * @param buffer_bounds a tuple containing the first and last index of the buffer
 * @param distribution_values a vector containing all distribution values
 * @param access_function this function will be used to access the distribution values
 */
void parallel_framework::copy_to_buffer
(
    const std::tuple<unsigned int, unsigned int> &buffer_bounds,
    std::vector<double> &distribution_values,
    access_function access_function
)
{
    unsigned int start = std::get<0>(buffer_bounds);
    unsigned int end = std::get<1>(buffer_bounds);

    for(auto buffer_node = start; buffer_node <= end; ++buffer_node)
    {
        copy_to_buffer_node(buffer_node, distribution_values, access_function);
    }
}

/**
 * @brief For the buffer node with the specified index, the directions pointing up will be copied from 
 *        the nodes below and the directions pointing down will be copied from the nodes above.
 * 
 * @param buffer_node the index of the buffer node that is to be updated
 * @param distribution_values a vector containing all distribution values
 * @param access_function this function will be used to access the distribution values
 */
void parallel_framework::copy_to_buffer_node
(   
    unsigned int buffer_node, 
    std::vector<double> &distribution_values,
    access_function access_function
)
{
        for(auto direction : {6,7,8})
        {
            distribution_values[access_function(buffer_node, direction)] = distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 1), direction)];
        }
        for(auto direction : {0,1,2})
        {
            distribution_values[access_function(buffer_node, direction)] = distribution_values[access_function(lbm_access::get_neighbor(buffer_node, 7), direction)];
        }
}

/**
 * @brief Performs the pre-iteration buffer initialization for the buffer with the specified boundaries.
 *        For every buffer node, the directions pointing up will be copied from the nodes below and the
 *        directions pointing down will be copied from the nodes above.
 * 
 * @param buffer_bounds a tuple containing the first and last index of the buffer
 * @param distribution_values a vector containing all distribution values
 * @param access_function this function will be used to access the distribution values
 */
void parallel_framework::copy_from_buffer
(
    const std::tuple<unsigned int, unsigned int> &buffer_bounds,
    std::vector<double> &distribution_values,
    access_function access_function
)
{
    unsigned int start = std::get<0>(buffer_bounds);
    unsigned int end = std::get<1>(buffer_bounds);
    std::vector<double> current(DIRECTION_COUNT, 0);
    unsigned int current_neighbor = 0;

    for(auto buffer_node = start; buffer_node <= end; ++buffer_node)
    {
        current_neighbor = lbm_access::get_neighbor(buffer_node, 7);
        for(auto direction : {6,7,8})
        {
            distribution_values[access_function(current_neighbor, direction)] = distribution_values[access_function(buffer_node, direction)];
        }

        current_neighbor = lbm_access::get_neighbor(buffer_node, 1);
        for(auto direction : {0,1,2})
        {
            distribution_values[access_function(current_neighbor, direction)] = distribution_values[access_function(buffer_node, direction)];
        }
    }
}

/**
 * @brief Updates the ghost nodes that represent inlet and outlet edges.
 *        When updating, a velocity border condition will be considered for the input
 *        and a density border condition for the output.
 *        The inlet velocity is constant throughout all inlet nodes whereas the outlet nodes
 *        all have the specified density.
 *        The corresponding values are constants defined in "../include/"defines.hpp".
 * 
 * @param distribution_values a vector containing the distribution values of all nodes
 * @param velocities a vector containing the velocities of all nodes
 * @param densities a vector containing the densities of all nodes
 * @param access_function the access function used to access the distribution values
 */
void parallel_framework::update_velocity_input_density_output
(
    const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
    std::vector<double> &distribution_values,
    std::vector<velocity> &velocities,
    std::vector<double> &densities, 
    const access_function access_function
)
{
    hpx::for_each(
        hpx::execution::par, std::get<0>(y_values).begin(), std::get<0>(y_values).end(),
        [&distribution_values, &velocities, &densities, access_function](int y)
        {
        // Update inlets
        int current_border_node = lbm_access::get_node_index(0,y);
        velocity v = INLET_VELOCITY;
        double density = INLET_DENSITY;
        std::vector<double> current_dist_vals = maxwell_boltzmann_distribution(v, density);
        lbm_access::set_distribution_values_of
        (
            current_dist_vals,
            distribution_values,
            current_border_node,
            access_function
        );
        velocities[current_border_node] = v;
        densities[current_border_node] = density;

        // Update outlets
        current_border_node = lbm_access::get_node_index(HORIZONTAL_NODES - 1,y);
        v = macroscopic::flow_velocity(lbm_access::get_distribution_values_of(distribution_values, lbm_access::get_neighbor(current_border_node, 3), access_function));
        density = OUTLET_DENSITY;
        current_dist_vals = maxwell_boltzmann_distribution(v, density);
        lbm_access::set_distribution_values_of
        (
            current_dist_vals,
            distribution_values,
            current_border_node,
            access_function
        );
        velocities[current_border_node] = v;
        densities[current_border_node] = density;
        });
}

/**
 * @brief Initializes the specified arguments to match the dimensions of the buffers.
 * 
 * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
 * @param y_values a tuple containing the y values of all regular layers (0) and all buffer layers (1)
 */
void parallel_framework::buffer_dimension_initializations
(
    std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges,
    std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values
)
{
    for (auto buffer_index = 0; buffer_index < BUFFER_COUNT; ++buffer_index)
    {
        buffer_ranges.push_back(parallel_framework::get_buffer_node_range(buffer_index));
    }
    std::vector<unsigned int> all_ghost_y_vals;
    std::vector<unsigned int> buffer_y_vals;
    unsigned int horizontal_counter = 0;
    for(auto y = 0; y < VERTICAL_NODES; ++y)
    {
        if(horizontal_counter < SUBDOMAIN_HEIGHT)
        {
            all_ghost_y_vals.push_back(y);
            horizontal_counter++;
        }
        else
        {
            buffer_y_vals.push_back(y);
            horizontal_counter = 0;
        }
    }
    std::vector<unsigned int> ghost_y_vals(all_ghost_y_vals.begin()+1, all_ghost_y_vals.end()-1);
    y_values = std::make_tuple(ghost_y_vals, buffer_y_vals);
}

/**
 * @brief Performs an outstream step for all border nodes in the directions where they border non-inout ghost nodes.
 *        The distribution values will be stored in the ghost nodes in inverted order such that
 *        after this method is executed, the border nodes can be treated like regular nodes when performing an instream.
 * 
 * @param bsi a border_swap_information generated by retrieve_border_swap_info
 * @param distribution_values a vector containing the distribution values of all nodes
 * @param access_function the access function used to access the distribution values
 */
void parallel_framework::emplace_bounce_back_values
(
    const border_swap_information &bsi,
    std::vector<double> &distribution_values,
    const access_function access_function
)
{
    hpx::for_each
    (
        hpx::execution::par, 
        bsi.begin(), 
        bsi.end(), 
        [&](const std::vector<unsigned int>& fluid_node)
        {
            for(auto direction_iterator = fluid_node.begin()+1; direction_iterator < fluid_node.end(); ++direction_iterator) 
            {
                distribution_values[
                    access_function(lbm_access::get_neighbor(fluid_node[0], *direction_iterator), invert_direction(*direction_iterator))] = 
                    distribution_values[access_function(fluid_node[0], *direction_iterator)];
            }
        }
    );
}

/**
 * @brief Performs the buffer update that is necessary to keep inlets and outlets up to date in the case of 
 *        parallel outstream algorithms.
 * 
 * @param distribution_values a vector containing all distribution values
 * @param y_values a tuple containing the y values of all regular layers (0) and all buffer layers (1)
 * @param access_function the access to node values will be performed according to this access function
 */
void parallel_framework::outstream_buffer_update
(
    std::vector<double> &distribution_values,    
    const std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> &y_values,
    const access_function access_function
)
{
    hpx::for_each(
    hpx::execution::par, std::get<1>(y_values).begin(), std::get<1>(y_values).end(),
    [&](int y)
    {
        int current_border_node = lbm_access::get_node_index(0,y);
        parallel_framework::copy_to_buffer_node(current_border_node, distribution_values, access_function);

        current_border_node = lbm_access::get_node_index(HORIZONTAL_NODES - 1,y);
        parallel_framework::copy_to_buffer_node(current_border_node, distribution_values, access_function);
    });
}
