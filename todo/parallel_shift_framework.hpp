#ifndef PARALLEL_SHIFT_FRAMEWORK_HPP
#define PARALLEL_SHIFT_FRAMEWORK_HPP

#include <vector>

#include "collision.hpp"
#include "boundaries.hpp"
#include "defines.hpp"
#include "file_interaction.hpp"
#include "macroscopic.hpp"
#include "parallel_framework.hpp"
#include "sequential_shift.hpp"

/**
 * @brief This namespace contains all methods required for the parallel shift algorithm.
 *        Notice that the shift algorithm forms a rather independent framework that differs 
 *        from the framework of the other algorithms.
 */
namespace parallel_shift_framework
{

    /**
     * @brief Performs the parallel shift algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
     * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
     *                            see documentation of border_swap_information
     * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
     * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
     *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param iterations          this many iterations will be performed
     */
    void run
    (  
        const std::vector<start_end_it_tuple> &fluid_nodes,       
        const std::vector<border_swap_information> &boundary_nodes,
        std::vector<double> &distribution_values,   
        const access_function access_function,
        const unsigned int iterations
    );

    /**
     * @brief Performs the parallel shift algorithm for the specified number of iterations.
     * 
     * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
     * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
     *                            see documentation of border_swap_information
     * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
     * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
     *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param iterations          this many iterations will be performed
     */
    void run_debug
    (  
        const std::vector<start_end_it_tuple> &fluid_nodes,       
        const std::vector<border_swap_information> &boundary_nodes,
        std::vector<double> &distribution_values,   
        const access_function access_function,
        const unsigned int iterations
    );

    /**
     * @brief Performs a combined collision and streaming step for the specified fluid node.
     * 
     * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
     * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
     *                            see documentation of border_swap_information
     * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
     * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
     *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param buffer_ranges       a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
     * @param iteration           the iteration the algorithm is currently processing
     */
    sim_data_tuple stream_and_collide
    (
        const std::vector<start_end_it_tuple> &fluid_nodes,
        const std::vector<border_swap_information> &bsi,
        std::vector<double> &distribution_values, 
        const access_function access_function,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges,
        const unsigned int iteration
    );

    /**
     * @brief Performs a combined collision and streaming step for the specified fluid node.
     *        This algorithm prints out various debug comments.
     * 
     * @param fluid_nodes         a vector of tuples of iterators pointing at the first and last fluid node of each domain
     * @param boundary_nodes      a vector of border_swap_information for each subdomain, 
     *                            see documentation of border_swap_information
     * @param distribution_values a vector containing all distribution values, including those of buffer and "overlap" nodes
     * @param access_function     An access function from the namespace parallel_shift_framework::access_functions.
     *                            Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param buffer_ranges       a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
     * @param iteration           the iteration the algorithm is currently processing
     */
    sim_data_tuple stream_and_collide_debug
    (
        const std::vector<start_end_it_tuple> &fluid_nodes,
        const std::vector<border_swap_information> &bsi,
        std::vector<double> &distribution_values, 
        const access_function access_function,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges,
        const unsigned int iteration
    );

    /**
     * @brief Performs the pre-iteration buffer initialization for the buffer with the specified boundaries for an odd time step.
     *        Northbound values from the nodes below remain within the buffer whereas southbound values from the nodes above will
     *        instead be written directly into the overlapping shift area.
     * 
     * @param buffer_bounds a tuple containing the first and last index of the buffer
     * @param distribution_values a vector containing all distribution values
     * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param buffer_offset the shift-related offset of this buffer
     */
    void buffer_update_odd_time_step
    (
        const std::tuple<unsigned int, unsigned int> &buffer_bounds,
        std::vector<double> &distribution_values,
        const access_function access_function,
        const unsigned int buffer_offset
    );

    /**
     * @brief Performs the pre-iteration buffer initialization for the buffer with the specified boundaries for an even time step.
     *        Southbound values from the nodes above remain within the buffer whereas northbound values from the nodes below will
     *        instead be written directly into the overlapping shift area.
     * 
     * @param buffer_bounds a tuple containing the first and last index of the buffer
     * @param distribution_values a vector containing all distribution values
     * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param buffer_offset the shift-related offset of this buffer
     */
    void buffer_update_even_time_step
    (
        const std::tuple<unsigned int, unsigned int> &buffer_bounds,
        std::vector<double> &distribution_values,
        const access_function access_function,
        const unsigned int buffer_offset
    );

    /**
     * @brief Performs the collision step for the fluid node with the specified index.
     * 
     * @param node the index of the fluid node
     * @param distribution_values the updated distribution values will be written to this vector
     * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param velocities a vector containing the velocities of all nodes
     * @param densities a vector containing the densities of all nodes
     * @param write_offset the write offset of the current iteration
     */
    inline void perform_collision
    (
        const unsigned int node,
        std::vector<double> &distribution_values, 
        const access_function &access_function, 
        std::vector<velocity> &velocities, 
        std::vector<double> &densities,
        const unsigned int write_offset
    )
    {
        std::vector<double> current_distributions = lbm_access::get_distribution_values_of(distribution_values, node + write_offset, access_function);
        velocities[node] = macroscopic::flow_velocity(current_distributions);
        densities[node] = macroscopic::density(current_distributions);
        current_distributions = collision::collide_bgk(current_distributions, velocities[node], densities[node]);
        lbm_access::set_distribution_values_of(current_distributions, distribution_values, node + write_offset, access_function);
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
     * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     */
    void setup_parallel_domain
    (    
        std::vector<double> &distribution_values,
        std::vector<unsigned int> &nodes,
        std::vector<unsigned int> &fluid_nodes,
        std::vector<bool> &phase_information,
        access_function access_function
    );

    /**
     * @brief Updates the ghost nodes that represent inlet and outlet edges.
     *        When updating, a velocity border condition will be considered for the input
     *        and a density border condition for the output.
     *        The corresponding values are constants defined in "../include/"defines.hpp".
     * 
     * @param distribution_values the updated distribution values will be written to this vector
     * @param velocities a vector containing the velocities of all nodes
     * @param densities a vector containing the densities of all nodes
     * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param offset the write offset of the current iteration
     */
    void update_velocity_input_density_output
    (
        std::vector<double> &distribution_values,
        std::vector<velocity> &velocities,
        std::vector<double> &densities, 
        const access_function access_function,
        const unsigned int offset
    );

    /**
     * @brief Performs an outstream step for all border nodes in the directions where they border non-inout ghost nodes.
     *        The distribution values will be stored in the ghost nodes in inverted order such that
     *        after this method is executed, the border nodes can be treated like regular nodes when performing an instream.
     * 
     * @param bsi a border_swap_information generated by retrieve_border_swap_info
     * @param distribution_values a vector containing the distribution values of all nodes
     * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param read_offset offset for shift algorithm, leave zero for all other algorithms
     */
    void emplace_bounce_back_values
    (
        const border_swap_information &bsi,
        std::vector<double> &distribution_values,
        const access_function access_function,
        const unsigned int read_offset
    );

    /**
     * @brief This helper function determines the offset of the node with the specified index at an even time step.
     * 
     * @param node index of the node in question
     * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
     */
    inline unsigned int determine_even_time_offset
    (
        const unsigned int node,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
    )
    {
        unsigned int result = 0;
        auto current = buffer_ranges.begin(); 
        while(current < buffer_ranges.end() && node >= std::get<0>(*current))
        {
            result++;
            current++;
        }
        return result * (SHIFT_OFFSET);
    }

    /**
     * @brief This helper function determines the offset of the node with the specified index at an even time step.
     * 
     * @param node index of the node in question
     * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
     */
    inline unsigned int determine_odd_time_offset
    (
        const unsigned int node,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
    )
    {
        unsigned int result = 1;
        auto current = buffer_ranges.begin(); 
        while(current < buffer_ranges.end() && node > std::get<1>(*current))
        {
            result++;
            current++;
        }
        return result * (SHIFT_OFFSET);
    }

    /**
     * @brief Prints all distribution values to the console.
     *        They are displayed in the original order, i.e. the origin is located at the lower left corner of the printed distribution chart.
     *        This version is adapted for the shift algorithm such that a proper visualization for debugging purposes is possible.
     * 
     * @param distribution_values a vector containing the distribution values of all nodes 
     * @param access_function An access function from the namespace parallel_shift_framework::access_functions.
     *                        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     * @param offset the offset applied in the process that is to be visualized
     * @param buffer_ranges a vector containing a tuple of the indices of the first and last node belonging to a certain buffer
     */
    inline void print_distribution_values
    (
        const std::vector<double> &distribution_values, 
        const access_function access_function,
        const unsigned int offset,
        const std::vector<std::tuple<unsigned int, unsigned int>> &buffer_ranges
    )
    {
        std::vector<std::vector<unsigned int>> print_dirs = {{6,7,8}, {3,4,5}, {0,1,2}};
        unsigned int current_node_index = 0;
        std::vector<double> current_values(DIRECTION_COUNT,0);
        std::cout << std::setprecision(3) << std::fixed;
        unsigned int line_counter = 0;
        unsigned int natural_offset = 0;

        for(auto y = VERTICAL_NODES; y-- > 0; )
        {
            if(line_counter == SUBDOMAIN_HEIGHT)
                std::cout << "\033[32m";

            for(auto i = 0; i < 3; ++i)
            {
                auto current_row = print_dirs[i];

                for(auto x = 0; x < HORIZONTAL_NODES; ++x)
                {
                    if(x == 0 || x == 1 || x == HORIZONTAL_NODES - 1 || x == HORIZONTAL_NODES - 2) std::cout << std::setprecision(5) << std::fixed;

                    if(x == 0 && y == 0) std::cout << "\033[31m";
                    else if(x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES - 1)) std::cout << "\033[34m";

                    current_node_index = lbm_access::get_node_index(x, y);

                    if(offset == 0)
                        natural_offset = parallel_shift_framework::determine_even_time_offset(current_node_index, buffer_ranges);
                    else
                        natural_offset = parallel_shift_framework::determine_odd_time_offset(current_node_index, buffer_ranges);

                    current_node_index = current_node_index + natural_offset;

                    current_values = lbm_access::get_distribution_values_of(distribution_values, current_node_index, access_function);

                    for(auto j = 0; j < 3; ++j)
                    {
                        auto direction = current_row[j];
                        std::cout << current_values[direction] << "  ";
                    }

                    std::cout << "\t";

                    if((x == 0 && y == 0) || (x == (HORIZONTAL_NODES - 1) && y == (VERTICAL_NODES - 1))) std::cout << "\033[0m";
                    if(x == 0 || x == 1 || x == HORIZONTAL_NODES - 1 || x == HORIZONTAL_NODES - 2) std::cout << std::setprecision(3) << std::fixed;
                }

                std::cout << std::endl;

            }

            std::cout << std::endl;
            std::cout << std::endl;

            if(line_counter == SUBDOMAIN_HEIGHT) line_counter = 0;
            else line_counter++;

            std::cout << "\033[0m";
        }

        std::cout << std::setprecision(3) << std::fixed;
    }

    /**
     * @brief This namespace contains all access functions in a version compatible with the shift algorithm.
     *        Caution: This algorithm is NOT compatible with the access functions from the namespace lbm_access.
     */
    namespace access_functions
    {
        /**
         * @brief Returns the vector index of the collision layout
         * 
         * @param node the node in the simulation domain
         * @param direction the direction of the velocity vector
         * @return the index of the vector storing the distribution values 
         */
        inline unsigned int collision(unsigned int node, unsigned int direction)
        {
            return DIRECTION_COUNT * node + direction;        
        }

        /**
         * @brief Returns the vector index of the stream layout
         * 
         * @param node the node in the simulation domain
         * @param direction the direction of the velocity vector
         * @return the index of the vector storing the distribution values  
         */
        inline unsigned int stream(unsigned int node, unsigned int direction)
        {
            return SHIFT_DISTRIBUTION_VALUE_COUNT * direction + node;
        }

        /**
         * @brief Returns the vector index of the bundle layout
         * 
         * @param node the node in the simulation domain
         * @param direction the direction of the velocity vector
         * @return the index of the vector storing the distribution values  
         */
        inline unsigned int bundle(unsigned int node, unsigned int direction)
        {
            return 3 * (direction / 3) * SHIFT_DISTRIBUTION_VALUE_COUNT + (direction % 3) + 3 * node; 
        }
    }
}

#endif