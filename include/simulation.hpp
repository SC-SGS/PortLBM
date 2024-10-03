/**
 * @file        simulation.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       Updated version of the lattice Boltzmann simulation.hpp first introduced in my SimTech project work:
 *              https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann.
 *              Initially, this file only contained operations to set up an example domain.
 *              Additional functionality was added to support the updated structure that suits the GPU implementation better. 
 * 
 * @version     2.0
 * 
 * @date        2024-09-27
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "access.hpp"
#include "boundaries.hpp"
#include "utils.hpp"
#include <memory>

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
void setup_example_domain
(
    std::vector<double> &distribution_values,
    std::vector<unsigned int> &nodes,
    std::vector<unsigned int> &fluid_nodes,
    std::vector<bool> &phase_information,
    access_function access_function,
    const bool enable_debug
);

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
template<class T> void setup_example_domain
(
    std::vector<double> &distribution_values,
    std::vector<unsigned int> &nodes,
    std::vector<unsigned int> &fluid_nodes,
    std::vector<bool> &phase_information,
    T &lbm_accessor,
    const bool enable_debug
)
{
    distribution_values.assign(TOTAL_NODE_COUNT * DIRECTION_COUNT, 0);


    std::vector<double> values = maxwell_boltzmann_distribution(VELOCITY_VECTORS.at(4), 1);
    for(auto i = 0; i < TOTAL_NODE_COUNT; ++i)
    {
        lbm_access::set_distribution_values_of(values, distribution_values, i, lbm_accessor);
    }    

    boundary_conditions::initialize_inout(distribution_values, lbm_accessor);

    if(enable_debug)
    {
        std::cout << "All distribution values were set, setting up the other required data..." << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Vertical node count is " << VERTICAL_NODES << "\n";
    std::cout << "Horizontal node count is " << HORIZONTAL_NODES << "\n";
    std::cout << "Total node count is " << TOTAL_NODE_COUNT << "\n";
    /* Set all nodes for direct access */
    for(auto i = 0; i < TOTAL_NODE_COUNT; ++i)
    {
        nodes.push_back(i);
    }

    std::cout << "Maximum node index is " << *(nodes.end()-1) << "\n";

    /* Set up vector containing fluid nodes within the simulation domain. */
    for(auto it = nodes.begin() + HORIZONTAL_NODES; it < nodes.end() - HORIZONTAL_NODES; ++it)
    {
        std::cout << "Processing node " << *it << "\n";
        std::cout << "Condition " << *it << " % " << HORIZONTAL_NODES << " != 0 evaluates to " << ((*it % HORIZONTAL_NODES) != 0);
        std::cout << "Condition " << *it << " % " << HORIZONTAL_NODES << " != HORIZONTAL_NODES - 1 evaluates to " << ((*it % HORIZONTAL_NODES) != (HORIZONTAL_NODES - 1)) << "\n";
        std::cout << "Is node " << *it << " fluid? " << (((*it % HORIZONTAL_NODES) != 0) && ((*it % HORIZONTAL_NODES) != (HORIZONTAL_NODES - 1))) << "\n";
        if(((*it % HORIZONTAL_NODES) != 0) && ((*it % HORIZONTAL_NODES) != (HORIZONTAL_NODES - 1))) fluid_nodes.push_back(*it);
    }
    
    /* Phase information vector */
    phase_information.assign(TOTAL_NODE_COUNT, false);
    for(auto x = 0; x < HORIZONTAL_NODES; ++x)
    {
        phase_information[lbm_access::get_node_index(x,0)] = true;
        phase_information[lbm_access::get_node_index(x,VERTICAL_NODES - 1)] = true;
    }
}

/**
 * @brief This structure contains all important properties of the simulation.
 *        It is a replacement of the Settings structure used in the project work.
 *        It was replaced owing to the introduced support of a grid decomposition
 *        and a structure-of-array representation of the simulation results.
 */
struct Properties
{
    /* Miscellanea */
    bool debug_mode;
    bool results_to_csv;
    double relaxation_time;
    unsigned int time_steps;

    /* Extents of the simulation domain */
    unsigned int vertical_nodes;
    unsigned int horizontal_nodes;
    
    // Total amount of nodes including ghost nodes but excluding buffer nodes
    unsigned int non_buffered_node_count;

    // Total amount of nodes including ghost nodes and buffer nodes
    unsigned int buffered_node_count;

    /* Extents and amount of subdomains */
    
    // Height of each subdomain
    unsigned int subdomain_height;

    // Width of each subdomain
    unsigned int subdomain_width;

    // Vertical amount of subdomains
    unsigned int subdomain_count_vertical;

    // Horizontal amount of subdomains
    unsigned int subdomain_count_horizontal;

    /* Inlet and outlet nodes*/
    double inlet_velocity_x;
    double inlet_velocity_y;
    double outlet_velocity_x;
    double outlet_velocity_y;

    /* Algorithmic options */
    std::string data_layout;
    std::string algorithm;

    /**
     * @brief Constructs a new properties object with the specified parameters.
     */
    explicit Properties
    (
        const bool debug_mode,
        const bool results_to_csv,
        const double relaxation_time,
        const unsigned int time_steps,
        const unsigned int vertical_nodes,
        const unsigned int horizontal_nodes,
        const unsigned int non_buffered_node_count,
        const unsigned int buffered_node_count,
        const unsigned int subdomain_height,
        const unsigned int subdomain_width,
        const unsigned int subdomain_count_vertical,
        const unsigned int subdomain_count_horizontal,
        const double inlet_velocity_x,
        const double inlet_velocity_y,
        const double outlet_velocity_x,
        const double outlet_velocity_y,
        const std::string &&data_layout,
        const std::string &&algorithm
    ); 
};

/**
 * @brief This structure contains the results of the simulation in a structure-of-arrays representation.
 *        It is a replacement of the sim_data_tuple used in the project work. 
 *        
 *        Notice that this structure stores shared pointers to vectors and not the vectors themselves.
 *        It is not recommended to resize the vectors as may require reallocation and may thus invalidate 
 *        the pointer. This can lead to poor predictability. If you want to resize the vectors, set the 
 *        pointers to new vectors of the desired size or construct a new SimulationResults object.
 *        
 *        Results are stored time-step-wise, that is, the values are stored according to the order of
 *        the node index for one time step and then for another. Hence, to properly access the results,
 *        the accessor must know the total amount of nodes including buffers.
 */
struct SimulationResults
{
    /**
     * @brief A shared pointer to a vector containing the densities of all nodes in the simulation domain.
     *        Solid nodes should always have the value '-1.0' for better distinction from the fluid nodes.
     *        Notice that in the case of an incompressible fluid, the density values still vary since these
     *        "virtual" densities are required by the simulation. Hence, in this case, these density values
     *        are not meaningful. However, they are meaningful for compressible fluids.
     */
    std::shared_ptr<std::vector<double>> densities;

    /**
     * @brief A shared pointer to a vector containing the pressure values of all nodes in the simulation domain.
     *        Solid nodes should always have the value '-1.0' for better distinction from the fluid nodes.
     *        Notice that unlike the density values, the pressure values are meaningful for both compressible
     *        and incompressible fluids.
     */
    std::shared_ptr<std::vector<double>> pressures;

    /**
     * @brief A shared pointer to a vector containing the x components of the velocity vectors of all nodes 
     *        in the simulation domain. Solid nodes should always have a zero component and are not differenciated
     *        further regarding their velocities. All differenciation between solid and fluid nodes is realized
     *        through the density values.
     */
    std::shared_ptr<std::vector<double>> x_velocities;

    /**
     * @brief A shared pointer to a vector containing the y components of the velocity vectors of all nodes 
     *        in the simulation domain. Solid nodes should always have a zero component and are not differenciated
     *        further regarding their velocities. All differenciation between solid and fluid nodes is realized
     *        through the density values.
     */
    std::shared_ptr<std::vector<double>> y_velocities;

    /**
     * @brief Constructs a new simulation results structure based on the provided properties structure.
     *        The internal vectors are initialized with the correct size and filled up with values such as all nodes
     *        were solid.
     * 
     * @param properties this structure of properties defines the total buffered node count and the number of time steps.
     */
    explicit SimulationResults(const Properties &properties);
};

/**
 * @brief This structure contains all data on which the simulation operates internally.
 * 
 * @tparam T any child class of lbm_access::LBMAccessorObject
 */
template <class T> struct SimulationData
{
    //typename std::enable_if<std::is_base_of<lbm_access::LBMAccessorObject, T>::value>::type* = nullptr

    std::shared_ptr<std::vector<bool>> phase_information;
    std::shared_ptr<std::vector<bool>> is_buffer;
    std::shared_ptr<std::vector<double>> distribution_values_0;
    std::shared_ptr<std::vector<double>> distribution_values_1;
    std::shared_ptr<std::vector<unsigned int>> boundary_interactions;

    unsigned int end_node_index_non_buffered;
    unsigned int end_node_index_buffered;

    std::shared_ptr<T> lbm_accessor;

    /**
     * @brief Constructs a new SimulationData object with an accessor object of the specified type.
     * 
     * @param properties the extents of the lattice specified in this Properties structure are used for initialization
     */
    explicit SimulationData<T>
    (
        const Properties &properties
    )
    :
    phase_information(std::make_shared<std::vector<bool>>(properties.buffered_node_count, false)),
    is_buffer(std::make_shared<std::vector<bool>>(properties.buffered_node_count, false)),
    distribution_values_0(std::make_shared<std::vector<double>>(properties.buffered_node_count, 0.0f)),
    distribution_values_1(std::make_shared<std::vector<double>>(properties.buffered_node_count, 0.0f)),
    boundary_interactions(std::make_shared<std::vector<unsigned int>>(properties.buffered_node_count * DIRECTION_COUNT, 0)),
    end_node_index_non_buffered(properties.non_buffered_node_count),
    end_node_index_buffered(properties.buffered_node_count),
    lbm_accessor(std::make_shared<T>(properties.horizontal_nodes, properties.buffered_node_count))
    {};
};

/**
 * @brief Specialized version of the SimulationData structure for use with a collision accessor object.
 *        The only difference is that the collision accessor object requires one less parameter for its constructor.
 */
template<> struct SimulationData<lbm_access::LBMCollisionAccessor>
{
    std::shared_ptr<std::vector<bool>> phase_information;
    std::shared_ptr<std::vector<bool>> is_buffer;
    std::shared_ptr<std::vector<double>> distribution_values_0;
    std::shared_ptr<std::vector<double>> distribution_values_1;
    std::shared_ptr<std::vector<unsigned int>> boundary_interactions;

    unsigned int end_node_index_non_buffered;
    unsigned int end_node_index_buffered;

    std::shared_ptr<lbm_access::LBMCollisionAccessor> lbm_accessor;

    /**
     * @brief Constructs a new SimulationData object with an accessor object of the specified type.
     * 
     * @param properties the extents of the lattice specified in this Properties structure are used for initialization
     */
    explicit SimulationData<lbm_access::LBMCollisionAccessor>
    (
        const Properties &properties
    )
    :
    phase_information(std::make_shared<std::vector<bool>>(properties.buffered_node_count, false)),
    is_buffer(std::make_shared<std::vector<bool>>(properties.buffered_node_count, false)),
    distribution_values_0(std::make_shared<std::vector<double>>(properties.buffered_node_count, 0.0f)),
    distribution_values_1(std::make_shared<std::vector<double>>(properties.buffered_node_count, 0.0f)),
    boundary_interactions(std::make_shared<std::vector<unsigned int>>(properties.buffered_node_count * DIRECTION_COUNT, 0)),
    end_node_index_non_buffered(properties.non_buffered_node_count),
    end_node_index_buffered(properties.buffered_node_count),
    lbm_accessor(std::make_shared<lbm_access::LBMCollisionAccessor>(properties.horizontal_nodes))
    {};
};

/**
 * @brief This structure contains a border_swap_information and an access function as used in the project work.
 *        Since this functionality is not compatible with the GPU implementation, it is considered "legacy" but kept for
 *        easier comparability. However, it is deprecated and might be removed in the future.
 * 
 */
struct LegacyData
{
    std::shared_ptr<border_swap_information> bsi;
    access_function acc_f;

    explicit LegacyData(const border_swap_information &bsi, const access_function &acc_f);
};

#endif
