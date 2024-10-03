#ifndef FILE_INTERACTION_HPP
#define FILE_INTERACTION_HPP

#include <vector>
#include "access.hpp"
#include "defines.hpp"

#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>

#include <boost/tokenizer.hpp>

typedef boost::tokenizer<boost::escaped_list_separator<char>> Tokenizer;

/**
 * @brief This structure contains all parameters necessary for a complete specification of any lattice Boltzmann algorithm.
 *        Notice that not all information is necessary for the individual algorithms but the specification of unnecessary 
 *        never leads to wrong behavior. Like this, specification is unified and less complicated.
 */
struct Settings
{
    /* General parameters */
    int debug_mode = 0;
    int results_to_csv = 0;
    std::string algorithm = "sequential_two_lattice";
    std::string access_pattern = "collision";
    unsigned int vertical_nodes = 26;
    unsigned int vertical_nodes_excluding_buffers = 24;
    unsigned int horizontal_nodes = 7;
    unsigned long total_node_count = 182;
    unsigned long total_nodes_excluding_buffers = 168;
    double relaxation_time = 1.4;
    unsigned int time_steps = 10;

    /* Parameters relevant for parallel algorithms */
    unsigned int subdomain_height = 8; // must be at least 2 for correct behavior, but why would you choose it so small?
    unsigned int subdomain_count = 3;
    unsigned int buffer_count = 2;

    /* Inlet and outlet specification */
    velocity inlet_velocity{0.1,0};
    velocity outlet_velocity{0,0};
    double inlet_density = 1;
    double outlet_density = 1;

    /* Parameters relevant for shift algorithms */
    unsigned long shift_distribution_value_count = 220;
    unsigned int shift_offset = 8;
};

/**
 * @brief Writes the data stored within a vector of sim_data_tuples and stores it to a csv file 
 *        with the given filename. If there is no such file with this name, a new one will be created.
 *        Caution: Those csv files quickly become very large!
 * 
 * @param data a vector of sim_data_tuples in which the velocity and density values within a certain
 *             time step are stored
 * @param filename a string containing the filename with or without the directory to store it to
 */
void sim_data_to_csv(std::vector<sim_data_tuple> &data, const std::string &filename);

/**
 * @brief Writes the data stored within a vector of sim_data_tuples and stores it to a csv file 
 *        with the given filename. If there is no such file with this name, a new one will be created.
 *        This method is intended for use with a domain utilizing buffers.
 *        Caution: Those csv files quickly become very large!
 * 
 * @param data a vector of sim_data_tuples in which the velocity and density values within a certain
 *             time step are stored
 * @param filename a string containing the filename with or without the directory to store it to
 */
void parallel_domain_sim_data_to_csv(std::vector<sim_data_tuple> &data, const std::string &filename);

/**
 * @brief Writes a csv configuration file based on the given struct settings.
 *        The following parameters of the struct must be specified:
 *        
 *        Always:
 *        - algorithm
 *        - access_pattern
 *        - relaxation_time
 *        - vertical_nodes_excluding_buffers
 *        - horizontal_nodes
 *        - inlet_velocity
 *        - outlet_velocity
 *        - inlet_density
 *        - outlet_density
 *  
 *        In the case of a parallel algorithm:
 *        - subdomain_count
 *        - Careful: The number of vertical nodes excluding buffers must be dividable by subdomain_count!
 * 
 *        False by default but may be activated:
 *        - debug_mode
 *        - results_to_csv
 * 
 * @param settings a struct specifying the essential parameters of the algorithm.
 */
void write_csv_config_file(const Settings &settings);

/**
 * @brief Returns a Settings struct that is set up according to the specified csv file.
 * 
 * @param filename a string containing the name of the csv file
 * @return Settings a struct containing the specified values, missing values will be initialized by default
 */
Settings retrieve_settings_from_csv(const std::string &filename);

/**
 * @brief Determines whether the specified string resembles a valid algorithm.
 * 
 * @param algorithm a string representing an algorithm
 * @return true if the specified string resembles a valid algorithm, and false if it does not
 */
inline bool is_valid_algorithm(const std::string &algorithm)
{
    return 
    // Sequential algorithms
    algorithm == "sequential_two_lattice" | 
    algorithm == "sequential_two_step" |
    algorithm == "sequential_swap" | 
    algorithm == "sequential_shift" | 
    // Parallel algorithms
    algorithm == "parallel_two_lattice" | 
    algorithm == "parallel_two_lattice_framework" | 
    algorithm == "parallel_two_step" |
    algorithm == "parallel_swap" | 
    algorithm == "parallel_shift";
}

#endif