
#include "../include/file_interaction.hpp"

/**
 * @brief Writes the data stored within a vector of sim_data_tuples and stores it to a csv file 
 *        with the given filename. If there is no such file with this name, a new one will be created.
 *        Caution: Those csv files quickly become very large!
 * 
 * @param data a vector of sim_data_tuples in which the velocity and density values within a certain
 *             time step are stored
 * @param filename a string containing the filename with or without the directory to store it to
 */
void sim_data_to_csv(std::vector<sim_data_tuple> &data, const std::string &filename)
{
    std::ofstream file;
    unsigned int current_node = 0;
    file.open("results.csv");
    file << "iteration,x,y,vx,vy,density\n"; 
    for(auto time = 0; time < data.size(); ++time)
    {
        for(auto y = 1; y < VERTICAL_NODES-1; ++y)
        {
            for(auto x = 1; x < HORIZONTAL_NODES-1; ++x)
            {
                current_node = lbm_access::get_node_index(x,y);
                file << time << ',' << x << ',' << y << ',' 
                << std::get<0>(data[time])[current_node][0] << ',' 
                << std::get<0>(data[time])[current_node][1] << ',' << std::get<1>(data[time])[current_node] << '\n';
            }
        }
    }
    file.close();
}

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
void parallel_domain_sim_data_to_csv(std::vector<sim_data_tuple> &data, const std::string &filename)
{
    std::ofstream file;
    unsigned int current_node = 0;
    file.open("results.csv");
    file << "iteration,x,y,vx,vy,density\n"; 
    for(auto time = 0; time < data.size(); ++time)
    {
        for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
        {
            for(auto y = subdomain * SUBDOMAIN_HEIGHT + subdomain; y < (subdomain+1) * SUBDOMAIN_HEIGHT + subdomain; ++y)
            {
                if(!(y == 0 || y == VERTICAL_NODES - 1))
                for(auto x = 1; x < HORIZONTAL_NODES - 1; ++x)
                {
                    current_node = lbm_access::get_node_index(x,y);
                    file << time << ',' << x << ',' << y - subdomain << ',' 
                    << std::get<0>(data[time])[current_node][0] << ',' 
                    << std::get<0>(data[time])[current_node][1] << ',' << std::get<1>(data[time])[current_node] << '\n';
                }
            }
        }


    }
    file.close();
}

/**
 * @brief Writes a csv configuration file based on the given struct settings.
 *        The following parameters of the struct must be specified:
 *        
 *        Always:
 *        - debug_mode
 *        - results_to_csv
 *        - is_parallel
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
 * @param settings a struct specifying the essential parameters of the algorithm.
 */
void write_csv_config_file(const Settings &settings)
{
    std::ofstream file;

    unsigned long total_node_count = 0;
    unsigned int subdomain_count = 0;
    unsigned int subdomain_height = 0;
    unsigned int buffer_count = 0;
    unsigned int shift_offset = 0;
    unsigned int vertical_nodes = 0;

    file.open("config.csv");

    bool is_parallel = 
        (
            settings.algorithm == "parallel_two_lattice" | 
            settings.algorithm == "parallel_two_lattice_framework" | 
            settings.algorithm == "parallel_two_step" | 
            settings.algorithm == "parallel_swap" | 
            settings.algorithm == "parallel_shift"
        );
    
    bool use_buffered_layout = is_parallel && settings.algorithm != "parallel_two_lattice";

    // Set algorithm
    if (!is_valid_algorithm(settings.algorithm))
    {
        std::cout << "The following algorithm is invalid and was not written to the csv file: " << settings.algorithm << "\n";
    }
    else
    {
        file << "algorithm," << settings.algorithm << "\n";
    }

    // Debug mode
    file << "debug_mode," << settings.debug_mode << "\n";

    // Write results to csv file?
    file << "results_to_csv," << settings.results_to_csv << "\n";

    // Set access patterns
    if(
        settings.access_pattern != "collision" && 
        settings.access_pattern != "stream" &&
        settings.access_pattern != "bundle")
    {
        std::cout << "The following access pattern is invalid and was not written to the csv file: " << settings.access_pattern << "\n";
    }
    else
    {
        file << "access_pattern," << settings.access_pattern << "\n";
    }

    // Set relaxation time
    file << "relaxation_time," << settings.relaxation_time << "\n";

    // Set time steps
    file << "time_steps," << settings.time_steps << "\n";
    
    /* Setup of domain parameters for parallel or sequential algorithms */
    file << "horizontal_nodes," << settings.horizontal_nodes << "\n";

    if(use_buffered_layout) 
    {
        file << "vertical_nodes_excluding_buffers," << settings.vertical_nodes_excluding_buffers << "\n";
        subdomain_count = settings.subdomain_count;
        file << "subdomain_count," << settings.subdomain_count << "\n";
        
        if(subdomain_count > 0)
        {
            subdomain_height = settings.vertical_nodes_excluding_buffers / settings.subdomain_count;
            file << "subdomain_height," << subdomain_height << "\n";
        }
        else
        {
            std::cout << "Invalid subdomain count (will not be written to the csv file): " << subdomain_count << "\n";
        }

        buffer_count = settings.subdomain_count - 1;
        file << "buffer_count," << buffer_count << "\n";

        vertical_nodes = settings.vertical_nodes_excluding_buffers + buffer_count;
        file << "vertical_nodes," << vertical_nodes << "\n";
        total_node_count = vertical_nodes * settings.horizontal_nodes;
        file << "total_node_count," << total_node_count << "\n";
        file << "total_nodes_excluding_buffers," << settings.vertical_nodes_excluding_buffers * settings.horizontal_nodes << "\n";
    }
    else // Non-buffered layout, i.e. sequential algorithm or non-framework parallel two-lattice
    {
        vertical_nodes = settings.vertical_nodes_excluding_buffers;
        file << "vertical_nodes," << vertical_nodes << "\n";
        file << "vertical_nodes_excluding_buffers," << vertical_nodes << "\n";
        file << "total_nodes_excluding_buffers," << (settings.vertical_nodes_excluding_buffers * settings.horizontal_nodes) << "\n";
        total_node_count = settings.vertical_nodes_excluding_buffers * settings.horizontal_nodes;
        file << "total_node_count," << total_node_count << "\n";

        if(is_parallel) // non-framework parallel two-lattice
        {
            subdomain_count = settings.subdomain_count;
            file << "subdomain_count," << settings.subdomain_count << "\n";
        
            subdomain_height = settings.vertical_nodes_excluding_buffers / settings.subdomain_count;
            file << "subdomain_height," << subdomain_height << "\n";

            buffer_count = 0;
            file << "buffer_count," << buffer_count << "\n";
        }
        else // Sequential algorithm
        {
            // Those specifications are theoretically unnecessary for sequential algorithms but are made for consistency
            file << "subdomain_height," << 0 << "\n";
            subdomain_count = 0;
            file << "subdomain_count," << subdomain_count << "\n";
            buffer_count = 0;
            file << "buffer_count," << 0 << "\n";
        }
    }

    // Specification of parameters for shift algorithms
    shift_offset = settings.horizontal_nodes + 1;
    file << "shift_offset," << shift_offset << "\n";
    file << "shift_distribution_value_count," << 
    total_node_count + buffer_count * settings.horizontal_nodes + subdomain_count * shift_offset << "\n";

    // Specification of inlet and outlet parameters
    file << "inlet_velocity," << settings.inlet_velocity[0] << "," << settings.inlet_velocity[1] << "\n";
    file << "outlet_velocity," << settings.outlet_velocity[0] << "," << settings.outlet_velocity[1] << "\n";
    file << "inlet_density," << settings.inlet_density << "\n";
    file << "outlet_density," << settings.outlet_density << "\n";

    file.close();
}

/**
 * @brief Returns a Settings struct that is set up according to the specified csv file.
 * 
 * @param filename a string containing the name of the csv file
 * @return Settings a struct containing the specified values, missing values will be initialized by default
 */
Settings retrieve_settings_from_csv(const std::string &filename)
{
    Settings settings;

    /* Open file */
    std::ifstream settings_file{filename};

    if(!settings_file.is_open())
    {
        std::cout << "Could not open file " << filename << std::endl;
    }
    else
    {
        std::vector<std::string> line_contents{};
        std::string line;

        while(std::getline(settings_file,line))
        {
            Tokenizer tokenizer(line);
            line_contents.assign(tokenizer.begin(),tokenizer.end());

            if(line_contents[0] == "algorithm")
            {
                settings.algorithm = line_contents[1]; 
            }
            else if(line_contents[0] == "debug_mode")
            {
                settings.debug_mode = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "results_to_csv")
            {
                settings.results_to_csv = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "access_pattern")
            {
                settings.access_pattern = line_contents[1];
            }
            else if(line_contents[0] == "relaxation_time")
            {
                settings.relaxation_time = std::stod(line_contents[1]);
            }
            else if(line_contents[0] == "time_steps")
            {
                settings.time_steps = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "horizontal_nodes")
            {
                settings.horizontal_nodes = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "vertical_nodes")
            {
                settings.vertical_nodes = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "vertical_nodes_excluding_buffers")
            {
                settings.vertical_nodes_excluding_buffers = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "total_node_count")
            {
                settings.total_node_count = std::stol(line_contents[1]);
            }
            else if(line_contents[0] == "total_nodes_excluding_buffers")
            {
                settings.total_nodes_excluding_buffers = std::stol(line_contents[1]);
            }
            else if(line_contents[0] == "subdomain_height")
            {
                settings.subdomain_height = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "subdomain_count")
            {
                settings.subdomain_count = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "buffer_count")
            {
                settings.buffer_count = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "shift_offset")
            {
                settings.shift_offset = std::stoi(line_contents[1]);
            }
            else if(line_contents[0] == "shift_distribution_value_count")
            {
                settings.shift_distribution_value_count = std::stol(line_contents[1]);
            }
            else if(line_contents[0] == "inlet_velocity")
            {
                settings.inlet_velocity = {std::stod(line_contents[1]), std::stod(line_contents[2])};
            }
            else if(line_contents[0] == "outlet_velocity")
            {
                settings.outlet_velocity = {std::stod(line_contents[1]), std::stod(line_contents[2])};
            }
            else if(line_contents[0] == "inlet_density")
            {
                settings.inlet_density = std::stod(line_contents[1]);
            }
            else if(line_contents[0] == "outlet_density")
            {
                settings.outlet_density = std::stod(line_contents[1]);
            }
        }
        
        settings_file.close();
    }

    return settings;
}

Settings csv_test(const std::string& algorithm)
{
    Settings result;
    result.debug_mode = 0;
    result.results_to_csv = 1;
    result.horizontal_nodes = 200;
    result.algorithm = algorithm;
    result.vertical_nodes_excluding_buffers = 50;
    result.subdomain_count = 5;
    result.access_pattern = "bundle";
    result.time_steps = 800;
    return result;
}