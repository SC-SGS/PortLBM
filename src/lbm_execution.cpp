#include "../include/lbm_execution.hpp"


void debug_prints
(
    const std::vector<double> &distribution_values,
    const std::vector<unsigned int> &nodes,
    const std::vector<unsigned int> &fluid_nodes,
    const std::vector<bool> &phase_information,
    const border_swap_information &swap_info
)
{
    std::cout << std::endl;
    std::cout << "Starting simulation..." << std::endl;
    std::cout << std::endl;
    std::cout << std::setprecision(3) << std::fixed;

    /* Color message */
    to_console::print_ansi_color_message();

    /* Illustration of the phase information */
    std::cout << "Illustration of lattice: " << std::endl;
    to_console::print_phase_vector(phase_information);
    std::cout << std::endl;

    /* Overview */
    std::cout << "Enumeration of all nodes within the lattice: " << std::endl;
    to_console::buffered::print_vector(nodes);
    std::cout << std::endl;

    std::cout << "Enumeration of all fluid nodes within the simulation domain: " << std::endl;
    to_console::print_vector(fluid_nodes, HORIZONTAL_NODES - 2);
    std::cout << std::endl;

    std::cout << "Swap info:" << std::endl;
    for(const auto& current : swap_info)
        to_console::print_vector(current, current.size());
    std::cout << std::endl;

    std::cout << "Initial distributions:" << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, ACCESS_FUNCTION);
    std::cout << std::endl;
}

void debug_prints
(
    const std::vector<double> &distribution_values,
    const std::vector<unsigned int> &nodes,
    const std::vector<unsigned int> &fluid_nodes,
    const std::vector<bool> &phase_information,
    const std::vector<border_swap_information> &swap_info
)
{
    std::cout << std::endl;
    std::cout << "Starting simulation..." << std::endl;
    std::cout << std::endl;
    std::cout << std::setprecision(3) << std::fixed;

    /* Color message */
    to_console::print_ansi_color_message();

    /* Illustration of the phase information */
    std::cout << "Illustration of lattice: " << std::endl;
    to_console::print_phase_vector(phase_information);
    std::cout << std::endl;

    /* Overview */
    std::cout << "Enumeration of all nodes within the lattice: " << std::endl;
    to_console::buffered::print_vector(nodes);
    std::cout << std::endl;

    std::cout << "Enumeration of all fluid nodes within the simulation domain: " << std::endl;
    to_console::print_vector(fluid_nodes, HORIZONTAL_NODES - 2);
    std::cout << std::endl;

    std::cout << "Subdomain-wise border swap information: " << std::endl;
    for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
    {
        std::cout << "Subdomain " <<  subdomain << ":" << std::endl;
        for(const auto& current : swap_info[subdomain])
        to_console::print_vector(current, current.size());
    }
    std::cout << std::endl;

    std::cout << "Initial distributions:" << std::endl;
    to_console::buffered::print_distribution_values(distribution_values, ACCESS_FUNCTION);
    std::cout << std::endl;
}

void debug_prints
(
    const std::vector<double> &distribution_values,
    const std::vector<unsigned int> &nodes,
    const std::vector<unsigned int> &fluid_nodes,
    const std::vector<bool> &phase_information
)
{
    /* Illustration of the phase information */
    std::cout << "Illustration of lattice: " << std::endl;
    to_console::print_phase_vector(phase_information);
    std::cout << std::endl;

    /* Overview */
    std::cout << "Enumeration of all nodes within the lattice: " << std::endl;
    to_console::print_vector(nodes);
    std::cout << std::endl;

    std::cout << "Enumeration of all fluid nodes within the simulation domain: " << std::endl;
    to_console::print_vector(fluid_nodes, HORIZONTAL_NODES - 2);
    std::cout << std::endl;

    std::cout << "Initial distributions:" << std::endl;
    to_console::print_distribution_values(distribution_values, ACCESS_FUNCTION);
    std::cout << std::endl;
}

void setup_global_variables(const Settings &settings)
{
    DEBUG_MODE = settings.debug_mode;
    std::cout << "DEBUG_MODE = " << DEBUG_MODE << std::endl;
    RESULTS_TO_CSV = settings.results_to_csv;
    std::cout << "RESULTS_TO_CSV = " << RESULTS_TO_CSV << std::endl;

    VERTICAL_NODES = settings.vertical_nodes;
    std::cout << "settings.vertical_nodes = " << settings.vertical_nodes << std::endl;
    std::cout << "VERTICAL_NODES = " << VERTICAL_NODES << std::endl;
    HORIZONTAL_NODES = settings.horizontal_nodes;
    std::cout << "settings.horizontal_nodes = " << settings.horizontal_nodes << std::endl;
    std::cout << "HORIZONTAL_NODES = " << HORIZONTAL_NODES << std::endl;
    TOTAL_NODE_COUNT = settings.total_node_count;
    std::cout << "TOTAL_NODE_COUNT = " << TOTAL_NODE_COUNT << std::endl;

    RELAXATION_TIME = settings.relaxation_time;
    TIME_STEPS = settings.time_steps;

    SUBDOMAIN_HEIGHT = settings.subdomain_height;
    SUBDOMAIN_COUNT = settings.subdomain_count;
    BUFFER_COUNT = settings.buffer_count;
    TOTAL_NODES_EXCLUDING_BUFFERS = settings.total_nodes_excluding_buffers;
    std::cout << "TOTAL_NODE_COUNT_EXCLUDING_BUFFERS = " << TOTAL_NODES_EXCLUDING_BUFFERS << std::endl;

    INLET_VELOCITY = settings.inlet_velocity;
    OUTLET_VELOCITY = settings.outlet_velocity;
    INLET_DENSITY = settings.inlet_density;
    OUTLET_DENSITY = settings.outlet_density;

    SHIFT_OFFSET = settings.shift_offset;
    SHIFT_DISTRIBUTION_VALUE_COUNT = settings.shift_distribution_value_count;

    if (settings.algorithm != "sequential_shift" && settings.algorithm != "parallel_shift")
    {
        if (settings.access_pattern == "collision")
        {
            ACCESS_FUNCTION = lbm_access::collision;
        }
        else if (settings.access_pattern == "stream")
        {
            ACCESS_FUNCTION = lbm_access::stream;
        }
        else
        {
            ACCESS_FUNCTION = lbm_access::bundle;
        }
    }
    else if (settings.algorithm == "sequential_shift")
    {
        // if (settings.access_pattern == "collision")
        // {
        //     ACCESS_FUNCTION = sequential_shift::access_functions::collision;
        // }
        // else if (settings.access_pattern == "stream")
        // {
        //     ACCESS_FUNCTION = sequential_shift::access_functions::stream;
        // }
        // else
        // {
        //     ACCESS_FUNCTION = sequential_shift::access_functions::bundle;
        // }
    }
    else
    {
        // if (settings.access_pattern == "collision")
        // {
        //     ACCESS_FUNCTION = parallel_shift_framework::access_functions::collision;
        // }
        // else if (settings.access_pattern == "stream")
        // {
        //     ACCESS_FUNCTION = parallel_shift_framework::access_functions::stream;
        // }
        // else
        // {
        //     ACCESS_FUNCTION = parallel_shift_framework::access_functions::bundle;
        // }
    }
}

// void execute_sequential_two_lattice()
// {
//     std::vector<double> distribution_values_0(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     border_swap_information swap_info;

//     setup_example_domain(distribution_values_0, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION, DEBUG_MODE);
//     swap_info = bounce_back::retrieve_border_swap_info(fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values_0, nodes, fluid_nodes, phase_information, swap_info);
//     }

//     std::vector<double> distribution_values_1 = distribution_values_0;

//     if(DEBUG_MODE)
//     {
//         sequential_two_lattice::run_debug
//         (
//             fluid_nodes, 
//             swap_info, 
//             distribution_values_0, 
//             distribution_values_1,   
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
//     else
//     {
//         sequential_two_lattice::run
//         (
//             fluid_nodes, 
//             swap_info, 
//             distribution_values_0, 
//             distribution_values_1,   
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }

// }

// void execute_sequential_two_step()
// {
//     std::vector<double> distribution_values(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     border_swap_information swap_info;

//     setup_example_domain(distribution_values, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION, DEBUG_MODE);
//     swap_info = bounce_back::retrieve_border_swap_info(fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values, nodes, fluid_nodes, phase_information, swap_info);
//         sequential_two_step::run_debug
//         (
//             fluid_nodes, 
//             distribution_values, 
//             swap_info, 
//             ACCESS_FUNCTION, 
//             TIME_STEPS
//         );
//     }
//     else
//     {
//         sequential_two_step::run
//         (
//             fluid_nodes, 
//             distribution_values, 
//             swap_info, 
//             ACCESS_FUNCTION, 
//             TIME_STEPS
//         );
//     }


// }

// void execute_sequential_swap()
// {
//     std::vector<double> distribution_values(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);

//     setup_example_domain(distribution_values, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION, DEBUG_MODE);

//     border_swap_information bsi = sequential_swap::retrieve_swap_info(fluid_nodes, phase_information);
   
//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values, nodes, fluid_nodes, phase_information, bsi);
//         sequential_swap::run_debug(fluid_nodes, bsi, distribution_values, ACCESS_FUNCTION, TIME_STEPS);
//     }
//     else
//     {
//         sequential_swap::run(fluid_nodes, bsi, distribution_values, ACCESS_FUNCTION, TIME_STEPS);
//     }

    
// }

// void execute_sequential_shift()
// {
//     std::vector<double> distribution_values(0, (TOTAL_NODE_COUNT + SHIFT_OFFSET) * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     border_swap_information swap_info;

//     sequential_shift::setup_example_domain(distribution_values, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION);
//     swap_info = bounce_back::retrieve_border_swap_info(fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values, nodes, fluid_nodes, phase_information, swap_info);
//         sequential_shift::run_debug(fluid_nodes, distribution_values, swap_info, ACCESS_FUNCTION, TIME_STEPS);
//     }
//     else
//     {
//         sequential_shift::run(fluid_nodes, distribution_values, swap_info, ACCESS_FUNCTION, TIME_STEPS);
//     }
// }

// void execute_parallel_two_lattice()
// {
//     std::vector<double> distribution_values_0(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     border_swap_information swap_info;

//     setup_example_domain(distribution_values_0, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION, DEBUG_MODE);
//     swap_info = bounce_back::retrieve_border_swap_info(fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values_0, nodes, fluid_nodes, phase_information, swap_info);   
//     }

//     std::vector<double> distribution_values_1 = distribution_values_0;

//     if(DEBUG_MODE)
//     {
//         parallel_two_lattice::run_debug
//         (
//             fluid_nodes, 
//             swap_info, 
//             distribution_values_0, 
//             distribution_values_1,   
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
//     else
//     {
//         parallel_two_lattice::run
//         (
//             fluid_nodes, 
//             swap_info, 
//             distribution_values_0, 
//             distribution_values_1,   
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         ); 
//     }
// }

// void execute_parallel_two_lattice_framework()
// {
//     std::vector<double> distribution_values_0(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     border_swap_information swap_info;

//     parallel_framework::setup_parallel_domain(distribution_values_0, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION);
    
//     std::vector<start_end_it_tuple> subdomain_fluid_bounds;
//     for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
//     {
//         subdomain_fluid_bounds.push_back(parallel_framework::get_subdomain_fluid_node_pointers(subdomain, fluid_nodes));
//     }

//     swap_info = parallel_framework::retrieve_border_swap_info(subdomain_fluid_bounds, fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values_0, nodes, fluid_nodes, phase_information, swap_info);     
//     }

//     std::vector<double> distribution_values_1 = distribution_values_0;

//     if(DEBUG_MODE)
//     {
//         parallel_two_lattice_framework::run_debug
//         (
//             subdomain_fluid_bounds, 
//             swap_info, 
//             distribution_values_0, 
//             distribution_values_1,   
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
//     else
//     {
//         parallel_two_lattice_framework::run
//         (
//             subdomain_fluid_bounds, 
//             swap_info, 
//             distribution_values_0, 
//             distribution_values_1,   
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );        
//     }

// }

// void execute_parallel_two_step()
// {
//     std::vector<double> distribution_values(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     border_swap_information swap_info;

//     parallel_framework::setup_parallel_domain(distribution_values, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION);

//     std::vector<start_end_it_tuple> subdomain_fluid_bounds;
//     for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
//     {
//         subdomain_fluid_bounds.push_back(parallel_framework::get_subdomain_fluid_node_pointers(subdomain, fluid_nodes));
//     }

//     swap_info = parallel_framework::retrieve_border_swap_info(subdomain_fluid_bounds, fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values, nodes, fluid_nodes, phase_information, swap_info);  
//         parallel_two_step_framework::run_debug
//         (
//             subdomain_fluid_bounds, 
//             distribution_values,
//             swap_info, 
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
//     else
//     {
//         parallel_two_step_framework::run
//         (
//             subdomain_fluid_bounds, 
//             distribution_values,
//             swap_info, 
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
// }

// void execute_parallel_swap()
// {
//     std::vector<double> distribution_values(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     border_swap_information swap_info;

//     parallel_framework::setup_parallel_domain(distribution_values, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION);
    
//     std::vector<start_end_it_tuple> subdomain_fluid_bounds;
//     for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
//     {
//         subdomain_fluid_bounds.push_back(parallel_framework::get_subdomain_fluid_node_pointers(subdomain, fluid_nodes));
//     }

//     swap_info = sequential_swap::retrieve_swap_info(fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values, nodes, fluid_nodes, phase_information, swap_info);  
//         parallel_swap_framework::run_debug
//         (
//             subdomain_fluid_bounds, 
//             distribution_values,
//             swap_info, 
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
//     else
//     {
//         parallel_swap_framework::run
//         (
//             subdomain_fluid_bounds, 
//             distribution_values,
//             swap_info, 
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
// }

// void execute_parallel_shift()
// {
//     std::vector<double> distribution_values;
//     std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
//     std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
//     std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);
//     std::vector<border_swap_information> swap_info;    

//     parallel_shift_framework::setup_parallel_domain(distribution_values, nodes, fluid_nodes, phase_information, ACCESS_FUNCTION);

//     std::vector<start_end_it_tuple> subdomain_fluid_bounds;
//     for(auto subdomain = 0; subdomain < SUBDOMAIN_COUNT; ++subdomain)
//     {
//         subdomain_fluid_bounds.push_back(parallel_framework::get_subdomain_fluid_node_pointers(subdomain, fluid_nodes));
//     }

//     swap_info = parallel_framework::subdomain_wise_border_swap_info(subdomain_fluid_bounds, fluid_nodes, phase_information);

//     if(DEBUG_MODE)
//     {
//         debug_prints(distribution_values, nodes, fluid_nodes, phase_information, swap_info);  
//         parallel_shift_framework::run_debug
//         (
//             subdomain_fluid_bounds, 
//             swap_info, 
//             distribution_values, 
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
//     else
//     {
//         parallel_shift_framework::run
//         (
//             subdomain_fluid_bounds, 
//             swap_info, 
//             distribution_values, 
//             ACCESS_FUNCTION,
//             TIME_STEPS
//         );
//     }
// }

void select_and_execute(const std::string &algorithm)
{
    if(algorithm == "sequential_two_lattice")
    {
        // execute_sequential_two_lattice();
    }
    // else if(algorithm == "sequential_two_step")
    // {
    //     execute_sequential_two_step();
    // }
    // else if(algorithm == "sequential_swap")
    // {
    //     execute_sequential_swap();
    // }
    // else if(algorithm == "sequential_shift")
    // {
    //     execute_sequential_shift();
    // }
    else if(algorithm == "parallel_two_lattice")
    {
        // execute_parallel_two_lattice();
    }
    // else if(algorithm == "parallel_two_lattice_framework")
    // {
    //     execute_parallel_two_lattice_framework();
    // }
    // else if(algorithm == "parallel_two_step")
    // {
    //     execute_parallel_two_step();
    // }
    // else if(algorithm == "parallel_swap")
    // {
    //     execute_parallel_swap();
    // }
    // else if(algorithm == "parallel_shift")
    // {
    //     execute_parallel_shift();
    // }
    else
    {
        std::cout << "Invalid algorithm: " << algorithm << std::endl; 
    }
}