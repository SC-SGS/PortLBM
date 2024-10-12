#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include "./include/defines.hpp"
#include "./include/old_file_interaction.hpp"

#include <hpx/hpx_init.hpp>

#include<sys/sysinfo.h>

char* algorithm_picker(unsigned int number_of_cores)
{
    std::string command_line_instruction = "./lattice_boltzmann";

    // Specify HPX settings
    command_line_instruction.append(" -t");
    command_line_instruction.append(std::to_string(number_of_cores));
    command_line_instruction.append(" --hpx:bind=thread:0-");
    command_line_instruction.append(std::to_string(number_of_cores - 1));
    command_line_instruction.append("=core:0-");
    command_line_instruction.append(std::to_string(number_of_cores - 1));
    command_line_instruction.append(".pu:0");

    // Convert to char array
    const int length = command_line_instruction.length(); 
    char* result = new char[length + 1]; 
    strcpy(result, command_line_instruction.c_str()); 

    return result;
}

void execute_sequential_tests
(
    const std::vector<std::string>& algorithms,
    const std::vector<std::string>& access_patterns,
    Settings& settings,
    unsigned int test_runs,
    const std::string& test_name
)
{
    std::cout << "Starting sequential simulations." << std::endl;

    std::ofstream results_file;
    hpx::chrono::high_resolution_timer timer;

    double runtime = 0;
    double total = algorithms.size() * access_patterns.size() * test_runs;
    double progress = 0; 

    std::string current_line = "";

    settings.subdomain_count = 0;

    for(auto i = 0; i < test_runs; ++i)
    {
        for(std::string algorithm : algorithms)
        {
            settings.algorithm = algorithm;

            for(const std::string& access_pattern : access_patterns) 
            {
                settings.access_pattern = access_pattern;

                // Write options file
                write_csv_config_file(settings);

                // Execute algorithm

                timer.restart();
                system("./lattice_boltzmann");      
                runtime = timer.elapsed();

                // Evaluate data
                current_line = algorithm + "," + access_pattern + "," + std::to_string(1) + "," + std::to_string(runtime) + "\n";
                results_file.open(test_name + "_results.csv", std::ios::out | std::ios::app);
                results_file << current_line;
                results_file.close();
                
                // Prepare for next test case
                current_line = {};
                runtime = 0;
            }
        }
        std::cout << "Finished test run " << std::to_string(i+1) << " / " << test_runs << std::endl;  
    }        
}

void execute_parallel_strong_scaling_tests
(
    const std::vector<std::string>& algorithms,
    const std::vector<std::string>& access_patterns,
    const std::vector<unsigned int>& core_counts,
    Settings& settings,
    unsigned int test_runs,
    const std::string& test_name
)
{
    std::ofstream results_file;
    hpx::chrono::high_resolution_timer timer;

    double runtime = 0;
    double total = algorithms.size() * access_patterns.size() * test_runs * core_counts.size();
    double progress = 0; 

    std::string current_line = "";

    std::cout << "Starting parallel simulations." << std::endl;

    for(auto i = 0; i < test_runs; ++i)
    {
        for(std::string algorithm : algorithms)
        {
            settings.algorithm = algorithm;

            for(const std::string& access_pattern : access_patterns) 
            {
                settings.access_pattern = access_pattern;

                for(const auto current_cores : core_counts)
                {
                    settings.subdomain_count = current_cores;

                    // Write options file
                    write_csv_config_file(settings);

                    // Execute algorithm
                    timer.restart();
                    system(algorithm_picker(current_cores));       
                    runtime = timer.elapsed();

                    // Evaluate data
                    current_line = algorithm + "," + access_pattern + "," + std::to_string(current_cores) + "," + std::to_string(runtime) + "\n";
                    results_file.open(test_name + "_results.csv", std::ios::out | std::ios::app);
                    results_file << current_line;
                    results_file.close();
                    
                    // Prepare for next test case
                    current_line = {};
                    runtime = 0;
                }
            }
        }  
        std::cout << "Finished test run " << std::to_string(i+1) << " / " << test_runs << std::endl;  
    }
}

void execute_parallel_weak_scaling_tests
(
    const std::vector<std::string>& algorithms,
    const std::vector<std::string>& access_patterns,
    const std::vector<unsigned int>& core_counts,
    Settings& settings,
    unsigned int test_runs,
    unsigned int base_subdomain_height,
    const std::string& test_name
)
{
    std::ofstream results_file;
    hpx::chrono::high_resolution_timer timer;

    double runtime = 0;
    double total = algorithms.size() * access_patterns.size() * test_runs * core_counts.size();
    double progress = 0; 

    std::string current_line = "";

    std::cout << "Starting parallel simulations." << std::endl;

    for(auto i = 0; i < test_runs; ++i)
    {
        for(std::string algorithm : algorithms)
        {
            settings.algorithm = algorithm;

            for(const std::string& access_pattern : access_patterns) 
            {
                settings.access_pattern = access_pattern;

                for(const auto current_cores : core_counts)
                {
                    settings.subdomain_count = current_cores;
                    settings.vertical_nodes_excluding_buffers = base_subdomain_height * current_cores;

                    // Write options file
                    write_csv_config_file(settings);

                    // Execute algorithm
                    timer.restart();
                    system(algorithm_picker(current_cores));       
                    runtime = timer.elapsed();

                    // Evaluate data
                    current_line = algorithm + "," + access_pattern + "," + std::to_string(current_cores) + "," + std::to_string(runtime) + "\n";
                    results_file.open(test_name + "_results.csv", std::ios::out | std::ios::app);
                    results_file << current_line;
                    results_file.close();
                    
                    // Prepare for next test case
                    current_line = {};
                    runtime = 0;
                }
            }
        }  
        std::cout << "Finished test run " << std::to_string(i+1) << " / " << test_runs << std::endl;  
    }
}

void strong_scaling_tests
(
    const std::vector<std::string> &sequential_algorithms,
    const std::vector<std::string> &parallel_algorithms,
    const std::vector<std::string> &access_patterns,
    const std::vector<unsigned int> &multi_core_counts,
    double relaxation_time,
    unsigned int time_steps
)
{
    unsigned int test_runs = 5;

    std::cout << "Starting strong scaling test." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Results will be stored to 'strong_scaling_results.csv'." << std::endl;
    
    std::ofstream results_file;
    results_file.open("../runtimes/strong_scaling_results.csv", std::ios::out | std::ios::app);

    std::string current_line{"algorithm,access_pattern,cores,runtime[s]\n"};
    results_file << current_line;
    results_file.close();

    current_line = {};

    Settings settings;
    settings.debug_mode = 0;
    settings.results_to_csv = 0;
    settings.horizontal_nodes = 128; // 512
    settings.vertical_nodes_excluding_buffers = 128; // 512
    settings.time_steps = time_steps;

    double runtime = 0;

    execute_sequential_tests(sequential_algorithms, access_patterns, settings, test_runs, "../runtimes/strong_scaling");
    execute_parallel_strong_scaling_tests(parallel_algorithms, access_patterns, multi_core_counts, settings, test_runs, "../runtimes/strong_scaling");
    
    std::cout << "Strong scaling test fully completed. " << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

void weak_scaling_tests
(
    const std::vector<std::string> &sequential_algorithms,
    const std::vector<std::string> &parallel_algorithms,
    const std::vector<std::string> &access_patterns,
    const std::vector<unsigned int> &multi_core_counts,
    double relaxation_time,
    unsigned int time_steps  
)
{
    unsigned int test_runs = 5;

    std::cout << "Starting weak scaling test." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Results will be stored to 'weak_scaling_results.csv'." << std::endl;

    std::ofstream results_file;
    results_file.open("../runtimes/weak_scaling_results.csv", std::ios::out | std::ios::app);

    std::string current_line{"algorithm,access_pattern,cores,runtime[s]\n"};
    results_file << current_line;
    results_file.close();

    current_line = {};

    unsigned int base_subdomain_height = 64; // 128
    unsigned int horizontal_nodes = 64; // 128

    Settings settings;
    settings.debug_mode = 0;
    settings.results_to_csv = 0;
    settings.horizontal_nodes = horizontal_nodes;
    settings.time_steps = time_steps;

    double runtime = 0;

    settings.vertical_nodes_excluding_buffers = base_subdomain_height * 1;
    execute_sequential_tests(sequential_algorithms, access_patterns, settings, test_runs, "../runtimes/weak_scaling");

    execute_parallel_weak_scaling_tests(parallel_algorithms, access_patterns, multi_core_counts, settings, test_runs, base_subdomain_height, "../runtimes/weak_scaling");

    std::cout << "Weak scaling test fully completed. " << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    /* Selections that actually vary */
    std::vector<std::string> sequential_algorithms{"sequential_two_lattice"}; //{"sequential_two_lattice", "sequential_two_step", "sequential_swap", "sequential_shift"};
    std::vector<std::string> parallel_algorithms{"parallel_two_lattice"}; //{"parallel_two_lattice", "parallel_two_lattice_framework", "parallel_two_step", "parallel_swap", "parallel_shift"};
    std::vector<std::string> access_patterns{"collision", "stream", "bundle"};

    /* Selections assumed static */
    const double relaxation_time = 1.4;
    const unsigned int time_steps = 5;
    const int write_csv = 0;
    const int debug_mode = 0;

    unsigned int available_cores = std::thread::hardware_concurrency() / 2;
    std::cout << "Up to " << available_cores << " concurrent threads are supported.\n";

    std::vector<unsigned int> multicore_setups;
    unsigned int current_max_core_count = 2;

    while(current_max_core_count <= available_cores)
    {
        multicore_setups.push_back(current_max_core_count);
        current_max_core_count *= 2;
    }

    std::cout << "Performing tests for up to" << current_max_core_count << " concurrent threads.\n";

    weak_scaling_tests(sequential_algorithms, parallel_algorithms, access_patterns, multicore_setups, relaxation_time, time_steps);
    strong_scaling_tests(sequential_algorithms, parallel_algorithms, access_patterns, multicore_setups, relaxation_time, time_steps);

    std::cout << "Benchmark finished." << std::endl;
}
