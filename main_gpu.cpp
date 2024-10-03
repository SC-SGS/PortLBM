#include <hpx/hpx_init.hpp>
#include <hpx/execution.hpp>
#include <string>
#include "./include/defines.hpp"
#include "./include/file_interaction.hpp"
#include "include/lbm_execution.hpp"
#include "./include/simulation.hpp"
#include "./include/gpu_two_lattice.hpp"

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::cout << "Starting LBM on GPU test with two-lattice algorithm..." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // Legacy settings
    Settings settings;
    settings.debug_mode = 1;
    settings.results_to_csv = 0;
    settings.horizontal_nodes = 10;
    settings.vertical_nodes_excluding_buffers = 21;
    settings.vertical_nodes = 21;
    settings.time_steps = 100;
    settings.subdomain_count = 0;
    settings.access_pattern = "stream";
    std::cout << "settings.vertical_nodes = " << settings.vertical_nodes << std::endl;
    std::cout << "settings.horizontal_nodes = " << settings.horizontal_nodes << std::endl;
    write_csv_config_file(settings);
    settings = retrieve_settings_from_csv("config.csv");
    setup_global_variables(settings);


    std::cout << "HORIZONTAL_NODES = " << HORIZONTAL_NODES << std::endl;
    std::cout << "VERTICAL_NODES = " << VERTICAL_NODES << std::endl;

    // New settings 
    Properties properties
    (
        true,
        false,
        settings.relaxation_time,
        settings.time_steps,
        settings.vertical_nodes_excluding_buffers,
        settings.horizontal_nodes,
        settings.total_nodes_excluding_buffers,
        settings.total_node_count,
        0,
        0,
        0,
        0,
        settings.inlet_velocity[0],
        settings.inlet_velocity[1],
        settings.outlet_velocity[0],
        settings.outlet_velocity[1],
        "stream",
        "gpu two-lattice"
    );

    SimulationResults sim_results(properties);
    SimulationData<lbm_access::LBMStreamAccessor> sim_data(properties);

    // Legacy initializations
    // std::vector<double> distribution_values_0(0, TOTAL_NODE_COUNT * DIRECTION_COUNT);
    std::vector<unsigned int> nodes(0, TOTAL_NODE_COUNT);
    std::vector<unsigned int> fluid_nodes(0, TOTAL_NODE_COUNT);
    // std::vector<bool> phase_information(false, TOTAL_NODE_COUNT);

    border_swap_information swap_info;
    setup_example_domain(*sim_data.distribution_values_0, nodes, fluid_nodes, *sim_data.phase_information, *sim_data.lbm_accessor, true);
    *sim_data.distribution_values_1 = *sim_data.distribution_values_0; 
    swap_info = bounce_back::retrieve_border_swap_info(fluid_nodes, *sim_data.phase_information);
    LegacyData legacy_data(swap_info, lbm_access::stream);

    if(DEBUG_MODE)
    {
        debug_prints(*sim_data.distribution_values_0, nodes, fluid_nodes, *sim_data.phase_information, swap_info);
    }

    gpu_two_lattice::run(properties, sim_data, legacy_data);

    std::cout << "Control comes back to main \n";

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}