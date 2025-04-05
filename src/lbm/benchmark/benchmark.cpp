/**
 * @file        benchmark.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the definition of benchmark functionality.
 * 
 * @version     1.1
 * 
 * @date        April 2025
 * 
 * @copyright   Copyright (c) Marcel Graf
 * 
 */

// LBM Benchmark
#include "../../../include/lbm/benchmark/benchmark.hpp"


// PROGRESS ///////////////////////////////////////////////////////////////////////////////////////////////////////////

lbm::benchmark::Progress::Progress(const unsigned int total_test_rows)
:
total_test_rows(total_test_rows),
current_test_rows(0),
current_average_time(0),
progress(0.0),
remaining_time_estimation(0.0)
{};


void lbm::benchmark::Progress::advance(double time)
{
    current_test_rows++;
    progress = 100 * (static_cast<double>(current_test_rows) / static_cast<double>(total_test_rows));
    current_average_time = ((current_test_rows - 1) * current_average_time + time) / current_test_rows;
    remaining_time_estimation = static_cast<double>(total_test_rows - current_test_rows) * current_average_time;
}

// BENCHMARK ////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<size_t> lbm::benchmark::Benchmark::get_work_group_sizes(size_t max_work_group_size)
{
    std::vector<size_t> work_group_sizes;

    for (size_t i = 4; i >= 1; --i)
    {
        size_t region = i * max_work_group_size / 4;
        bool has_stripe_height_2 = false;
        bool has_stripe_height_1 = false;

        for (size_t j = 0; j < 4; ++j)
        {
            size_t size = region - j;

            if (core::power_functions::which_power_of_4(size) || core::power_functions::which_power_of_2(size)) 
            { work_group_sizes.push_back(size); }
            else if (!(size % 2) && !has_stripe_height_2)
            {
                work_group_sizes.push_back(size);
                has_stripe_height_2 = true;
            }
            else if ((size % 2) && !has_stripe_height_1)
            {
                work_group_sizes.push_back(size);
                has_stripe_height_1 = true;
            }
        }
    }

    return work_group_sizes;
}


lbm::benchmark::Benchmark::Benchmark(std::shared_ptr<execution::SYCLAlgorithmHandler> &sycl_algorithm_handler)
:
sycl_algorithm_handler(sycl_algorithm_handler),
properties(std::make_unique<core::Properties>(
    file_interaction::json_to_properties("../settings/settings.json", -2))
),
runtime_timer(std::make_unique<core::Timer>()),
benchmark_timer(std::make_unique<core::Timer>()),
work_group_sizes(
    std::make_unique<std::vector<size_t>>(
        get_work_group_sizes(
            sycl_algorithm_handler->queue->get_device().get_info<sycl::info::device::max_work_group_size>()
        )
    )
),
device_name(sycl_algorithm_handler->queue->get_device().get_info<sycl::info::device::name>()),
optimal_data_layouts(std::make_unique<std::map<std::string, std::string>>()),
optimal_work_group_sizes(std::make_unique<std::map<std::string, unsigned int>>()),
optimal_total_times(std::make_unique<std::map<std::string, double>>())
{
    // Ensure correctness
    properties->debug_mode = false;
    properties->frame_update_interval = properties->time_steps + 1;
    properties->scenario = "Hagen-Poiseuille";

    // Read benchmark settings
    std::ifstream benchmark_settings_file("../settings/benchmark.json");
    nlohmann::json benchmark_settings = nlohmann::json::parse(benchmark_settings_file);
    benchmark_settings_file.close();
    
    // Set algorithms, data layouts and number of test runs
    algorithms = 
    std::make_unique<std::vector<std::string>>(benchmark_settings.at("algorithms").get<std::vector<std::string>>());
    test_linear_two_lattice = benchmark_settings.at("testLinearTwoLattice").get<bool>();

    data_layouts = 
    std::make_unique<std::vector<std::string>>(benchmark_settings.at("dataLayouts").get<std::vector<std::string>>());

    test_runs = benchmark_settings.at("testRunsPerConfig").get<unsigned int>();

    phase_2_scenarios =
    std::make_unique<std::vector<std::string>>(
        benchmark_settings.at("phaseTwoScenarios").get<std::vector<std::string>>()
    );

    for (const auto& algorithm : *algorithms)
    {
        (*optimal_data_layouts)[algorithm] = "stream";
        (*optimal_work_group_sizes)[algorithm] = 1;
        (*optimal_total_times)[algorithm] = std::numeric_limits<double>::max();
    }

    if(test_linear_two_lattice)
    {
        (*optimal_data_layouts)["gpu-two-lattice-linear"] = "stream";
        (*optimal_work_group_sizes)["gpu-two-lattice-linear"] = 1;
        (*optimal_total_times)["gpu-two-lattice-linear"] = std::numeric_limits<double>::max();
    }
}

// PHASES 0 AND 1 /////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json lbm::benchmark::Benchmark::prepare_file_data_phase_0_and_1(const std::string &algorithm)
{
    nlohmann::json file_data;
    file_data["device"] = device_name;
    file_data["algorithm"] = algorithm;
    file_data["verticalNodes"] = properties->vertical_nodes;
    file_data["horizontalNodes"] = properties->horizontal_nodes;

    // Prepare file
    for (const auto& data_layout : *data_layouts)
    {
        file_data[data_layout] = nlohmann::json::array();

        for(int i = 0; i < work_group_sizes->size(); ++i)
        {
            size_t work_group_size =  (*work_group_sizes)[i];

            file_data[data_layout].push_back(nlohmann::json::object());

            file_data[data_layout][i]["workGroupSize"] = work_group_size;
            file_data[data_layout][i]["initializationTimes"] = nlohmann::json::array();
            file_data[data_layout][i]["runtimes"] = nlohmann::json::array();
        }  
    }

    return file_data;
}


nlohmann::json lbm::benchmark::Benchmark::prepare_file_data_phase_0_and_1_tl_linear()
{
    nlohmann::json file_data;
    file_data["device"] = device_name;
    file_data["algorithm"] = "gpu-two-lattice-linear";
    file_data["verticalNodes"] = properties->vertical_nodes;
    file_data["horizontalNodes"] = properties->horizontal_nodes;

    // Prepare file
    for (const auto& data_layout : *data_layouts)
    {
        file_data[data_layout] = nlohmann::json::array();

        std::string work_group_size = "auto";

        file_data[data_layout].push_back(nlohmann::json::object());

        file_data[data_layout][0]["workGroupSize"] = work_group_size;
        file_data[data_layout][0]["initializationTimes"] = nlohmann::json::array();
        file_data[data_layout][0]["runtimes"] = nlohmann::json::array();
    }

    return file_data;
}


void lbm::benchmark::Benchmark::phase_0_and_1_inner_loop
(
    const int phase,
    const size_t work_group_size_index, 
    const std::string &ofdir, 
    nlohmann::json &file_data, 
    Progress &progress
)
{
    double runtime = 0;
    double average_total_time = 0;

    // Data layout loop //
    for (const auto& data_layout : *data_layouts)
    {
        std::vector<double> total_times;
        properties->data_layout = data_layout;

        // Create properties JSON file //
        lbm::file_interaction::properties_to_json(*properties);

        // Test run loop //
        for (int test_run = 0; test_run < test_runs; ++test_run)
        {
            benchmark_timer->restart();
            average_total_time = 0;

            std::ofstream file;
            file.open(ofdir, std::ofstream::out);

            sycl_algorithm_handler->initialize();

            file_data[data_layout][work_group_size_index]["initializationTimes"].push_back(
                sycl_algorithm_handler->initialization_time
            );

            runtime_timer->restart();
            sycl_algorithm_handler->start();
            sycl_algorithm_handler->block_until_finished();
            runtime = runtime_timer->elapsed();
            file_data[data_layout][work_group_size_index]["runtimes"].push_back(runtime);

            progress.advance(benchmark_timer->elapsed());
            
            std::cout << "Progress: " 
                    << progress.current_test_rows << " / " << progress.total_test_rows 
                    << " (" << std::floor(progress.progress) << "%)" << ", remaining time estimation: "
                    << seconds_to_time_format(static_cast<unsigned int>(progress.remaining_time_estimation)) 
                    << " \n";

            file << std::setw(4) << file_data;
            file.close();

            total_times.push_back(sycl_algorithm_handler->initialization_time + runtime);
        }

        average_total_time = std::accumulate(total_times.begin(), total_times.end(), 0.0) / test_runs;

        if(phase == 1 && (average_total_time < (*optimal_total_times)[properties->algorithm]))
        {
            (*optimal_data_layouts)[properties->algorithm] = properties->data_layout;
            (*optimal_work_group_sizes)[properties->algorithm] = properties->work_group_size;
            (*optimal_total_times)[properties->algorithm] = average_total_time;
        }
    }
}


void lbm::benchmark::Benchmark::phase_0_and_1_helper(int phase)
{
    std::cout << "====================================================================================================\n";
    std::cout << "Starting benchmark phase " << phase << ".\n\n";

    unsigned int total_test_cases = algorithms->size() * data_layouts->size() * work_group_sizes->size() * test_runs;
    if(test_linear_two_lattice)
    {
        total_test_cases += data_layouts->size() * test_runs;
    }

    Progress progress_phase(total_test_cases);

    for (const auto& algorithm : *algorithms)
    {
        properties->algorithm = algorithm;

        // Create JSON file for algorithm
        std::string ofdir = 
            "../benchmarks/phase" 
            + std::to_string(phase) 
            + "/results_phase" 
            + std::to_string(phase) 
            + "_" + std::string{algorithm} 
            + "_" 
            + device_name 
            + ".json";

        nlohmann::json file_data = prepare_file_data_phase_0_and_1(algorithm);

        // Work-group size loop ///////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < work_group_sizes->size(); ++j)
        {
            size_t work_group_size = (*work_group_sizes)[j];
            properties->work_group_size = work_group_size;

            phase_0_and_1_inner_loop(phase, j, ofdir, file_data, progress_phase);
        }
    }

    if(test_linear_two_lattice)
    {
        properties->algorithm = "gpu-two-lattice-linear";
        std::string ofdir = 
            "../benchmarks/phase" 
            + std::to_string(phase) 
            + "/results_phase" 
            + std::to_string(phase) 
            + "_gpu-two-lattice-linear_" 
            + device_name 
            + ".json";

        nlohmann::json file_data = prepare_file_data_phase_0_and_1_tl_linear();
        phase_0_and_1_inner_loop(phase, 0, ofdir, file_data, progress_phase);
    }

    std::cout << "\nSuccessfully completed phase " << phase << " of benchmark\n";
    std::cout << "====================================================================================================\n\n";
}


void lbm::benchmark::Benchmark::phase_0()
{
    unsigned int original_time_steps = properties->time_steps;
    properties->time_steps = 10;
    phase_0_and_1_helper(0);
    properties->time_steps = original_time_steps;
}


void lbm::benchmark::Benchmark::phase_1()
{
    phase_0_and_1_helper(1);

    std::cout << "\n";

    for(const auto& algorithm : *algorithms)
    {
        std::cout << "Optimal setup for algorithm " << algorithm << ":\n";
        std::cout << "\tData layout: " << (*optimal_data_layouts)[algorithm] << "\n";
        std::cout << "\tWork-group size: " << (*optimal_work_group_sizes)[algorithm] << "\n";
        std::cout << "\tAverage total time: " << (*optimal_total_times)[algorithm]  << "\n";
    }

    if(test_linear_two_lattice)
    {
        std::cout << "Optimal setup for algorithm gpu-two-lattice-linear:\n";
        std::cout << "\tData layout: " << (*optimal_data_layouts)["gpu-two-lattice-linear"] << "\n";
        std::cout << "\tAverage total time: " << (*optimal_total_times)["gpu-two-lattice-linear"]  << "\n";
    }

    std::cout << "\n";
}

// PHASE 2 ////////////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json lbm::benchmark::Benchmark::prepare_file_data_phase_2()
{
    nlohmann::json file_data;
    file_data["device"] = device_name;
    file_data["algorithm"] = properties->algorithm;
    file_data["dataLayout"] = properties->data_layout;
    file_data["workGroupSize"] = properties->work_group_size;
    file_data["verticalNodes"] = properties->vertical_nodes;
    file_data["horizontalNodes"] = properties->horizontal_nodes;
    file_data["scenario"] = properties->scenario;
    file_data["initializationTimes"] = nlohmann::json::array();
    file_data["runtimes"] = nlohmann::json::array();

    return file_data;    
}


nlohmann::json lbm::benchmark::Benchmark::prepare_file_data_phase_2_tl_linear()
{
    nlohmann::json file_data;
    file_data["device"] = device_name;
    file_data["algorithm"] = properties->algorithm;
    file_data["dataLayout"] = properties->data_layout;
    file_data["workGroupSize"] = "auto";
    file_data["verticalNodes"] = properties->vertical_nodes;
    file_data["horizontalNodes"] = properties->horizontal_nodes;
    file_data["scenario"] = properties->scenario;
    file_data["initializationTimes"] = nlohmann::json::array();
    file_data["runtimes"] = nlohmann::json::array();

    return file_data;    
}


void lbm::benchmark::Benchmark::phase_2_inner_loop
(
    const std::string &ofdir, 
    nlohmann::json &file_data, 
    Progress &progress
)
{
    double runtime = 0;

    // Test run loop //
    for (int test_run = 0; test_run < test_runs; ++test_run)
    {
        benchmark_timer->restart();

        std::ofstream file;
        file.open(ofdir, std::ofstream::out);

        sycl_algorithm_handler->initialize();

        file_data["initializationTimes"].push_back(
            sycl_algorithm_handler->initialization_time
        );

        runtime_timer->restart();
        sycl_algorithm_handler->start();
        sycl_algorithm_handler->block_until_finished();
        runtime = runtime_timer->elapsed();
        file_data["runtimes"].push_back(runtime);

        progress.advance(benchmark_timer->elapsed());
        
        std::cout << "Progress: " 
                << progress.current_test_rows << " / " << progress.total_test_rows 
                << " (" << std::floor(progress.progress) << "%)" << ", remaining time estimation: "
                << seconds_to_time_format(static_cast<unsigned int>(progress.remaining_time_estimation)) 
                << " \n";

        file << std::setw(4) << file_data;
        file.close();
    }
}


void lbm::benchmark::Benchmark::phase_2()
{
    std::cout << "====================================================================================================\n";
    std::cout << "Starting benchmark phase 2.\n\n";

    unsigned int total_test_cases = algorithms->size() * phase_2_scenarios->size() * test_runs;
    if(test_linear_two_lattice)
    {
        total_test_cases += phase_2_scenarios->size() * test_runs;
    }

    Progress progress_phase(total_test_cases);

    for (const auto& algorithm : *algorithms)
    {
        properties->algorithm = algorithm;
        properties->data_layout = (*optimal_data_layouts)[algorithm];
        properties->work_group_size = (*optimal_work_group_sizes)[algorithm];

        for (const auto& scenario : *phase_2_scenarios)
        {
            properties->scenario = scenario;
            
            // Create properties JSON file //
            lbm::file_interaction::properties_to_json(*properties);

            // Create JSON file for algorithm
            std::string ofdir = 
                "../benchmarks/phase2/phase2_" 
                + std::string{algorithm} 
                + "_" 
                + properties->scenario 
                + "_" 
                + device_name 
                + ".json";
            nlohmann::json file_data = prepare_file_data_phase_2();

            phase_2_inner_loop(ofdir, file_data, progress_phase);
        }
    }

    if(test_linear_two_lattice)
    {
        properties->algorithm = "gpu-two-lattice-linear";
        properties->data_layout = (*optimal_data_layouts)["gpu-two-lattice-linear"];
        properties->work_group_size = (*optimal_work_group_sizes)["gpu-two-lattice-linear"];

        for (const auto& scenario : *phase_2_scenarios)
        {
            properties->scenario = scenario;

            // Create properties JSON file //
            lbm::file_interaction::properties_to_json(*properties);

            // Create JSON file for algorithm
            std::string ofdir = 
                "../benchmarks/phase2/phase2_gpu-two-lattice-linear_" 
                + properties->scenario 
                + "_" 
                + device_name 
                + ".json";
            nlohmann::json file_data = prepare_file_data_phase_2_tl_linear();

            phase_2_inner_loop(ofdir, file_data, progress_phase);
        }
    }

    std::cout << "\nSuccessfully completed phase 2 of benchmark\n";
    std::cout << "====================================================================================================\n\n";
}