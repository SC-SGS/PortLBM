/**
 * @brief       This header file contains the declaration of benchmark functionality.
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 *              Copyright (c) 2026 Alexander Strack
 */

#ifndef LBM_BENCHMARK_HPP
#define LBM_BENCHMARK_HPP

#include "../execution/sycl_algorithm_handler.hpp"

// LBM core functionality
#include "../core/constants.hpp"
#include "../core/simulation.hpp"
#include "../core/timer.hpp"

// LBM file interaction
#include "../file_interaction/file_interaction.hpp"

// Standard library
#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace lbm
{

namespace benchmark
{
/**
 * @brief   Creates a string from the specified number of seconds where it is recalculated into days, hours,
 *          minutes and seconds.
 *
 * @param[in]   seconds the total number of seconds
 *
 * @return  a string representation of the total amount of seconds
 */
inline std::string seconds_to_time_format(unsigned int seconds)
{
    unsigned int days = seconds / 86'400;
    seconds -= days * 86'400;

    unsigned int hours = seconds / 3600;
    seconds -= hours * 3600;

    unsigned int minutes = seconds / 60;
    seconds -= minutes * 60;

    return std::to_string(days) + "d, " + std::to_string(days) + "h, " + std::to_string(minutes) + "m, "
           + std::to_string(seconds) + "s";
}

/**
 * @brief   This structure is used to track the progress of the benchmark phases.
 */
struct Progress
{
    int total_test_rows;
    int current_test_rows;

    double current_average_time;
    double progress;
    double remaining_time_estimation;

    explicit Progress(const unsigned int total_test_rows);

    /**
     * @brief   Increments the number of completed test rows by one and updates the current average time
     *          and the estimation for the remaining time.
     *
     * @param[in]   time    the last benchmark test row took this many seconds
     */
    void advance(double time);
};

class Benchmark
{
  private:
    std::shared_ptr<execution::SYCLAlgorithmHandler> sycl_algorithm_handler;

    std::string device_name;

    std::unique_ptr<core::Properties> properties;
    std::unique_ptr<core::Timer> runtime_timer;
    std::unique_ptr<core::Timer> benchmark_timer;

    std::unique_ptr<std::vector<size_t>> work_group_sizes;
    std::unique_ptr<std::vector<std::string>> algorithms;
    std::unique_ptr<std::vector<std::string>> data_layouts;
    std::unique_ptr<std::vector<std::string>> phase_2_scenarios;

    std::unique_ptr<std::map<std::string, std::string>> optimal_data_layouts;
    std::unique_ptr<std::map<std::string, unsigned int>> optimal_work_group_sizes;
    std::unique_ptr<std::map<std::string, double>> optimal_total_times;

    unsigned int test_runs;

    bool test_linear_two_lattice;

    /**
     * @brief   Determines a vector of work-group sizes to test based on the maximum work-group size. In steps
     *          of quarters of the maximum work-group size, the method attempts to find a suitable work-group
     *          size for as many domain shapes as possible.
     *
     * @param[in]   max_work_group_size the maximum work-group size, as returned by a SYCL queue
     *
     * @return  a vector containing various work-group sizes that are tested
     */
    std::vector<size_t> get_work_group_sizes(size_t max_work_group_size);

    void phase_0_and_1_inner_loop(const int phase,
                                  const size_t work_group_size_index,
                                  const std::string &ofdir,
                                  nlohmann::json &file_data,
                                  Progress &progress);

    void phase_0_and_1_helper(int phase);

    nlohmann::json prepare_file_data_phase_0_and_1(const std::string &algorithm);

    nlohmann::json prepare_file_data_phase_0_and_1_tl_linear();

    nlohmann::json prepare_file_data_phase_2();

    nlohmann::json prepare_file_data_phase_2_tl_linear();

    void phase_2_inner_loop(const std::string &ofdir, nlohmann::json &file_data, Progress &progress);

  public:
    explicit Benchmark(std::shared_ptr<execution::SYCLAlgorithmHandler> &sycl_algorithm_handler);

    /**
     * @brief   Executes phase 0 of the benchmark. Here, the JIT compiler is prepared to compile all
     *          configurations.
     */
    void phase_0();

    /**
     * @brief   Executes phase 1 of the benchmark. Here, all configurations are tested with the Hagen-
     *          Poiseuille scenario and the specified properties.
     */
    void phase_1();

    /**
     * @brief   Executes phase 2 of the benchmark. Here, the configurations that turned out to be optimal in
     *          phase 1 are tested in further specified scenarios.
     */
    void phase_2();
};

}  // namespace benchmark

}  // namespace lbm

#endif  // ! LBM_BENCHMARK_HPP
