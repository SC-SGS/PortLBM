/**
 * @file    hpx_executor.cpp
 * 
 * @author  Marcel Graf
 * 
 * @brief   Definition of the HPX executor declared in hpx_executor.hpp.
 * 
 * @version 1.0
 * 
 * @date    2024-08-28
 * 
 * @copyright Copyright (c) 2024 Marcel Graf
 */

#include "../../../include/lbm/execution/sycl_executor.hpp"

// HpxExecutor::HpxExecutor(GuiSimulationData &gui_simulation_data) : 
// sim_data_(std::make_unique<hpx::future<sim_data_tuple>>(hpx::async([&]{return gui_simulation_data.current_result;}))) {};

// void HpxExecutor::execute(GuiSimulationData &gui_simulation_data)
// {
//     sim_data_ = std::make_unique<hpx::future<sim_data_tuple>>(
//         hpx::async([&]
//         {
//             return parallel_two_lattice::run_new(
//             gui_simulation_data.fluid_nodes, 
//             gui_simulation_data.border_swap_information, 
//             gui_simulation_data.distribution_values_0, 
//             gui_simulation_data.distribution_values_1, 
//             gui_simulation_data.access_function);
//         }
//         ));
// }

// bool HpxExecutor::is_ready() const
// {
//     return sim_data_->is_ready();
// }

// std::unique_ptr<sim_data_tuple> HpxExecutor::get()
// {
//     return std::make_unique<sim_data_tuple>(sim_data_->get());
// }

// void HpxExecutor::initialize(GuiSimulationData &gui_simulation_data)
// {
//     gui_simulation_data.distribution_values_0 = {};
//     gui_simulation_data.nodes = {};
//     gui_simulation_data.fluid_nodes = {};
//     gui_simulation_data.phase_information = {};
//     gui_simulation_data.border_swap_information = {};

//     setup_example_domain
//     (
//         gui_simulation_data.distribution_values_0, 
//         gui_simulation_data.nodes, 
//         gui_simulation_data.fluid_nodes, 
//         gui_simulation_data.phase_information, 
//         gui_simulation_data.access_function,
//         false
//     );

//     gui_simulation_data.border_swap_information = 
//         bounce_back::retrieve_border_swap_info(gui_simulation_data.fluid_nodes, gui_simulation_data.phase_information);
//     gui_simulation_data.distribution_values_1 = gui_simulation_data.distribution_values_0;

//     gui_simulation_data.current_result = std::make_tuple(std::vector<velocity>(TOTAL_NODE_COUNT, {0,0}), std::vector<double>(TOTAL_NODE_COUNT,0));
//     sim_data_ = std::make_unique<hpx::future<sim_data_tuple>>(hpx::async([&]{return gui_simulation_data.current_result;}));

//     gui_simulation_data.absolute_velocities->assign(TOTAL_NODE_COUNT, 0);
//     gui_simulation_data.x_velocities->assign(TOTAL_NODE_COUNT, 0);
//     gui_simulation_data.y_velocities->assign(TOTAL_NODE_COUNT, 0);
// }
