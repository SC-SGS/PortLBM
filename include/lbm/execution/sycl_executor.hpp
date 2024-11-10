/**
 * @file    hpx_executor.hpp
 * 
 * @author  Marcel Graf
 * 
 * @brief   This header file contains the declaration of a class for an HPX executor.
 *          Through this executor, it is possible for the GUI to launch simulation iterations
 *          and to retrieve results.
 * 
 * @version 1.0
 * 
 * @date    2024-08-28
 * 
 * @copyright Copyright (c) 2024 Marcel Graf
 */

#ifndef SYCL_EXECUTOR_HPP
#define SYCL_EXECUTOR_HPP

#include "executor.hpp"
#include "lbm_gui.hpp"

// #include "../include/defines.hpp"

// #include <hpx/future.hpp>

// // Forward declarations
// struct GuiSimulationData;

// /**
//  * @brief Executor class for an HPX backend.
//  */
// class HpxExecutor : public Executor<GuiSimulationData, sim_data_tuple>
// {
//     public:
//         /**
//          * @brief Construct a new HPX executor and initializes its future such that it stores
//          *        the current simulation data present by default.
//          * 
//          * @param gui_simulation_data a pointer to a struct containing the simulation data
//          */
//         HpxExecutor(GuiSimulationData &gui_simulation_data);

//         /**
//          * @brief Launches an HPX task to start a new iteration of the simulation.
//          *        The results are stored to the specified struct.
//          * 
//          * @param gui_simulation_data reference to a struct containing the relevant data
//          */
//         void execute(GuiSimulationData &gui_simulation_data) override;

//         /**
//          * @brief Returns whether or not the future of this executor is ready.
//          * 
//          */
//         bool is_ready() const override;

//         /**
//          * @brief Retrieves the simulation data tuple stored within the future of this executor.
//          *        Careful: Once retrieved, the future is no longer ready and does not store its old value anymore.
//          * 
//          */
//         std::unique_ptr<sim_data_tuple> get() override;

//         /**
//          * @brief Initializes the future of this executor using the specified data.
//          * 
//          * @param data a reference to the simulation data tuple that will be the content of the future
//          */
//         void initialize(GuiSimulationData &gui_simulation_data) override;

//     private:
//         /**
//          * @brief This future holds the simulation results after each iteration.
//          * 
//          */
//         std::unique_ptr<hpx::future<sim_data_tuple>> sim_data_;
// };

#endif