/**
 * @brief       This file contains implementations for GUI-related structs declared in lbm_gui.hpp.
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 *              Copyright (c) 2026 Alexander Strack
 */

#include "../../../include/lbm/gui/lbm_gui.hpp"

lbm::gui::Windows::Windows(bool debug) :
    show_properties(true),
    show_simulation_status(true),
    show_framerate(true),
    show_read_from_file_window(false),
    show_density(true),
    show_velocity(true),
    enable_live_visualization(true),
    enable_velocity_quiver(false),
    show_debug_window(true),
    menu_bar_size(0.0f)
{
    node_content = debug ? "%g" : nullptr;
}

lbm::gui::SimulationControl::SimulationControl() :
    is_paused(false),
    is_simulation_active(false),
    results_to_csv(false),
    result_file_name("results.csv"){};

lbm::gui::Progress::Progress() :
    progress(0.0),
    framerate_backend(0.0),
    frametime_backend(0.0),
    framerate_frontend(0.0),
    frametime_frontend(0.0){};

lbm::gui::Colormaps::Colormaps() :
    density_colormap(ImPlotColormap_Viridis),
    density_colormap_lower_scale(0.0f),
    density_colormap_upper_scale(0.0f),
    velocity_colormap(ImPlotColormap_Jet),
    velocity_colormap_lower_scale(0.0),
    velocity_colormap_upper_scale(0.0f){};

lbm::gui::VelocityQuiverData::VelocityQuiverData(const size_t &size) :
    x_values{ std::make_unique<std::vector<float>>(size) },
    y_values{ std::make_unique<std::vector<float>>(size) } {};

#ifdef BENCHMARK_MODE

lbm::gui::FPSBenchmarkValues::FPSBenchmarkValues() :
    frontend_fps(std::make_unique<std::vector<double>>()),
    backend_fps(std::make_unique<std::vector<double>>()),
    is_free(true),
    benchmark_counter(0)
{ }

#endif
