/**
 * @file        lbm_gui.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This file contains implementations for GUI operations declared in lbm_gui.hpp.
 *              
 * @version     2.0
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/gui/lbm_gui.hpp"
#include <nlohmann/json.hpp>

lbm::gui::Windows::Windows() 
:
show_properties(true),
show_simulation_status(true),
show_read_from_file_window(false),
show_density(true),
show_velocity(true),
enable_live_visualization(true),
enable_velocity_quiver(false),
menu_bar_size(0.0f)
{};

lbm::gui::SimulationControl::SimulationControl() 
:
is_paused(false),
is_simulation_active(false),
results_to_csv(false),
result_file_name("results.csv")
{};

lbm::gui::Progress::Progress() 
:
current_iter(0),
progress(0.0),
framerate(0.0),
frametime(0)
{};

lbm::gui::Monitor::Monitor() 
:
monitor_x_scale(1),
monitor_y_scale(1),
display_width(1),
display_height(1)
{};

lbm::gui::Colormaps::Colormaps() 
:
density_colormap(ImPlotColormap_Viridis),
density_colormap_lower_scale(0.0f),
density_colormap_upper_scale(0.0f),
velocity_colormap(ImPlotColormap_Hot),
velocity_colormap_lower_scale(0.0),
velocity_colormap_upper_scale(0.0f)
{};

lbm::gui::VelocityQuiverData::VelocityQuiverData(size_t size) 
:
x_values{std::make_unique<std::vector<double>>(size)},
y_values{std::make_unique<std::vector<double>>(size)}
{};

lbm::gui::PropertiesBuffer::PropertiesBuffer(const core::Properties &properties)
:
algorithm(properties.algorithm),
data_layout(properties.data_layout),
time_steps(properties.time_steps),
relaxation_time(properties.relaxation_time),
debug_mode(properties.debug_mode),
results_to_csv(properties.results_to_csv),
vertical_nodes(properties.vertical_nodes - 2),
horizontal_nodes(properties.horizontal_nodes - 2),
inlet_velocity_x(properties.inlet_velocity_x),
inlet_velocity_y(properties.inlet_velocity_y),
inlet_density(properties.inlet_density),
outlet_velocity_x(properties.outlet_velocity_x),
outlet_velocity_y(properties.outlet_velocity_y),
outlet_density(properties.outlet_density),
changed(false)
{};

void lbm::gui::PropertiesBuffer::to_json(const std::string &path)
{
    nlohmann::json file_data;

    file_data["algorithmic"]["debugMode"] = debug_mode;
    file_data["algorithmic"]["resultsToCsv"] = results_to_csv;
    file_data["algorithmic"]["relaxationTime"] = relaxation_time;
    file_data["algorithmic"]["timeSteps"] = time_steps;
    file_data["algorithmic"]["dataLayout"] = data_layout;
    file_data["algorithmic"]["algorithm"] = algorithm;

    file_data["domain"]["verticalNodes"] = vertical_nodes;
    file_data["domain"]["horizontalNodes"]= horizontal_nodes;

    file_data["inlets"]["velocity"]["x"] = inlet_velocity_x;
    file_data["inlets"]["velocity"]["y"] = inlet_velocity_y;
    file_data["inlets"]["density"] = inlet_density;

    file_data["outlets"]["velocity"]["x"] = outlet_velocity_x;
    file_data["outlets"]["velocity"]["y"] = outlet_velocity_y;
    file_data["outlets"]["density"] = outlet_density;

    std::ofstream file(path);
    file << std::setw(4) << file_data;
    file.close();
}

void lbm::gui::windows::read_from_file_window
(
    const Monitor &gui_monitor, 
    Windows &window_settings
)
{
    if(window_settings.show_read_from_file_window)
    {
        if(ImGui::Begin("Read from file", &window_settings.show_read_from_file_window))
        {
            ImGui::Text("This feature may be available in the future.");
            ImGui::PushItemWidth(ImGui::GetWindowWidth() / 2);
            if (ImGui::Button("Load", ImVec2(ImGui::GetWindowSize().x * 0.5f, 0)))
            {

            }
        }     
        ImGui::End();
    }
}

void lbm::gui::windows::density_window
(
    const core::Properties &properties,
    const Monitor &gui_monitor,
    const SimulationControl &gui_simulation_control,
    const core::SimulationResults &simulation_results,
    const Progress &gui_progress,
    Windows &windows, 
    Colormaps &gui_colormaps
)
{
    if(windows.show_density)
    {
        ImGui::SetNextWindowPos
        (
            ImVec2
            (
                (1.0 / 4) * gui_monitor.viewport_work_size.x, 
                1.0 / 2 * windows.menu_bar_size + 1.0 / 2 * gui_monitor.viewport_work_size.y
            ), 
            ImGuiCond_Appearing
        ); 

        ImGui::SetNextWindowSize
        (
            ImVec2
            (
                (3.0 / 4) * gui_monitor.viewport_work_size.x, 
                1.0 / 2 * gui_monitor.viewport_work_size.y - 1.0 / 2 * windows.menu_bar_size
            ), 
            ImGuiCond_Appearing
        );

        if(ImGui::Begin("Density", &windows.show_density))
        {
            ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
            items::colormap_picker(gui_colormaps.density_colormap, "Density");
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20, 0));
            ImGui::SameLine();
            ImGui::InputDouble("Lower scale", &gui_colormaps.density_colormap_lower_scale, 0.05, 0.1);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20, 0));
            ImGui::SameLine();
            ImGui::InputDouble("Upper scale", &gui_colormaps.density_colormap_upper_scale, 0.05, 0.1);
            ImPlot::PushColormap(gui_colormaps.density_colormap);

            if
            (
                ImPlot::BeginPlot
                (
                    "Density",
                    ImVec2
                    (
                        ImGui::GetContentRegionAvail().x - gui_monitor.monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
                        ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                    ),
                    ImPlotFlags_NoLegend|ImPlotFlags_NoMouseText|ImPlotFlags_Crosshairs
                )
            ) 
            {
                ImPlot::SetupAxes
                (
                    nullptr, 
                    nullptr, 
                    ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks, 
                    ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks
                );
                ImPlot::PlotHeatmap
                (
                    "",
                    simulation_results.densities->data(),
                    properties.vertical_nodes - 2,
                    properties.horizontal_nodes - 2,
                    gui_colormaps.density_colormap_lower_scale,
                    gui_colormaps.density_colormap_upper_scale,
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(properties.horizontal_nodes - 2, properties.vertical_nodes - 2)
                );

                ImPlot::PushColormap("SOLID_MASK");
                ImPlot::PlotHeatmap
                (
                    "Solid mask for density",
                    simulation_results.densities->data(),
                    properties.vertical_nodes - 2,
                    properties.horizontal_nodes - 2,
                    -1,
                    1, 
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(properties.horizontal_nodes - 2, properties.vertical_nodes - 2)
                );

                if (ImPlot::IsPlotHovered()) 
                {
                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();

                    if(0 <= mouse.x && mouse.x < (properties.horizontal_nodes - 2) && 0 <= mouse.y && mouse.y < (properties.vertical_nodes - 2))
                    {
                        ImGui::BeginTooltip();
                        ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);
                        double value = simulation_results.densities->at((properties.horizontal_nodes - 2) * ((int)floor(mouse.y)) + (int)floor(mouse.x));
                        if(value != -1)
                        {
                            ImGui::Text("Density: %f", value);
                        }
                        else
                        {
                            ImGui::Text("Solid node");
                        }
                        ImGui::EndTooltip();
                    }
                }

                ImPlot::EndPlot();
            }
            ImGui::SameLine();
            ImPlot::ColormapScale
            (
                "Density",
                gui_colormaps.density_colormap_lower_scale,
                gui_colormaps.density_colormap_upper_scale, 
                ImVec2
                (
                    gui_monitor.monitor_x_scale * 100,
                    ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                ), 
                "%g", 
                ImPlotColormapScaleFlags_None, 
                gui_colormaps.density_colormap
            );   
        }
        ImGui::End();
    }
}

void lbm::gui::windows::velocity_window
(
    const core::Properties &properties,
    const Monitor &gui_monitor,
    const SimulationControl &gui_simulation_control,
    const core::SimulationResults &simulation_results,
    const Progress &gui_progress,
    Windows &windows, 
    VelocityQuiverData &gui_velocity_quiver_data,
    Colormaps &gui_colormaps
)
{
    if(windows.show_velocity)
    {
        ImGui::SetNextWindowPos
        (
            ImVec2
            (
                (1.0 / 4) * gui_monitor.viewport_work_size.x, 
                windows.menu_bar_size
            ), 
            ImGuiCond_Appearing
        ); 

        ImGui::SetNextWindowSize
        (
            ImVec2
            (
                (3.0 / 4) * gui_monitor.viewport_work_size.x,
                (1.0 / 2) * gui_monitor.viewport_work_size.y - 1.0 / 2 * windows.menu_bar_size
            ), 
            ImGuiCond_Appearing
        );

        if(ImGui::Begin("Velocity", &windows.show_velocity))
        {
            ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
            items::colormap_picker(gui_colormaps.velocity_colormap, "Velocity");

            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
            ImGui::SameLine();
            ImGui::InputDouble("Lower scale", &gui_colormaps.velocity_colormap_lower_scale, 0.01, 0.1);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
            ImGui::SameLine();
            ImGui::InputDouble("Upper scale", &gui_colormaps.velocity_colormap_upper_scale, 0.01, 0.1);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
            ImGui::SameLine();
            if(ImGui::Checkbox("Enable vector plot", &windows.enable_velocity_quiver)){}


            ImPlot::PushColormap(gui_colormaps.velocity_colormap);

            if
            (
                ImPlot::BeginPlot
                (
                    "Velocity",
                    ImVec2
                    (
                        ImGui::GetContentRegionAvail().x - gui_monitor.monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
                        ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                    ),
                    ImPlotFlags_NoLegend|ImPlotFlags_NoMouseText|ImPlotFlags_Crosshairs
                )
            ) 
            {
                ImPlot::SetupAxes
                (
                    nullptr, 
                    nullptr, 
                    ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks, 
                    ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks
                );

                ImPlot::PlotHeatmap
                (
                    "Absolute velocities",
                    simulation_results.absolute_velocities->data(),
                    properties.vertical_nodes - 2,
                    properties.horizontal_nodes - 2,
                    gui_colormaps.velocity_colormap_lower_scale, 
                    gui_colormaps.velocity_colormap_upper_scale, 
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(properties.horizontal_nodes - 2, properties.vertical_nodes - 2)
                );

                ImPlot::PopColormap;
                ImPlot::PushColormap("SOLID_MASK");

                ImPlot::PlotHeatmap
                (
                    "Solid mask for velocity",
                    simulation_results.densities->data(),
                    properties.vertical_nodes - 2,
                    properties.horizontal_nodes - 2,
                    -1,
                    1, 
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(properties.horizontal_nodes - 2, properties.vertical_nodes - 2)
                );

                ImPlot::PopColormap;
                ImPlot::PushColormap(ImPlotColormap_Greys);
                ImPlot::SetNextLineStyle(ImVec4(0, 0, 0, 0.5), 3.f);

                if(windows.enable_velocity_quiver)
                {
                    ImPlot::PlotLine
                    (
                        "Quiver plot", 
                        gui_velocity_quiver_data.x_values->data(), 
                        gui_velocity_quiver_data.y_values->data(), 
                        2 * simulation_results.absolute_velocities->size(), 
                        ImPlotLineFlags_Segments
                    );
                }

                if(ImPlot::IsPlotHovered()) 
                {
                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();

                    if(0 <= mouse.x && mouse.x < (properties.horizontal_nodes - 2) && 0 <= mouse.y && mouse.y < (properties.vertical_nodes - 2))
                    {
                        ImGui::BeginTooltip();
                        ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);

                        if(simulation_results.densities->at((properties.horizontal_nodes - 2) * (int)floor(mouse.y) + (int)floor(mouse.x)) != -1)
                        {
                            ImGui::Text("x velocity: %.6f", simulation_results.x_velocities->at((properties.horizontal_nodes - 2) * ((int)floor(mouse.y)) + (int)floor(mouse.x)));
                            ImGui::Text("y velocity: %.6f", simulation_results.y_velocities->at((properties.horizontal_nodes - 2) * ((int)floor(mouse.y)) + (int)floor(mouse.x)));
                        }
                        else
                        {
                            ImGui::Text("Solid node"); 
                        }
                        ImGui::EndTooltip();
                    }
                }

                ImPlot::EndPlot();
            }
            ImGui::SameLine();
            ImPlot::ColormapScale
            (
                "Velocity",
                gui_colormaps.velocity_colormap_lower_scale, 
                gui_colormaps.velocity_colormap_upper_scale, 
                ImVec2
                (
                    gui_monitor.monitor_x_scale * 100,
                    ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                ), 
                "%g", 
                ImPlotColormapScaleFlags_None, 
                gui_colormaps.velocity_colormap
            );   
        }
        ImGui::End();
    }
}
