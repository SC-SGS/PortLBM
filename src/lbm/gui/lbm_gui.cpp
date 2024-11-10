/**
 * @file        lbm_gui.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This file contains implementations for GUI operations declared in lbm_gui.hpp.
 *              
 * @version     1.0
 * 
 * @date        2024-08-28
 * 
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#include "../../../include/lbm/gui/lbm_gui.hpp"

// GuiWindows::GuiWindows() :
//     show_properties(true),
//     show_simulation_status(true),
//     show_read_from_file_window(false),
//     show_density(true),
//     show_velocity(true),
//     enable_live_visualization(true),
//     enable_velocity_quiver(false),
//     menu_bar_size(0.0f)
//     {};

// GuiSimulationControl::GuiSimulationControl() :
//     is_paused(false),
//     is_simulation_active(false),
//     results_to_csv(false),
//     result_file_name("results.csv")
//     {};

// GuiProgress::GuiProgress() :
//     current_iter(0),
//     progress(0.0),
//     framerate(0.0),
//     frametime(0)
//     {};

// GuiSimulationData::GuiSimulationData() :
//     all_results({}), 
//     current_result({}),
//     result_file_name("results.csv"),
//     results_to_csv(false),
//     distribution_values_0({}),
//     distribution_values_1({}),
//     nodes({}),
//     fluid_nodes({}),
//     phase_information({}),
//     border_swap_information({}),
//     access_function(lbm_access::collision),
//     x_velocities(std::make_shared<std::vector<double>>()),
//     y_velocities(std::make_shared<std::vector<double>>()),
//     absolute_velocities(std::make_shared<std::vector<double>>())
//     {};

// GuiMonitor::GuiMonitor() :
//     monitor_x_scale(1),
//     monitor_y_scale(1),
//     display_width(1),
//     display_height(1)
//     {};

//     // struct  GuiMonitor
//     // {
//     //     // GLFWmonitor *primary_monitor = glfwGetPrimaryMonitor();
//     //     // const GLFWvidmode *video_mode = glfwGetVideoMode(primary_monitor);
//     //     // float video_mode_width = video_mode->width;
//     //     // float video_mode_height = video_mode->height;
//     //     GLFWmonitor *primary_monitor;
//     //     const GLFWvidmode *video_mode;
//     //     float video_mode_width;
//     //     float video_mode_height;
//     //     float monitor_x_scale = 1;
//     //     float monitor_y_scale = 1;
//     //     int display_width = 1;
//     //     int display_height = 1;
//     //     // ImGuiViewport viewport = *ImGui::GetMainViewport();
//     //     // ImVec2 viewport_work_size = viewport.WorkSize;
//     //     ImGuiViewport viewport;
//     //     ImVec2 viewport_work_size;
//     // };

// GuiAlgorithmic::GuiAlgorithmic() :
//     algorithms{{"parallel_two_lattice", "parallel_two_step", "parallel_swap", "parallel_shift"}},
//     current_algorithm("parallel_two_lattice"),
//     data_layouts{{"collision", "stream", "bundle"}},
//     current_data_layout("collision")
//     {};

// GuiColormaps::GuiColormaps() :
//     density_colormap(ImPlotColormap_Viridis),
//     density_colormap_lower_scale(0.0f),
//     density_colormap_upper_scale(0.0f),
//     velocity_colormap(ImPlotColormap_Hot),
//     velocity_colormap_lower_scale(0.0),
//     velocity_colormap_upper_scale(0.0f)
//     {};

// GuiVelocityQuiverData::GuiVelocityQuiverData() :
//     x_values{std::make_unique<std::vector<double>>()},
//     y_values{std::make_unique<std::vector<double>>()}
//     {};

// GuiVisualizationData::GuiVisualizationData() :
//     vertical_nodes(0),
//     horizontal_nodes(0),
//     density_values(std::make_shared<std::vector<double>>()),
//     absolute_velocity_values(std::make_shared<std::vector<double>>()),
//     x_velocity_values(std::make_shared<std::vector<double>>()),
//     y_velocity_values(std::make_shared<std::vector<double>>())
//     {};

// void gui_windows::properties_window
// (
//     const GuiMonitor &gui_monitor,
//     Settings &settings,
//     GuiWindows &gui_windows,
//     GuiSimulationControl &gui_simulation_control,
//     GuiAlgorithmic &gui_algorithmic,
//     GuiVisualizationData &gui_visualization_data
// )
// {
//     if (gui_windows.show_properties)
//     {    
//         ImGui::SetNextWindowSize(
//             {
//                 1 * gui_monitor.viewport_work_size.x / 4, 
//                 4 * gui_monitor.viewport_work_size.y / 5
//             }
//         );

//         ImGui::SetNextWindowPos(
//             {
//                 0, 
//                 gui_windows.menu_bar_size + gui_monitor.viewport_work_size.y/5
//             }
//         );

//         if(ImGui::Begin("Properties", &gui_windows.show_properties, ImGuiWindowFlags_NoResize))
//         {
//             ImGui::PushItemWidth(ImGui::GetWindowWidth() / 2);

//             ImGui::BeginDisabled(gui_simulation_control.is_simulation_active);
            
//             gui_items::algorithm_selection(settings, gui_algorithmic);
//             gui_items::data_layout_selection(settings, gui_algorithmic);
//             gui_items::save_results_selection(settings.results_to_csv, gui_simulation_control);
//             ImGui::Checkbox("Live visualization", &gui_windows.enable_live_visualization);
//             gui_items::properties_simulation_and_domain(settings, gui_visualization_data);
//             gui_items::properties_shift_algorithm(settings);
//             gui_items::properties_fluid(settings);
 
//             ImGui::SeparatorText("");
//             if (ImGui::Button("Create configuration file", ImVec2(ImGui::GetWindowSize().x*0.5f, 0)))
//             {
//                     write_csv_config_file(settings);
//             }

//             ImGui::EndDisabled();   

//             if(gui_simulation_control.is_simulation_active)
//             {
//                 ImGui::Text("The properties of an active simulation cannot be changed.");
//             } 
//         }   
//         ImGui::End();                    
//     }
// }

// void gui_windows::read_from_file_window(const GuiMonitor &gui_monitor, GuiWindows &window_settings)
// {
//     if(window_settings.show_read_from_file_window)
//     {
//         if(ImGui::Begin("Read from file", &window_settings.show_read_from_file_window))
//         {
//             ImGui::Text("This feature may be available in the future.");
//             ImGui::PushItemWidth(ImGui::GetWindowWidth() / 2);
//             if (ImGui::Button("Load", ImVec2(ImGui::GetWindowSize().x*0.5f, 0)))
//             {

//             }
//         }     
//         ImGui::End();
//     }
// }

// void gui_windows::density_window
// (
//     const GuiMonitor &gui_monitor,
//     const GuiSimulationControl &gui_simulation_control,
//     const GuiAlgorithmic &gui_algorithmic,
//     const GuiVisualizationData &gui_visualization_data,
//     const GuiProgress &gui_progress,
//     GuiWindows &gui_windows, 
//     GuiColormaps &gui_colormaps
// )
// {
//     if(gui_windows.show_density)
//     {
//         ImGui::SetNextWindowPos
//         (
//             ImVec2
//             (
//                 (1.0/4) * gui_monitor.viewport_work_size.x, 
//                 1.0/2 * gui_windows.menu_bar_size + 1.0/2 * gui_monitor.viewport_work_size.y
//             ), 
//             ImGuiCond_Appearing
//         ); 

//         ImGui::SetNextWindowSize
//         (
//             ImVec2
//             (
//                 (3.0/4) * gui_monitor.viewport_work_size.x, 
//                 1.0/2 * gui_monitor.viewport_work_size.y - 1.0/2 * gui_windows.menu_bar_size
//             ), 
//             ImGuiCond_Appearing
//         );

//         if(ImGui::Begin("Density", &gui_windows.show_density))
//         {
//             ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
//             gui_items::colormap_picker(gui_colormaps.density_colormap, "Density");
//             ImGui::SameLine();
//             ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
//             ImGui::SameLine();
//             ImGui::InputDouble("Lower scale", &gui_colormaps.density_colormap_lower_scale, 0.05, 0.1);
//             ImGui::SameLine();
//             ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
//             ImGui::SameLine();
//             ImGui::InputDouble("Upper scale", &gui_colormaps.density_colormap_upper_scale, 0.05, 0.1);
//             ImPlot::PushColormap(gui_colormaps.density_colormap);

//             if
//             (
//                 ImPlot::BeginPlot
//                 (
//                     "Density",
//                     ImVec2
//                     (
//                         ImGui::GetContentRegionAvail().x - gui_monitor.monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
//                         ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
//                     ),
//                     ImPlotFlags_NoLegend|ImPlotFlags_NoMouseText|ImPlotFlags_Crosshairs
//                 )
//             ) 
//             {
//                 ImPlot::SetupAxes
//                 (
//                     nullptr, 
//                     nullptr, 
//                     ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks, 
//                     ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks
//                 );
//                 ImPlot::PlotHeatmap
//                 (
//                     "",
//                     gui_visualization_data.density_values->data(),
//                     gui_visualization_data.vertical_nodes,
//                     gui_visualization_data.horizontal_nodes,
//                     gui_colormaps.density_colormap_lower_scale,
//                     gui_colormaps.density_colormap_upper_scale,
//                     nullptr, 
//                     ImPlotPoint(0,0),
//                     ImPlotPoint(gui_visualization_data.horizontal_nodes, gui_visualization_data.vertical_nodes)
//                 );

//                 ImPlot::PushColormap("SOLID_MASK");
//                 ImPlot::PlotHeatmap
//                 (
//                     "Solid mask for density",
//                     gui_visualization_data.density_values->data(),
//                     gui_visualization_data.vertical_nodes,
//                     gui_visualization_data.horizontal_nodes,
//                     -1,
//                     1, 
//                     nullptr, 
//                     ImPlotPoint(0,0),
//                     ImPlotPoint(gui_visualization_data.horizontal_nodes, gui_visualization_data.vertical_nodes)
//                 );

//                 if (ImPlot::IsPlotHovered()) 
//                 {
//                     ImPlotPoint mouse = ImPlot::GetPlotMousePos();

//                     if(0 <= mouse.x && mouse.x < gui_visualization_data.horizontal_nodes && 0 <= mouse.y && mouse.y < gui_visualization_data.vertical_nodes)
//                     {
//                         ImGui::BeginTooltip();
//                         ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);
//                         double value = gui_visualization_data.density_values->at(gui_visualization_data.horizontal_nodes * ((int)floor(mouse.y)) + (int)floor(mouse.x));
//                         if(value != -1)
//                         {
//                             ImGui::Text("Density: %f", value);
//                         }
//                         else
//                         {
//                             ImGui::Text("Solid node");
//                         }
//                         ImGui::EndTooltip();
//                     }
//                 }

//                 ImPlot::EndPlot();
//             }
//             ImGui::SameLine();
//             ImPlot::ColormapScale
//             (
//                 "Density",
//                 gui_colormaps.density_colormap_lower_scale,
//                 gui_colormaps.density_colormap_upper_scale, 
//                 ImVec2
//                 (
//                     gui_monitor.monitor_x_scale * 100,
//                     ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
//                 ), 
//                 "%g", 
//                 ImPlotColormapScaleFlags_None, 
//                 gui_colormaps.density_colormap
//             );   
//         }
//         ImGui::End();
//     }
// }

// void gui_windows::velocity_window
// (
//     const GuiMonitor &gui_monitor,
//     const GuiSimulationControl &gui_simulation_control,
//     const GuiVisualizationData &gui_visualization_data,
//     const GuiProgress &gui_progress,
//     GuiWindows &gui_windows, 
//     GuiVelocityQuiverData &gui_velocity_quiver_data,
//     GuiColormaps &gui_colormaps
// )
// {
//     if(gui_windows.show_velocity)
//     {
//         ImGui::SetNextWindowPos
//         (
//             ImVec2
//             (
//                 (1.0 / 4) * gui_monitor.viewport_work_size.x, 
//                 gui_windows.menu_bar_size
//             ), 
//             ImGuiCond_Appearing
//         ); 

//         ImGui::SetNextWindowSize
//         (
//             ImVec2
//             (
//                 (3.0 / 4) * gui_monitor.viewport_work_size.x,
//                 (1.0 / 2) * gui_monitor.viewport_work_size.y - 1.0/2 * gui_windows.menu_bar_size
//             ), 
//             ImGuiCond_Appearing
//         );

//         if(ImGui::Begin("Velocity", &gui_windows.show_velocity))
//         {
//             ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
//             gui_items::colormap_picker(gui_colormaps.velocity_colormap, "Velocity");

//             ImGui::SameLine();
//             ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
//             ImGui::SameLine();
//             ImGui::InputDouble("Lower scale", &gui_colormaps.velocity_colormap_lower_scale, 0.01, 0.1);
//             ImGui::SameLine();
//             ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
//             ImGui::SameLine();
//             ImGui::InputDouble("Upper scale", &gui_colormaps.velocity_colormap_upper_scale, 0.01, 0.1);
//             ImGui::SameLine();
//             ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
//             ImGui::SameLine();
//             if(ImGui::Checkbox("Enable vector plot", &gui_windows.enable_velocity_quiver)){}


//             ImPlot::PushColormap(gui_colormaps.velocity_colormap);

//             if
//             (
//                 ImPlot::BeginPlot
//                 (
//                     "Velocity",
//                     ImVec2
//                     (
//                         ImGui::GetContentRegionAvail().x - gui_monitor.monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
//                         ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
//                     ),
//                     ImPlotFlags_NoLegend|ImPlotFlags_NoMouseText|ImPlotFlags_Crosshairs
//                 )
//             ) 
//             {
//                 ImPlot::SetupAxes
//                 (
//                     nullptr, 
//                     nullptr, 
//                     ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks, 
//                     ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks
//                 );

//                 ImPlot::PlotHeatmap
//                 (
//                     "Absolute velocities",
//                     gui_visualization_data.absolute_velocity_values->data(),
//                     gui_visualization_data.vertical_nodes,
//                     gui_visualization_data.horizontal_nodes,
//                     gui_colormaps.velocity_colormap_lower_scale, 
//                     gui_colormaps.velocity_colormap_upper_scale, 
//                     nullptr, 
//                     ImPlotPoint(0,0),
//                     ImPlotPoint(gui_visualization_data.horizontal_nodes, gui_visualization_data.vertical_nodes)
//                 );

//                 ImPlot::PopColormap;
//                 ImPlot::PushColormap("SOLID_MASK");

//                 ImPlot::PlotHeatmap
//                 (
//                     "Solid mask for velocity",
//                     gui_visualization_data.density_values->data(),
//                     gui_visualization_data.vertical_nodes,
//                     gui_visualization_data.horizontal_nodes,
//                     -1,
//                     1, 
//                     nullptr, 
//                     ImPlotPoint(0,0),
//                     ImPlotPoint(gui_visualization_data.horizontal_nodes, gui_visualization_data.vertical_nodes)
//                 );

//                 ImPlot::PopColormap;
//                 ImPlot::PushColormap(ImPlotColormap_Greys);
//                 ImPlot::SetNextLineStyle(ImVec4(0,0,0,0.5), 3.f);

//                 if(gui_windows.enable_velocity_quiver)
//                 {
//                     ImPlot::PlotLine
//                     (
//                         "Quiver plot", 
//                         gui_velocity_quiver_data.x_values->data(), 
//                         gui_velocity_quiver_data.y_values->data(), 
//                         2 * gui_visualization_data.absolute_velocity_values->size(), 
//                         ImPlotLineFlags_Segments
//                     );
//                 }

//                 if(ImPlot::IsPlotHovered()) 
//                 {
//                     ImPlotPoint mouse = ImPlot::GetPlotMousePos();

//                     if(0 <= mouse.x && mouse.x < gui_visualization_data.horizontal_nodes && 0 <= mouse.y && mouse.y < gui_visualization_data.vertical_nodes)
//                     {
//                         ImGui::BeginTooltip();
//                         ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);

//                         if(gui_visualization_data.density_values->at(gui_visualization_data.horizontal_nodes * (int)floor(mouse.y) + (int)floor(mouse.x)) != -1)
//                         {
//                             ImGui::Text("x velocity: %.6f", gui_visualization_data.x_velocity_values->at(gui_visualization_data.horizontal_nodes * ((int)floor(mouse.y)) + (int)floor(mouse.x)));
//                             ImGui::Text("y velocity: %.6f", gui_visualization_data.y_velocity_values->at(gui_visualization_data.horizontal_nodes * ((int)floor(mouse.y)) + (int)floor(mouse.x)));
//                         }
//                         else
//                         {
//                             ImGui::Text("Solid node"); 
//                         }
//                         ImGui::EndTooltip();
//                     }
//                 }

//                 ImPlot::EndPlot();
//             }
//             ImGui::SameLine();
//             ImPlot::ColormapScale
//             (
//                 "Velocity",
//                 gui_colormaps.velocity_colormap_lower_scale, 
//                 gui_colormaps.velocity_colormap_upper_scale, 
//                 ImVec2
//                 (
//                     gui_monitor.monitor_x_scale * 100,
//                     ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
//                 ), 
//                 "%g", 
//                 ImPlotColormapScaleFlags_None, 
//                 gui_colormaps.velocity_colormap
//             );   
//         }
//         ImGui::End();
//     }
// }
