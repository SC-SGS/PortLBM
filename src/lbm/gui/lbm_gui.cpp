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

lbm::gui::Colormaps::Colormaps() 
:
density_colormap(ImPlotColormap_Viridis),
density_colormap_lower_scale(0.0f),
density_colormap_upper_scale(0.0f),
velocity_colormap(ImPlotColormap_Hot),
velocity_colormap_lower_scale(0.0),
velocity_colormap_upper_scale(0.0f)
{};

lbm::gui::VelocityQuiverData::VelocityQuiverData(const size_t &size) 
:
x_values{std::make_unique<std::vector<double>>(size)},
y_values{std::make_unique<std::vector<double>>(size)}
{};

lbm::gui::Gui::Gui(const std::string &&window_title)
:
windows(std::make_unique<Windows>()),
simulation_control(std::make_unique<SimulationControl>()),
progress(std::make_unique<Progress>()),
monitor(std::make_unique<Monitor>()),
colormaps(std::make_unique<Colormaps>()),
properties_stored(lbm::file_interaction::json_to_properties()),
properties_buffered(lbm::file_interaction::json_to_properties()),
velocity_quiver_data(std::make_unique<VelocityQuiverData>(2 * properties_stored->domain_node_count)),
window_title(std::make_unique<std::string>(window_title)),
properties_changed(false)
{};

// void lbm::gui::initialize_executor
// (
//     const std::string &algorithm,
//     const std::string &data_layout,
//     std::unique_ptr<execution::Executor<core::SimulationResults>> &executor
// )
// {
//     if(algorithm == "gpu-two-lattice-linear")
//     {
//         if(data_layout == "stream")
//         {
//             executor = std::make_unique<execution::SYCLExecutor<core::access::LBMStreamAccessor>>(new execution::SYCLExecutor<core::access::LBMStreamAccessor>());
//         }
//         else if(data_layout == "collision")
//         {
//             executor = std::make_unique<execution::SYCLExecutor<core::access::LBMCollisionAccessor>>(new execution::SYCLExecutor<core::access::LBMCollisionAccessor>());
//         }
//         else if(data_layout == "bundle")
//         {
//             executor = std::make_unique<execution::SYCLExecutor<core::access::LBMBundleAccessor>>(new execution::SYCLExecutor<core::access::LBMBundleAccessor>());
//         }
//         else
//         {
//             throw exceptions::Exception("Unknown data layout: " + data_layout);
//         }
//     }
//     else if(algorithm == "gpu-two-lattice")
//     {
//         throw exceptions::Exception("This algorithm is not implemented yet.");
//     }
//     else if(algorithm == "gpu-swap")
//     {
//         throw exceptions::Exception("This algorithm is not implemented yet.");
//     }
//     else
//     {
//         throw exceptions::Exception("Unknown algorithm: " + algorithm);
//     }
// }

int lbm::gui::Gui::run()
{
    glfwSetErrorCallback(rendering::glfw_error_callback);
    if (!glfwInit())
        return 1;

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    std::unique_ptr<lbm::execution::SYCLExecutor<core::SimulationResults>> executor;

    // LBM initializations
    //initialize_executor(properties_stored->algorithm, properties_stored->data_layout, executor);
    executor = std::make_unique<execute::SYCLExecutor>();

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(monitor->video_mode->width, monitor->video_mode->height, window_title->c_str(), nullptr, nullptr);
    if (window == nullptr)
    {
        return 1;
    }
        
    glfwMakeContextCurrent(window);
    //glfwSwapInterval(1); // Enable vsync

    contexts::create_contexts();
    styles::set_light_style();
    glfwGetMonitorContentScale(monitor->primary_monitor, &(monitor->monitor_x_scale), &(monitor->monitor_x_scale));

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.Fonts->AddFontFromFileTTF("../fonts/DroidSans.ttf", 2 * sqrt(monitor->monitor_x_scale * monitor->monitor_x_scale) * 9.0f);

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    glfwGetFramebufferSize(window, &(monitor->display_width), &(monitor->display_height));

    // Timer initializations
    SimpleTimer timer;
    SimpleTimer timer_framerate;

    misc::add_solid_colormap();

    ImPlotStyle& implot_style = ImPlot::GetStyle();
    implot_style.PlotPadding = ImVec2(monitor->monitor_x_scale * 20, monitor->monitor_y_scale * 20);

    colormaps->density_colormap_lower_scale = std::min({executor->properties->inlet_density, executor->properties->outlet_density});
    colormaps->density_colormap_upper_scale = std::max({executor->properties->inlet_density, executor->properties->outlet_density});
    colormaps->velocity_colormap_upper_scale = sqrt(pow(executor->properties->inlet_velocity_x, 2) + pow(executor->properties->inlet_velocity_y, 2)); 

    // Eternal loop of imaging magic
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        monitor->viewport = std::make_unique<ImGuiViewport>(*ImGui::GetMainViewport());

        items::menu_bar(*windows);

        windows::read_from_file_window(*monitor, *windows);

        windows::properties_window
        (
            *monitor,
            *executor,
            *properties_buffered,
            properties_changed,
            *windows,
            *simulation_control
        );

        windows::simulation_status_window
        (
            *monitor,
            *windows,
            *simulation_control,
            *progress,
            *executor
        );

        if(simulation_control->is_simulation_active && !simulation_control->is_paused)
        {
            if(executor->is_ready())
            {
                if(timer_framerate.elapsed() > 0.25)
                {
                    progress->framerate = 1 / timer.elapsed();
                    progress->frametime = 1000.0f / progress->framerate;
                    timer_framerate.restart();
                }
                
                timer.restart();
                progress->current_iter++;
                progress->progress = (double)progress->current_iter / executor->properties->time_steps;

                if(simulation_control->results_to_csv)
                {
                    
                }

                for(int i = 0; i < executor->properties->domain_node_count; ++i)
                {
                    (*(executor->simulation_results->absolute_velocities))[i] = sqrt(pow((*(executor->simulation_results->x_velocities))[i], 2) + pow((*(executor->simulation_results->y_velocities))[i], 2));
                }

                if(windows->enable_velocity_quiver)
                {
                    velocity_quiver_data->x_values->assign(2 * executor->properties->domain_node_count, 0);
                    velocity_quiver_data->y_values->assign(2 * executor->properties->domain_node_count, 0);

                    for(int y = 1; y < executor->properties->vertical_nodes - 1; ++y)
                    {
                        for(int x = 1; x < executor->properties->horizontal_nodes - 1; ++x)
                        {
                            unsigned int dnode_index = core::access::get_node_index(x-1, y-1, executor->properties->horizontal_nodes-2);
                            unsigned int node_index = core::access::get_node_index(x, y, executor->properties->horizontal_nodes);
                            unsigned int velocity_value_index = core::access::results::get_result_index_no_ghosts(core::access::get_node_index(x, y, executor->properties->horizontal_nodes), executor->properties->horizontal_nodes);
                            //std::cout << "Reaching node index " << node_index << ", calculated from coordinates (x = " << x << ", y = " << y << ")\n";
                            if(executor->simulation_results->absolute_velocities->at(velocity_value_index) > 1e-15)
                            {
                                double base_x = 0.5 + x - 1;
                                double base_y = 0.5 + y - 1;
                                
                                (*velocity_quiver_data->x_values)[2 * dnode_index] = base_x;
                                (*velocity_quiver_data->y_values)[2 * dnode_index] = base_y;

                                double offset_x = base_x + 0.5 * (1.0 / executor->simulation_results->absolute_velocities->at(velocity_value_index)) * executor->simulation_results->x_velocities->at(velocity_value_index);
                                double offset_y = base_y + 0.5 * (1.0 / executor->simulation_results->absolute_velocities->at(velocity_value_index)) * executor->simulation_results->y_velocities->at(velocity_value_index);

                                (*velocity_quiver_data->x_values)[2 * dnode_index + 1] = offset_x;
                                (*velocity_quiver_data->y_values)[2 * dnode_index + 1] = offset_y;
                            }
                        }  
                    }
                }
                executor->execute();

                // Check if simulation is finished
                if(progress->current_iter == executor->properties->time_steps)
                {
                    progress->current_iter = 0;
                    progress->progress = 0;
                    simulation_control->is_simulation_active = false;

                    if(simulation_control->results_to_csv)
                    {

                    }
                }
            }
        }

        windows::density_window
        (   
            *executor->properties,
            *monitor, 
            *simulation_control, 
            *executor->simulation_results,
            *progress, 
            *windows, 
            *colormaps
        );

        windows::velocity_window
        (
            *executor->properties,
            *monitor, 
            *simulation_control, 
            *executor->simulation_results,
            *progress, 
            *windows, 
            *velocity_quiver_data,
            *colormaps
        );

        rendering::render(window, *monitor);
    }
    std::cout << "Exiting program \n";
    contexts::destroy(window);
    std::cout << "Contexts destroyed \n";
    return 0;
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
                (1.0 / 4) * gui_monitor.viewport->WorkSize.x, 
                1.0 / 2 * windows.menu_bar_size + 1.0 / 2 * gui_monitor.viewport->WorkSize.y
            ), 
            ImGuiCond_Appearing
        ); 

        ImGui::SetNextWindowSize
        (
            ImVec2
            (
                (3.0 / 4) * gui_monitor.viewport->WorkSize.x, 
                1.0 / 2 * gui_monitor.viewport->WorkSize.y - 1.0 / 2 * windows.menu_bar_size
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
                (1.0 / 4) * gui_monitor.viewport->WorkSize.x, 
                windows.menu_bar_size
            ), 
            ImGuiCond_Appearing
        ); 

        ImGui::SetNextWindowSize
        (
            ImVec2
            (
                (3.0 / 4) * gui_monitor.viewport->WorkSize.x,
                (1.0 / 2) * gui_monitor.viewport->WorkSize.y - 1.0 / 2 * windows.menu_bar_size
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
