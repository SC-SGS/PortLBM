/**
 * @file        lbm_gui.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This file contains implementations for GUI operations declared in lbm_gui.hpp.
 *              
 * @version     2.0
 * 
 * @date        December 2024
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
properties_stored(std::make_unique<core::Properties>(lbm::file_interaction::json_to_properties())),
properties_buffered(std::make_unique<core::Properties>(lbm::file_interaction::json_to_properties())),
velocity_quiver_data(std::make_unique<VelocityQuiverData>(2 * properties_stored->domain_node_count)),
window_title(std::make_unique<std::string>(window_title)),
properties_changed(false)
{};

void lbm::gui::Gui::run()
{
    glfwSetErrorCallback(rendering::glfw_error_callback);
    if (!glfwInit()) {throw exceptions::Exception("Failed to initialize GLFW.");}

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    // Setting up executor and initializing all important simulation data
    std::unique_ptr<execution::SYCLExecutor> executor = std::make_unique<execution::SYCLExecutor>();

    // Create window with graphics context
    monitor->primary_monitor = glfwGetPrimaryMonitor();
    monitor->video_mode = glfwGetVideoMode(monitor->primary_monitor);
    GLFWwindow* window = glfwCreateWindow(monitor->video_mode->width, monitor->video_mode->height, window_title->c_str(), nullptr, nullptr);
    if (window == nullptr) {throw exceptions::Exception("Failed to create GLFW window.");}
        
    glfwMakeContextCurrent(window);
    //glfwSwapInterval(1); // Enable vsync
    contexts::create_contexts();
    styles::set_light_style();
    glfwGetMonitorContentScale(monitor->primary_monitor, &(monitor->monitor_x_scale), &(monitor->monitor_x_scale));

    // IO
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

    // ImPlot style
    ImPlotStyle& implot_style = ImPlot::GetStyle();
    implot_style.PlotPadding = ImVec2(monitor->monitor_x_scale * 20, monitor->monitor_y_scale * 20);

    // ImPlot colormaps
    misc::add_solid_colormap();
    colormaps->density_colormap_lower_scale = 
        std::min({executor->algorithm->simulation->properties->inlet_density, executor->algorithm->simulation->properties->outlet_density});
    colormaps->density_colormap_upper_scale = 
        std::max({executor->algorithm->simulation->properties->inlet_density, executor->algorithm->simulation->properties->outlet_density});
    colormaps->velocity_colormap_upper_scale = 
        sqrt(pow(executor->algorithm->simulation->properties->inlet_velocity_x, 2) + pow(executor->algorithm->simulation->properties->inlet_velocity_y, 2)); 

    // Eternal loop of imaging magic
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        monitor->viewport = std::make_unique<ImGuiViewport>(*ImGui::GetMainViewport());

        items::menu_bar(*this);

        windows::read_from_file_window(*this);

        windows::properties_window(executor, properties_buffered, properties_changed, *this);

        windows::simulation_status_window(*this, executor);

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
                progress->progress = (double)progress->current_iter / executor->algorithm->simulation->properties->time_steps;

                if(simulation_control->results_to_csv)
                {
                    
                }

                if(windows->enable_velocity_quiver)
                {
                    unsigned int dnode_index = 0;
                    unsigned int node_index = 0;
                    unsigned int velocity_value_index = 0;
                    double base_x = 0;
                    double base_y = 0;
                    double offset_x = 0;
                    double offset_y = 0;

                    velocity_quiver_data->x_values->assign(2 * executor->algorithm->simulation->properties->domain_node_count, 0);
                    velocity_quiver_data->y_values->assign(2 * executor->algorithm->simulation->properties->domain_node_count, 0);

                    for(int y = 1; y < executor->algorithm->simulation->properties->vertical_nodes - 1; ++y)
                    {
                        for(int x = 1; x < executor->algorithm->simulation->properties->horizontal_nodes - 1; ++x)
                        {
                            dnode_index = core::access::get_node_index(x-1, y-1, executor->algorithm->simulation->properties->horizontal_nodes-2);
                            node_index = core::access::get_node_index(x, y, executor->algorithm->simulation->properties->horizontal_nodes);

                            velocity_value_index = 
                                core::access::results::get_result_index_no_ghosts(
                                    core::access::get_node_index(x, y, executor->algorithm->simulation->properties->horizontal_nodes), 
                                    executor->algorithm->simulation->properties->horizontal_nodes
                                );

                            if(executor->algorithm->simulation->results->absolute_velocities_cpu->at(velocity_value_index) > 1e-15)
                            {
                                base_x = 0.5 + x - 1;
                                base_y = 0.5 + y - 1;
                                
                                (*velocity_quiver_data->x_values)[2 * dnode_index] = base_x;
                                (*velocity_quiver_data->y_values)[2 * dnode_index] = base_y;

                                offset_x = base_x + 0.5 * (1.0 / executor->algorithm->simulation->results->absolute_velocities_cpu->at(velocity_value_index)) 
                                                        * executor->algorithm->simulation->results->x_velocities_cpu->at(velocity_value_index);

                                offset_y = base_y + 0.5 * (1.0 / executor->algorithm->simulation->results->absolute_velocities_cpu->at(velocity_value_index)) 
                                                        * executor->algorithm->simulation->results->y_velocities_cpu->at(velocity_value_index);

                                (*velocity_quiver_data->x_values)[2 * dnode_index + 1] = offset_x;
                                (*velocity_quiver_data->y_values)[2 * dnode_index + 1] = offset_y;
                            }
                        }  
                    }
                }

                executor->execute();

                // Check if simulation is finished
                if(progress->current_iter == executor->algorithm->simulation->properties->time_steps)
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

        windows::density_window(*executor, *this);
        windows::velocity_window(*executor, *this);

        rendering::render(window, *this);
    }
    contexts::destroy(window);
}

void lbm::gui::windows::read_from_file_window(gui::Gui &gui)
{
    if(gui.windows->show_read_from_file_window)
    {
        if(ImGui::Begin("Read from file", &gui.windows->show_read_from_file_window))
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
    const execution::SYCLExecutor &sycl_executor,
    gui::Gui &gui
)
{
    if(gui.windows->show_density)
    {
        ImGui::SetNextWindowPos
        (
            ImVec2
            (
                (1.0 / 4) * gui.monitor->viewport->WorkSize.x, 
                1.0 / 2 * gui.windows->menu_bar_size + 1.0 / 2 * gui.monitor->viewport->WorkSize.y
            ), 
            ImGuiCond_Appearing
        ); 

        ImGui::SetNextWindowSize
        (
            ImVec2
            (
                (3.0 / 4) * gui.monitor->viewport->WorkSize.x, 
                1.0 / 2 * gui.monitor->viewport->WorkSize.y - 1.0 / 2 * gui.windows->menu_bar_size
            ), 
            ImGuiCond_Appearing
        );

        if(ImGui::Begin("Density", &gui.windows->show_density))
        {
            ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
            items::colormap_picker(gui.colormaps->density_colormap, "Density");
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20, 0));
            ImGui::SameLine();
            ImGui::InputDouble("Lower scale", &gui.colormaps->density_colormap_lower_scale, 0.05, 0.1);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20, 0));
            ImGui::SameLine();
            ImGui::InputDouble("Upper scale", &gui.colormaps->density_colormap_upper_scale, 0.05, 0.1);
            ImPlot::PushColormap(gui.colormaps->density_colormap);

            if
            (
                ImPlot::BeginPlot
                (
                    "Density",
                    ImVec2
                    (
                        ImGui::GetContentRegionAvail().x - gui.monitor->monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
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
                    sycl_executor.algorithm->simulation->results->densities_cpu->data(),
                    sycl_executor.algorithm->simulation->properties->vertical_nodes - 2,
                    sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2,
                    gui.colormaps->density_colormap_lower_scale,
                    gui.colormaps->density_colormap_upper_scale,
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2, sycl_executor.algorithm->simulation->properties->vertical_nodes - 2)
                );

                ImPlot::PushColormap("SOLID_MASK");
                ImPlot::PlotHeatmap
                (
                    "Solid mask for density",
                    sycl_executor.algorithm->simulation->results->densities_cpu->data(),
                    sycl_executor.algorithm->simulation->properties->vertical_nodes - 2,
                    sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2,
                    -1,
                    1, 
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2, sycl_executor.algorithm->simulation->properties->vertical_nodes - 2)
                );

                if (ImPlot::IsPlotHovered()) 
                {
                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();

                    if(0 <= mouse.x && mouse.x < (sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2) && 0 <= mouse.y && mouse.y < (sycl_executor.algorithm->simulation->properties->vertical_nodes - 2))
                    {
                        ImGui::BeginTooltip();
                        ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);
                        double value = sycl_executor.algorithm->simulation->results->densities_cpu->at((sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2) * ((int)floor(mouse.y)) + (int)floor(mouse.x));
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
                gui.colormaps->density_colormap_lower_scale,
                gui.colormaps->density_colormap_upper_scale, 
                ImVec2
                (
                    gui.monitor->monitor_x_scale * 100,
                    ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                ), 
                "%g", 
                ImPlotColormapScaleFlags_None, 
                gui.colormaps->density_colormap
            );   
        }
        ImGui::End();
    }
}

void lbm::gui::windows::velocity_window
(
    const execution::SYCLExecutor &sycl_executor,
    gui::Gui &gui
)
{
    if(gui.windows->show_velocity)
    {
        ImGui::SetNextWindowPos
        (
            ImVec2
            (
                (1.0 / 4) * gui.monitor->viewport->WorkSize.x, 
                gui.windows->menu_bar_size
            ), 
            ImGuiCond_Appearing
        ); 

        ImGui::SetNextWindowSize
        (
            ImVec2
            (
                (3.0 / 4) * gui.monitor->viewport->WorkSize.x,
                (1.0 / 2) * gui.monitor->viewport->WorkSize.y - 1.0 / 2 * gui.windows->menu_bar_size
            ), 
            ImGuiCond_Appearing
        );

        if(ImGui::Begin("Velocity", &gui.windows->show_velocity))
        {
            ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
            items::colormap_picker(gui.colormaps->velocity_colormap, "Velocity");

            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
            ImGui::SameLine();
            ImGui::InputDouble("Lower scale", &gui.colormaps->velocity_colormap_lower_scale, 0.01, 0.1);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
            ImGui::SameLine();
            ImGui::InputDouble("Upper scale", &gui.colormaps->velocity_colormap_upper_scale, 0.01, 0.1);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
            ImGui::SameLine();
            if(ImGui::Checkbox("Enable vector plot", &gui.windows->enable_velocity_quiver)){}


            ImPlot::PushColormap(gui.colormaps->velocity_colormap);

            if
            (
                ImPlot::BeginPlot
                (
                    "Velocity",
                    ImVec2
                    (
                        ImGui::GetContentRegionAvail().x - gui.monitor->monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
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
                    sycl_executor.algorithm->simulation->results->absolute_velocities_cpu->data(),
                    sycl_executor.algorithm->simulation->properties->vertical_nodes - 2,
                    sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2,
                    gui.colormaps->velocity_colormap_lower_scale, 
                    gui.colormaps->velocity_colormap_upper_scale, 
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2, sycl_executor.algorithm->simulation->properties->vertical_nodes - 2)
                );

                ImPlot::PopColormap(1);
                ImPlot::PushColormap("SOLID_MASK");

                ImPlot::PlotHeatmap
                (
                    "Solid mask for velocity",
                    sycl_executor.algorithm->simulation->results->densities_cpu->data(),
                    sycl_executor.algorithm->simulation->properties->vertical_nodes - 2,
                    sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2,
                    -1,
                    1, 
                    nullptr, 
                    ImPlotPoint(0,0),
                    ImPlotPoint(sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2, sycl_executor.algorithm->simulation->properties->vertical_nodes - 2)
                );

                ImPlot::PopColormap(1);
                ImPlot::PushColormap(ImPlotColormap_Greys);
                ImPlot::SetNextLineStyle(ImVec4(0, 0, 0, 0.5), 3.f);

                if(gui.windows->enable_velocity_quiver)
                {
                    ImPlot::PlotLine
                    (
                        "Quiver plot", 
                        gui.velocity_quiver_data->x_values->data(), 
                        gui.velocity_quiver_data->y_values->data(), 
                        2 * sycl_executor.algorithm->simulation->results->absolute_velocities_cpu->size(), 
                        ImPlotLineFlags_Segments
                    );
                }

                if(ImPlot::IsPlotHovered()) 
                {
                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();

                    if(0 <= mouse.x && mouse.x < (sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2) && 0 <= mouse.y && mouse.y < (sycl_executor.algorithm->simulation->properties->vertical_nodes - 2))
                    {
                        ImGui::BeginTooltip();
                        ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);

                        if(sycl_executor.algorithm->simulation->results->densities_cpu->at((sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2) * (int)floor(mouse.y) + (int)floor(mouse.x)) != -1)
                        {
                            ImGui::Text("x velocity: %.6f", sycl_executor.algorithm->simulation->results->x_velocities_cpu->at((sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2) * ((int)floor(mouse.y)) + (int)floor(mouse.x)));
                            ImGui::Text("y velocity: %.6f", sycl_executor.algorithm->simulation->results->y_velocities_cpu->at((sycl_executor.algorithm->simulation->properties->horizontal_nodes - 2) * ((int)floor(mouse.y)) + (int)floor(mouse.x)));
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
                gui.colormaps->velocity_colormap_lower_scale, 
                gui.colormaps->velocity_colormap_upper_scale, 
                ImVec2
                (
                    gui.monitor->monitor_x_scale * 100,
                    ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                ), 
                "%g", 
                ImPlotColormapScaleFlags_None, 
                gui.colormaps->velocity_colormap
            );   
        }
        ImGui::End();
    }
}
