/**
 * @file        lbm_gui.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains all declarations and some implementations of the GUI
 *              for the lattice Boltzmann simulation developed in my Bachelor thesis.
 * 
 * @version     2.2
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_GUI_HPP
#define LBM_GUI_HPP

// INCLUDES /////////////////////////////////////////////////////////////////////////////////////////////////////////// 

// LBM
#include "simple_timer.hpp"
#include "gui_constants.hpp"
#include "../file_interaction/file_interaction.hpp"
#include "../execution/algorithm_handler.hpp"
#include "../core/simulation.hpp"
#include "../exceptions/exceptions.hpp"

// Standard library
#include <memory>

// ImGui
#include "../../external/imgui/imgui.h"
#include "../../external/imgui/imgui_internal.h"
#include "../../external/imgui/imgui_impl_glfw.h"
#include "../../external/imgui/imgui_impl_opengl3.h"

// ImPlot
#include "../../external/implot/implot.h"
#include "../../external/implot/implot_internal.h"

// GL Silence Deprecation
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif

// GLFW
#include <GLFW/glfw3.h>

// NLohmann JSON
#include <nlohmann/json.hpp>

// END OF INCLUDES // START OF CODE ///////////////////////////////////////////////////////////////////////////////////

namespace lbm
{

    namespace gui
    {

        /**
        * @brief Helper function displaying an error callback message on the console.
        * 
        * @param[in]    error       the integer error id
        * @param[in]    description char array containing an error message
        */
        inline void glfw_error_callback(int error, const char* description)
        {
            fprintf(stderr, "GLFW Error %d: %s\n", error, description);
        }

        /**
         * @brief This struct contains information regarding the windows of the GUI:
         *        
         *        - Which windows should be displayed
         * 
         *        - Whether or not the live visualization should be displayed
         * 
         *        - Whether or not the velocity quiver is to be displayed 
         *          (as the velocity heatmap can be plotted individually)
         * 
         *        - The size of the menu bar for window sizing and placement purposes 
         */
        struct Windows
        {
            explicit Windows(bool debug);

            bool show_properties;
            bool show_simulation_status;
            bool show_framerate;
            bool show_read_from_file_window;
            bool show_density;
            bool show_velocity;
            bool enable_live_visualization;
            bool enable_velocity_quiver;
            bool show_debug_window;
            float menu_bar_size;
            const char *node_content;
        };

        /**
         * @brief This struct contains information regarding the control of the simulation.
         */
        struct SimulationControl
        {
            explicit SimulationControl();

            bool is_paused;
            bool is_simulation_active;
            bool results_to_csv;
            char result_file_name [200];
        };

        /**
         * @brief This struct keeps track of the progress of the simulation and the framerate.
         */
        struct Progress
        {
            explicit Progress();

            double progress;
            double framerate_backend;
            double frametime_backend;
            double framerate_frontend;
            double frametime_frontend;
        };

        /**
         * @brief This struct contains information on the primary monitor and the viewport.
         */
        struct Monitor
        {
            GLFWmonitor *primary_monitor;
            const GLFWvidmode *video_mode;

            float monitor_x_scale;
            float monitor_y_scale;

            // Amount of pixels in horizontal direction
            int display_width;

            // Amount of pixels in vertical direction
            int display_height;
        
            std::unique_ptr<ImGuiViewport> viewport;

            /**
             * @brief   Constructs a Monitor object. Notice that upon first initialization, `primary_monitor` and `video_mode` are
             *          potentially initialized to `nullptr` because setting them up with proper values requires an existing
             *          GLFW context.
             */
            inline explicit Monitor()
            :
            primary_monitor(glfwGetPrimaryMonitor()),
            video_mode(glfwGetVideoMode(primary_monitor)),
            monitor_x_scale(1.f),
            monitor_y_scale(1.f),
            display_width(0),
            display_height(0),
            viewport(std::make_unique<ImGuiViewport>())
            {};
        };

        /**
         * @brief This struct contains information on the colormaps used in the density and velocity plot.
         */
        struct Colormaps
        {
            explicit Colormaps();

            ImPlotColormap density_colormap;
            double density_colormap_lower_scale;
            double density_colormap_upper_scale;
            ImPlotColormap velocity_colormap;
            double velocity_colormap_lower_scale;
            double velocity_colormap_upper_scale;
        };

        /**
         * @brief This struct contains the x and y values of the velocity quiver.
         */
        struct VelocityQuiverData
        {
            explicit VelocityQuiverData(const size_t &size);

            std::unique_ptr<std::vector<double>> x_values;
            std::unique_ptr<std::vector<double>> y_values;
        };

        struct FPSBuffer 
        {
            int max_size;
            int offset;
            ImVector<ImVec2> data;
            float current_time;

            explicit FPSBuffer(const int size)
            :
            max_size(size),
            offset(0),
            current_time(0)
            {
                data.reserve(max_size);
            }

            inline void add(const double x, const float y) 
            {
                if (data.size() < max_size) { data.push_back(ImVec2(x,y)); }
                else 
                {
                    data[offset] = ImVec2(x,y);
                    offset =  (offset + 1) % max_size;
                }
            }
            inline void shrink_to_zero() 
            {
                if (data.size() > 0) 
                {
                    data.shrink(0);
                    offset = 0;
                }
            }
        };

// GUI CLASS //////////////////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * @brief   This class implements a runnable GUI relying on the specified algorithm handler to interact with
         *          the simulation running in the background.
         * 
         * @tparam  A   any child class of `lbm::execution::AlgorithmHandler`
         */
        template <execution::AlgorithmHandlerConcept A>
        class Gui
        {
            private:

// MEMBERS ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            std::unique_ptr<core::Properties> properties_gui;
            std::unique_ptr<Windows> windows;
            std::unique_ptr<SimulationControl> simulation_control;
            std::unique_ptr<Progress> progress;
            std::unique_ptr<Monitor> monitor;
            std::unique_ptr<Colormaps> colormaps;
            std::unique_ptr<VelocityQuiverData> velocity_quiver_data;
            std::unique_ptr<std::string> window_title;
            std::unique_ptr<FPSBuffer> fps_buffer;
            std::unique_ptr<A> algorithm_handler;

            GLFWwindow *glfw_window;

            bool properties_changed;

// STYLES /////////////////////////////////////////////////////////////////////////////////////////////////////////////

            /** @brief Sets the style of ImGui and ImPlot to light colors. */
            inline void set_light_style()
            {
                ImGui::StyleColorsLight();
                ImVec4 clear_color = ImVec4(0.7f, 0.7f, 0.7f, 1.00f);
                glClearColor
                (
                    clear_color.x * clear_color.w, 
                    clear_color.y * clear_color.w, 
                    clear_color.z * clear_color.w, 
                    clear_color.w
                );
                glClear(GL_COLOR_BUFFER_BIT);
                ImPlot::StyleColorsLight();
            }

            /** @brief Sets the style of ImGui and ImPlot to dark colors. */
            inline void set_dark_style()
            {
                ImGui::StyleColorsDark();
                ImVec4 clear_color = ImVec4(0.1f, 0.1f, 0.1f, 1.00f);
                glClearColor
                (
                    clear_color.x * clear_color.w, 
                    clear_color.y * clear_color.w, 
                    clear_color.z * clear_color.w, 
                    clear_color.w
                );
                glClear(GL_COLOR_BUFFER_BIT);
                ImPlot::StyleColorsDark();
            }

            /** @brief Sets the style of ImGui and ImPlot to the classical ImGui style. */
            inline void set_classic_style()
            {
                ImGui::StyleColorsClassic();
                ImVec4 clear_color = ImVec4(0.2f, 0.2f, 0.2f, 1.00f);
                glClearColor
                (
                    clear_color.x * clear_color.w, 
                    clear_color.y * clear_color.w, 
                    clear_color.z * clear_color.w, 
                    clear_color.w
                );
                glClear(GL_COLOR_BUFFER_BIT);
                ImPlot::StyleColorsClassic();
            }

// BUTTONS ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief Creates a run button that is capable of launching a new simulation or resuming a paused simulation.
             *        Simulation values are initialized with placeholder values.
             *        
             *        Pressing this button when no simulation is active, i.e., after the start of the program, 
             *        after a completed simulation or after the abortion of a simulation run enacts the launch of a new 
             *        simulation run.
             *       
             *        Pressing this button when a simulation is active but paused resumes work on it.
             * 
             *        Pressing this button when a simulation is active and not paused has no effect.
             * 
             */
            inline void run_button()
            {
                ImGui::PushID(0);
                ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.333f, 0.5f, 0.6f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.333f, 0.75f, 0.75f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.333f, 0.9f, 0.9f));

                if (ImGui::Button("run", ImVec2(1.0/4 * ImGui::GetWindowSize().x*0.5f, 0.0f)))
                {
                    simulation_control->is_paused = false;
                    progress->progress = 0;

                    if(!simulation_control->is_simulation_active)
                    {
                        simulation_control->is_simulation_active = true;
                        file_interaction::properties_to_json(*properties_gui);
                        properties_gui.reset();
                        properties_gui = std::make_unique<core::Properties>(file_interaction::json_to_properties("../settings/settings.json", -2));
                        properties_changed = false;
                        algorithm_handler->pause();
                        algorithm_handler->initialize();

                        colormaps->density_colormap_lower_scale = 
                            std::min({algorithm_handler->get_inlet_density(), algorithm_handler->get_outlet_density()});
                        colormaps->density_colormap_upper_scale = 
                            std::max({algorithm_handler->get_inlet_density(), algorithm_handler->get_outlet_density()});
                    }

                    algorithm_handler->start();
                }

                ImGui::PopStyleColor(3);
                ImGui::PopID();
            }

            /**
             * @brief Creates a pause button that is capable of pausing an active simulation.
             *        
             *        Pressing this button when no simulation is active, i.e., after the start of the program, 
             *        after a completed simulation or after the abortion of a simulation run has no effect.
             *       
             *        Pressing this button when a simulation is active but already paused has no effect.
             * 
             *        Pressing this button when a simulation is active and not already paused pauses it.
             * 
             */
            inline void pause_button()
            {
                ImGui::PushID(1);
                ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.166f, 0.5f, 0.6f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.166f, 0.75f, 0.75f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.166f, 0.9f, 0.9f));
                if(ImGui::Button("pause", ImVec2(1.0/4 * ImGui::GetWindowSize().x*0.5f, 0.0f)))                           
                {
                    simulation_control->is_paused = true;
                    algorithm_handler->pause();
                }
                ImGui::PopStyleColor(3);
                ImGui::PopID();
            }

            /**
             * @brief Creates an abort button that is capable of aborting an active simulation.
             *        All progess counters are reset to zero.
             *        Values are not re-initialized such that the last progress remains visible unless the user performs any changes.
             *        
             *        Pressing this button when no simulation is active, i.e., after the start of the program, 
             *        after a completed simulation or after the abortion of a simulation run has no effect.
             *       
             *        Pressing this button when a simulation is active aborts it and resets all progress but keeps the last result.
             */
            inline void abort_button()
            {
                ImGui::PushID(2);
                ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.00f, 0.5f, 0.6f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.0f, 0.75f, 0.75f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.0f, 0.9f, 0.9f));
                if (ImGui::Button("abort", ImVec2(1.0/4 * ImGui::GetWindowSize().x*0.5f, 0.0f)) && simulation_control->is_simulation_active)                           
                {
                    simulation_control->is_paused = false;
                    simulation_control->is_simulation_active = false;
                    algorithm_handler->pause();
                    if(properties_gui->debug_mode)
                    {
                        algorithm_handler->block_until_finished();
                    }
                    progress->progress = 0;
                }    
                ImGui::PopStyleColor(3);
                ImGui::PopID();
            }

            /** @brief Creates the menu bar of the GUI. Its height is stored in the specified Windows object. */
            inline void menu_bar()
            {
                if (ImGui::BeginMainMenuBar()) 
                {
                    if (ImGui::BeginMenu("Windows"))
                    {
                        if (ImGui::MenuItem("Simulation status", NULL, windows->show_simulation_status)) 
                        { 
                            windows->show_simulation_status = !windows->show_simulation_status;
                        }
                        if (ImGui::MenuItem("Framerate", NULL, windows->show_framerate)) 
                        { 
                            windows->show_framerate = !windows->show_framerate;
                        }
                        if (ImGui::MenuItem("Properties", NULL, windows->show_properties)) 
                        {
                            windows->show_properties = !windows->show_properties;
                        }
                        if (ImGui::MenuItem("Density", NULL, windows->show_density)) 
                        { 
                            windows->show_density = !windows->show_density;
                        }
                        if (ImGui::MenuItem("Velocity", NULL, windows->show_velocity)) 
                        { 
                            windows->show_velocity = !windows->show_velocity;
                        }
                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Theme"))
                    {
                        if (ImGui::MenuItem("Light"))   { set_light_style(); }
                        if (ImGui::MenuItem("Dark"))    { set_dark_style(); }
                        if (ImGui::MenuItem("Classic")) { set_classic_style(); }
                        ImGui::EndMenu();
                    }
                }
                windows->menu_bar_size = ImGui::GetWindowHeight();
                ImGui::EndMainMenuBar();
            }

            /**
             * @brief   Creates a gray question mark showing a tooltip when hovered.
             * 
             * @param[in]   description     The description shown when hovering as a char array 
             *                              (or C-style string by calling `c_str()` on a `std::string`)
             */
            inline void help_marker(const char* description)
            {
                ImGui::TextDisabled("(?)");
                if (ImGui::BeginItemTooltip())
                {
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextUnformatted(description);
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
            }

// SELECTION FIELDS ///////////////////////////////////////////////////////////////////////////////////////////////////

            /** @brief Creates an ImGui Combo for the selection of the algorithm. */
            inline void algorithm_selection()
            {
                ImGui::SeparatorText("Algorithmic");
                bool is_selected;

                if (ImGui::BeginCombo("Algorithm",  properties_gui->algorithm.c_str()))
                {
                    for (int n = 0; n < lbm::gui::constants::algorithms.size(); n++)
                    {
                        is_selected = ( properties_gui->algorithm == lbm::gui::constants::algorithms[n]); 

                        if (ImGui::Selectable(lbm::gui::constants::algorithms[n].data(), is_selected))
                        {
                            properties_gui->algorithm = lbm::gui::constants::algorithms[n];
                            properties_changed = true;
                        }
                        if (is_selected) { ImGui::SetItemDefaultFocus(); }   
                    }
                    ImGui::EndCombo();
                }
            }

            /** @brief Creates an ImGui Combo for the selection of the data layout (and the corresponding access pattern). */
            inline void data_layout_selection()
            {
                if (ImGui::BeginCombo("Data layout",  properties_gui->data_layout.c_str())) 
                {
                    for (int n = 0; n < lbm::gui::constants::data_layouts.size(); n++)
                    {
                        bool is_selected = ( properties_gui->data_layout == lbm::gui::constants::data_layouts[n]); 
                        if (ImGui::Selectable(lbm::gui::constants::data_layouts[n].data(), is_selected))
                        {
                            properties_gui->data_layout = lbm::gui::constants::data_layouts[n];
                            properties_changed = true;
                        }
                        if (is_selected) { ImGui::SetItemDefaultFocus(); }      
                    }
                    ImGui::EndCombo();
                }
            }

            /** @brief Creates an ImGui Combo for the selection of the data layout (and the corresponding access pattern). */
            inline void scenario_selection()
            {
                if (ImGui::BeginCombo("Scenario", properties_gui->scenario.c_str())) 
                {
                    for (int n = 0; n < lbm::gui::constants::scenarios.size(); n++)
                    {
                        bool is_selected = ( properties_gui->scenario == lbm::gui::constants::scenarios[n]); 
                        if (ImGui::Selectable(lbm::gui::constants::scenarios[n].data(), is_selected))
                        {
                            properties_gui->scenario = lbm::gui::constants::scenarios[n];
                            properties_changed = true;
                        }
                        if (is_selected) { ImGui::SetItemDefaultFocus(); }      
                    }
                    ImGui::EndCombo();
                }
            }

            /**
             * @brief Creates a checkbox governing whether or not simulation results should be stored in a csv file
             *        and a text input field where the filename is specified.
             *        Up to 200 characters are allowed.
             */
            inline void save_results_selection()
            {
                if(ImGui::Checkbox("Save results", &simulation_control->results_to_csv))
                {
                    //results_to_csv = simulation_control->results_to_csv;
                }

                ImGui::SameLine();
                ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
                ImGui::SameLine();

                ImGui::InputTextWithHint(
                    "File name", "enter_name.csv", simulation_control->result_file_name, IM_ARRAYSIZE(simulation_control->result_file_name)
                );
            }

            /**
             * @brief Creates various input fields for properties regarding the simulation and its domain.
             *        The density and velocity plots are automatically adapting to new domain extents.
             *        Notice that any previous simulation results are replaced by fedault values when resizing.
             */
            inline void properties_simulation_and_domain()
            {
                ImGui::SeparatorText("Simulation and domain");

                if
                (
                    ImGui::InputUnsignedInt("Time steps", &(properties_gui->time_steps), 1, 10) |
                    ImGui::InputUnsignedInt("Horizontal nodes", &(properties_gui->horizontal_nodes), 10, 100) |
                    ImGui::InputUnsignedInt("Vertical nodes", &(properties_gui->vertical_nodes), 10, 100)
                )
                { properties_changed = true;}
            }

            /** @brief Creates input fields for various values related to general fluid properties. */
            inline void properties_fluid()
            {
                ImGui::SeparatorText("Fluid properties");

                if
                (
                    ImGui::InputDouble("Inlet density", &(properties_gui->inlet_density), 0.01f, 0.1f) |
                    ImGui::InputDouble("Outlet density", &(properties_gui->outlet_density), 0.01f, 0.1f) |
                    ImGui::InputDouble("Inlet velocity x", &(properties_gui->inlet_velocity_x), 0.01f, 0.1f) |
                    ImGui::InputDouble("Inlet velocity y", &(properties_gui->inlet_velocity_y), 0.01f, 0.1f) |
                    // Outlet velocity commented out on Jan 22, 2025 since it is currently without effect
                    // ImGui::InputDouble("Outlet velocity x", &(properties_gui->outlet_velocity_x), 0.01f, 0.1f) |
                    // ImGui::InputDouble("Outlet velocity y", &(properties_gui->outlet_velocity_y), 0.01f, 0.1f) |
                    ImGui::InputDouble("Relaxation time", &(properties_gui->relaxation_time), 0.01, 0.1)
                )
                { properties_changed = true; }
            }

// SIMULATION STATUS //////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief Provides information on the simulation status and displays the current progress using a progress bar.
             */
            inline void show_simulation_status()
            {
                if(!simulation_control->is_simulation_active)
                {
                    ImGui::Text("Ready to start simulation.");
                    ImGui::ProgressBar(0, ImVec2(0.975 * ImGui::GetWindowWidth(), 0.0f));
                }
                else if(simulation_control->is_paused)
                {
                    ImGui::Text("Simulation is paused.");
                    ImGui::ProgressBar(progress->progress, ImVec2(0.975 * ImGui::GetWindowWidth(), 0.0f));
                }
                else
                {
                    ImGui::Text("Simulation is running.");
                    ImGui::ProgressBar(progress->progress, ImVec2(0.975 * ImGui::GetWindowWidth(), 0.0f));
                }
            }

// COLORMAPS //////////////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief Creates an ImGui Combo for the selection of a colormap for the plot with the specified title.
             * 
             * @param colormap a reference to the currently selected colormap
             * @param plot_title the title of the plot for which the colormap is chosen
             */
            inline void colormap_picker(ImPlotColormap &colormap, const char *plot_title)
            {
                ImPlot::ColormapIcon(colormap);
                ImGui::SameLine();

                if (ImGui::BeginCombo("Colormap", ImPlot::GetColormapName(colormap)))
                {
                    bool is_selected;
                    for (int n = 0; n < constants::color_maps.size(); n++)
                    {
                        is_selected = (colormap == constants::color_maps.at(n)); 
                        ImPlot::ColormapIcon(constants::color_maps.at(n));
                        ImGui::SameLine();
                        if (ImGui::Selectable(ImPlot::GetColormapName(constants::color_maps.at(n)), is_selected))
                        {
                            colormap = constants::color_maps.at(n);
                            ImPlot::BustColorCache(plot_title);
                        }
                        if (is_selected) { ImGui::SetItemDefaultFocus(); }
                    }
                    ImGui::EndCombo();
                }
            }

            /**
            * @brief Adds the colormap used to distinguish solid and fluid nodes.
            *        It is meant to be used in a heatmap on top of another displaying the densities and velocities.
            *        Solid nodes are black regardless of the chosen color theme.
            */
            inline void add_solid_colormap()
            {
                ImVector<ImVec4> custom;
                custom.push_back(ImVec4(0, 0, 0, 1));
                custom.push_back(ImVec4(1, 1, 1, 0));
                ImPlot::AddColormap("SOLID_MASK", custom.Data, custom.Size, true);
            }

// RENDERING //////////////////////////////////////////////////////////////////////////////////////////////////////////

            /** @brief Creates the contexts for ImGui and ImPlot. */
            inline void create_contexts()
            {
                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                ImPlot::CreateContext();
            }

            /**
             * @brief Destroys the contexts of ImGui and ImPlot and prepares a graceful termination of the program.
             * 
             * @param window a pointer to the GLFW window that will be destroyed
             */
            inline void destroy()
            {
                ImGui_ImplOpenGL3_Shutdown();
                ImGui_ImplGlfw_Shutdown();
                ImPlot::DestroyContext();
                ImGui::DestroyContext();
                glfwDestroyWindow(glfw_window);
                glfwTerminate();
            }

            /**
            * @brief Renders the contents of the specified window.
            * 
            * @param window the window whose contents are drawn
            */
            inline void render()
            {
                ImGui::Render();
                glViewport(0, 0, monitor->display_width, monitor->display_height);
                glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                glfwSwapBuffers(glfw_window);
            }

// WINDOWS ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief Creates a window showing status information and means to control simulations.
             *        The window contains the following buttons from the buttons namespace: run, pause, and abort.
             *        Furthermore, it informs the user about whether a simulation is currently active and what its status is.
             *        It also shows the framerate at which new density and velocity plots are created. 
             *        The window can intentionally not be relocated.
             */
            inline void simulation_status_window()
            {
                if (windows->show_simulation_status)
                {
                    ImGui::SetNextWindowSize
                    (
                        { 
                            1 * monitor->viewport->WorkSize.x / 4, 
                            3 * monitor->viewport->WorkSize.y / 20
                        }
                    );
                    ImGui::SetNextWindowPos({0, windows->menu_bar_size});

                    if(ImGui::Begin("Simulation", &windows->show_simulation_status, ImGuiWindowFlags_NoResize))
                    {
                        ImGui::SeparatorText("Control");

                        run_button();
                        ImGui::SameLine();
                        pause_button();
                        ImGui::SameLine();
                        abort_button();
                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20, 0));
                        ImGui::SameLine();
                        ImGui::Checkbox("Live visualization", &windows->enable_live_visualization);

                        ImGui::SeparatorText("Status");
                        show_simulation_status();
                    }                        
                    ImGui::End();
                }
            }

            inline void framerate_window()
            {
                if(windows->show_framerate)
                {
                    ImGui::SetNextWindowSize
                    (
                        {
                            1 * monitor->viewport->WorkSize.x / 4, 
                            7 * monitor->viewport->WorkSize.y / 20
                        }
                    );

                    ImGui::SetNextWindowPos
                    (
                        {
                            0, 
                            windows->menu_bar_size + 3 * monitor->viewport->WorkSize.y / 20
                        }
                    );

                    if(ImGui::Begin("Framerate", &windows->show_simulation_status, ImGuiWindowFlags_NoResize))
                    {
                        ImGui::Text("Backend: %.1f FPS, %.3f ms/frame ", progress->framerate_backend, progress->frametime_backend);

                        ImGui::Text("Frontend: %.1f FPS, %.3f ms/frame ", progress->framerate_frontend, progress->frametime_frontend);

                        ImGui::SeparatorText("Backend framerate");
                        fps_buffer->current_time += ImGui::GetIO().DeltaTime;
                        fps_buffer->add(fps_buffer->current_time, progress->framerate_backend);

                        if (ImPlot::BeginPlot("##Scrolling", ImVec2(-1, ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y), ImPlotFlags_NoLegend|ImPlotFlags_NoMouseText)) 
                        {
                            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoTickLabels, 0);
                            ImPlot::SetupAxisLimits(ImAxis_X1, fps_buffer->current_time - 10, fps_buffer->current_time, ImGuiCond_Always);
                            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1000);
                            ImPlot::PlotLine("fps", &fps_buffer->data[0].x, &fps_buffer->data[0].y, fps_buffer->data.size(), 0, fps_buffer->offset, 2*sizeof(float));
                            ImPlot::EndPlot();
                        }
                    }                        
                    ImGui::End(); 
                }
            }

            /**
            * @brief Creates a window allowing to set various properties of the algorithm and the domain used in a simulation.
            *        The window can intentionally not be relocated.
            */
            inline void properties_window()
            {
                if (windows->show_properties)
                {    
                    ImGui::SetNextWindowSize
                    (
                        {
                            1 * monitor->viewport->WorkSize.x / 4, 
                            5 * monitor->viewport->WorkSize.y / 10
                        }
                    );

                    ImGui::SetNextWindowPos
                    (
                        {
                            0, 
                            windows->menu_bar_size + 5 * monitor->viewport->WorkSize.y / 10
                        }
                    );

                    if(ImGui::Begin("Properties", &windows->show_properties, ImGuiWindowFlags_NoResize))
                    {
                        ImGui::PushItemWidth(ImGui::GetWindowWidth() / 2);

                        ImGui::BeginDisabled(simulation_control->is_simulation_active);
                        
                        algorithm_selection();
                        data_layout_selection();
                        scenario_selection();
                        if
                        (
                            ImGui::InputUnsignedInt("Frame update interval", &(properties_gui->frame_update_interval), 1, 10)
                        )
                        { properties_changed = true;}

                        properties_simulation_and_domain();
                        properties_fluid();
            
                        ImGui::SeparatorText("");

                        ImGui::BeginDisabled(!properties_changed);
                        if (ImGui::Button("Undo changes", ImVec2(ImGui::GetWindowSize().x * 0.45f, 0)))
                        {
                            properties_gui.reset();
                            properties_gui = std::make_unique<core::Properties>(file_interaction::json_to_properties("../settings/settings.json", -2));
                            properties_changed = false;
                        }
                        ImGui::EndDisabled();  

                        ImGui::EndDisabled();   

                        if(simulation_control->is_simulation_active)
                        {
                            ImGui::Text("The properties of an active simulation cannot be changed.");
                        } 
                    }   
                    ImGui::End();                    
                }
            }

            /**
            * @brief Work in progress: Creates the window showing control options for simulation data read from a csv file.
            */
            void read_from_file_window()
            {
                if(windows->show_read_from_file_window)
                {
                    if(ImGui::Begin("Read from file", &windows->show_read_from_file_window))
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

            void debug_message()
            {
                if(properties_gui->debug_mode && windows->show_debug_window)
                {
                    ImGui::SetNextWindowSize
                    (
                        {
                            monitor->viewport->WorkSize.x / 3, 
                            0
                        }
                    );

                    ImGui::SetNextWindowPos
                    (
                        {
                            monitor->viewport->WorkSize.x / 3, 
                            windows->menu_bar_size + monitor->viewport->WorkSize.y / 3
                        }
                    );
                    
                    if(ImGui::Begin("Debug message", &windows->show_debug_window, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize))
                    {
                        ImGui::TextColored({255, 0, 0, 1}, "Debug mode detected.");
                        ImGui::TextWrapped(
                            "In debug mode, the absolutes of all velocity values and the density values are printed on the node. "
                            "The velocities can be retrieved through hovering over the according node as usual. "
                            "Thorough debug information is printed to the console, which has massive performance impact. "
                            "It is recommended to keep the lattice extents small for performance and readability reasons. "
                            "Please consider that the console output might be poorly aligned if the output is too large. "
                            "You can address this by copying the contents to an external reader and resizing accordingly. "
                            "When aborting a simulation, the GUI has to wait for the console to finish printing. "
                            "When running debug mode with a huge number of time steps, this can cause long waiting times. "
                            "Debug mode is not intended for performance measures. "
                        );
                        ImGui::NewLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowSize().x * 0.099f, 0.0f));
                        ImGui::SameLine();
                        if (ImGui::Button("okay", ImVec2(ImGui::GetWindowSize().x * 0.35f, 0)))
                        {
                            windows->show_debug_window = false;
                        }
                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowSize().x * 0.099f, 0.0f));
                        ImGui::SameLine();
                        if (ImGui::Button("disable", ImVec2(ImGui::GetWindowSize().x * 0.35f, 0)))
                        {
                            windows->show_debug_window = false;
                            properties_gui->debug_mode = false;
                            lbm::file_interaction::properties_to_json(*properties_gui);
                            windows->node_content = nullptr;
                        }
                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowSize().x * 0.099f, 0.0f));
                    }     
                    ImGui::End();
                }  
            }

            /**
            * @brief Creates the window visualizing the density of the fluid.
            *        The main component is a heatmap plot with an adjustable colormap.
            *        When hovering over the plot, the precise coordinates and the corresponding 
            *        density value are shown in a tooltip.
            * 
            *        The window can be resized and moved freely.
            * 
            *        Closing this window yields minor to medium performance improvements.
            * 
            */
            void density_window()
            {
                if(windows->show_density && windows->enable_live_visualization)
                {
                    ImGui::SetNextWindowPos
                    (
                        ImVec2
                        (
                            (1.0 / 4) * monitor->viewport->WorkSize.x, 
                            1.0 / 2 * windows->menu_bar_size + 1.0 / 2 * monitor->viewport->WorkSize.y
                        ), 
                        ImGuiCond_Appearing
                    ); 

                    ImGui::SetNextWindowSize
                    (
                        ImVec2
                        (
                            (3.0 / 4) * monitor->viewport->WorkSize.x, 
                            1.0 / 2 * monitor->viewport->WorkSize.y - 1.0 / 2 * windows->menu_bar_size
                        ), 
                        ImGuiCond_Appearing
                    );

                    if(ImGui::Begin("Density", &windows->show_density))
                    {
                        ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
                        colormap_picker(colormaps->density_colormap, "Density");
                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20, 0));
                        ImGui::SameLine();
                        ImGui::InputDouble("Lower scale", &colormaps->density_colormap_lower_scale, 0.001, 0.01);
                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20, 0));
                        ImGui::SameLine();
                        ImGui::InputDouble("Upper scale", &colormaps->density_colormap_upper_scale, 0.001, 0.01);
                        ImPlot::PushColormap(colormaps->density_colormap);

                        if
                        (
                            ImPlot::BeginPlot
                            (
                                "Density",
                                ImVec2
                                (
                                    ImGui::GetContentRegionAvail().x - monitor->monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
                                    ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                                ),
                                ImPlotFlags_NoLegend|ImPlotFlags_NoMouseText|ImPlotFlags_Crosshairs|ImPlotFlags_NoTitle
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
                                algorithm_handler->get_densities().data(),
                                algorithm_handler->get_vertical_nodes() - 2,
                                algorithm_handler->get_horizontal_nodes() - 2,
                                colormaps->density_colormap_lower_scale,
                                colormaps->density_colormap_upper_scale,
                                windows->node_content,
                                ImPlotPoint(0,0),
                                ImPlotPoint(algorithm_handler->get_horizontal_nodes() - 2, algorithm_handler->get_vertical_nodes() - 2)
                            );

                            ImPlot::PushColormap("SOLID_MASK");
                            ImPlot::PlotHeatmap
                            (
                                "Solid mask for density",
                                algorithm_handler->get_densities().data(),
                                algorithm_handler->get_vertical_nodes() - 2,
                                algorithm_handler->get_horizontal_nodes() - 2,
                                -1,
                                1, 
                                nullptr, 
                                ImPlotPoint(0,0),
                                ImPlotPoint(algorithm_handler->get_horizontal_nodes() - 2, algorithm_handler->get_vertical_nodes() - 2)
                            );

                            if (ImPlot::IsPlotHovered()) 
                            {
                                ImPlotPoint mouse = ImPlot::GetPlotMousePos();

                                if(
                                    0 <= mouse.x && mouse.x < (algorithm_handler->get_horizontal_nodes() - 2) 
                                    && 0 <= mouse.y && mouse.y < (algorithm_handler->get_vertical_nodes() - 2)
                                )
                                {
                                    ImGui::BeginTooltip();
                                    ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);
                                    double value = 
                                        algorithm_handler->get_densities().at((algorithm_handler->get_horizontal_nodes() - 2) * 
                                        ((int)floor(mouse.y)) + (int)floor(mouse.x));

                                    if(value != -1) { ImGui::Text("Density: %f", value); }
                                    else { ImGui::Text("Solid node"); }
                                    ImGui::EndTooltip();
                                }
                            }
                            ImPlot::EndPlot();
                        }
                        ImGui::SameLine();
                        ImPlot::ColormapScale
                        (
                            "Density",
                            colormaps->density_colormap_lower_scale,
                            colormaps->density_colormap_upper_scale, 
                            ImVec2
                            (
                                monitor->monitor_x_scale * 100,
                                ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                            ), 
                            "%g", 
                            ImPlotColormapScaleFlags_None, 
                            colormaps->density_colormap
                        );   
                    }
                    ImGui::End();
                }
            }

            /**
            * @brief Creates the window visualizing the velocity of the fluid.
            *        The main component is a heatmap plot with an adjustable colormap.
            *        When hovering over the plot, the precise coordinates and the corresponding 
            *        velocity values in x and y direction are shown in a tooltip.
            *        In addition to the heatmap plot, a quiver plot indicating the direction of the fluid velocity
            *        can be added. Enabling the quiver plot is recommended for small simulation domains or a zoomed
            *        section of larger domains.
            * 
            *        The window can be resized and moved freely.
            * 
            *        Closing this window yields medium to major performance improvements.
            * 
            */
            void velocity_window()
            {
                if(windows->show_velocity && windows->enable_live_visualization)
                {
                    ImGui::SetNextWindowPos
                    (
                        ImVec2
                        (
                            (1.0 / 4) * monitor->viewport->WorkSize.x, 
                            windows->menu_bar_size
                        ), 
                        ImGuiCond_Appearing
                    ); 

                    ImGui::SetNextWindowSize
                    (
                        ImVec2
                        (
                            (3.0 / 4) * monitor->viewport->WorkSize.x,
                            (1.0 / 2) * monitor->viewport->WorkSize.y - 1.0 / 2 * windows->menu_bar_size
                        ), 
                        ImGuiCond_Appearing
                    );

                    if(ImGui::Begin("Velocity", &windows->show_velocity))
                    {
                        ImGui::PushItemWidth(ImGui::GetWindowWidth() / 10);
                        colormap_picker(colormaps->velocity_colormap, "Velocity");

                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
                        ImGui::SameLine();
                        ImGui::InputDouble("Lower scale", &colormaps->velocity_colormap_lower_scale, 0.01, 0.1);
                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
                        ImGui::SameLine();
                        ImGui::InputDouble("Upper scale", &colormaps->velocity_colormap_upper_scale, 0.01, 0.1);
                        ImGui::SameLine();
                        ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
                        ImGui::SameLine();
                        if(ImGui::Checkbox("Enable vector plot", &windows->enable_velocity_quiver)){}

                        ImPlot::PushColormap(colormaps->velocity_colormap);

                        if
                        (
                            ImPlot::BeginPlot
                            (
                                "Velocity",
                                ImVec2
                                (
                                    ImGui::GetContentRegionAvail().x - monitor->monitor_x_scale * 100 - ImGui::GetStyle().ItemSpacing.x,
                                    ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                                ),
                                ImPlotFlags_NoLegend|ImPlotFlags_NoMouseText|ImPlotFlags_Crosshairs|ImPlotFlags_NoTitle
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
                                algorithm_handler->get_absolute_velocities().data(),
                                algorithm_handler->get_vertical_nodes() - 2,
                                algorithm_handler->get_horizontal_nodes() - 2,
                                colormaps->velocity_colormap_lower_scale, 
                                colormaps->velocity_colormap_upper_scale, 
                                windows->node_content,
                                ImPlotPoint(0,0),
                                ImPlotPoint(algorithm_handler->get_horizontal_nodes() - 2, algorithm_handler->get_vertical_nodes() - 2)
                            );

                            ImPlot::PopColormap(1);
                            ImPlot::PushColormap("SOLID_MASK");

                            ImPlot::PlotHeatmap
                            (
                                "Solid mask for velocity",
                                algorithm_handler->get_densities().data(),
                                algorithm_handler->get_vertical_nodes() - 2,
                                algorithm_handler->get_horizontal_nodes() - 2,
                                -1,
                                1, 
                                nullptr,
                                ImPlotPoint(0,0),
                                ImPlotPoint(algorithm_handler->get_horizontal_nodes() - 2, algorithm_handler->get_vertical_nodes() - 2)
                            );

                            ImPlot::PopColormap(1);
                            ImPlot::PushColormap(ImPlotColormap_Greys);
                            ImPlot::SetNextLineStyle(ImVec4(0, 0, 0, 0.5), 3.f);

                            if(windows->enable_velocity_quiver)
                            {
                                ImPlot::PlotLine
                                (
                                    "Quiver plot", 
                                    velocity_quiver_data->x_values->data(), 
                                    velocity_quiver_data->y_values->data(), 
                                    2 * algorithm_handler->get_absolute_velocities().size(), 
                                    ImPlotLineFlags_Segments
                                );
                            }

                            if(ImPlot::IsPlotHovered()) 
                            {
                                ImPlotPoint mouse = ImPlot::GetPlotMousePos();

                                if
                                (
                                    0 <= mouse.x && mouse.x < (algorithm_handler->get_horizontal_nodes() - 2) && 
                                    0 <= mouse.y && mouse.y < (algorithm_handler->get_vertical_nodes() - 2)
                                )
                                {
                                    ImGui::BeginTooltip();
                                    ImGui::Text("Coordinates: %.2f, %.2f", mouse.x, mouse.y);

                                    if
                                    (
                                        algorithm_handler->get_densities().at((algorithm_handler->get_horizontal_nodes() - 2) 
                                        * (int)floor(mouse.y) + (int)floor(mouse.x)) 
                                        != -1
                                    )
                                    {
                                        ImGui::Text(
                                            "x velocity: %.6f", 
                                            algorithm_handler->get_x_velocities().at((algorithm_handler->get_horizontal_nodes() - 2) * 
                                            ((int)floor(mouse.y)) + (int)floor(mouse.x))
                                        );
                                        ImGui::Text(
                                            "y velocity: %.6f", 
                                            algorithm_handler->get_y_velocities().at((algorithm_handler->get_horizontal_nodes() - 2) * 
                                            ((int)floor(mouse.y)) + (int)floor(mouse.x)));
                                    }
                                    else { ImGui::Text("Solid node");}
                                    ImGui::EndTooltip();
                                }
                            }

                            ImPlot::EndPlot();
                        }
                        ImGui::SameLine();
                        ImPlot::ColormapScale
                        (
                            "Velocity",
                            colormaps->velocity_colormap_lower_scale, 
                            colormaps->velocity_colormap_upper_scale, 
                            ImVec2
                            (
                                monitor->monitor_x_scale * 100,
                                ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y
                            ), 
                            "%g", 
                            ImPlotColormapScaleFlags_None, 
                            colormaps->velocity_colormap
                        );   
                    }
                    ImGui::End();
                }
            }


// PUBLIC API /////////////////////////////////////////////////////////////////////////////////////////////////////////

            public:

            explicit Gui(const std::string &&window_title)
            :
            simulation_control(std::make_unique<SimulationControl>()),
            progress(std::make_unique<Progress>()),
            colormaps(std::make_unique<Colormaps>()),
            properties_gui(std::make_unique<core::Properties>(lbm::file_interaction::json_to_properties("../settings/settings.json", -2))),
            windows(std::make_unique<Windows>(properties_gui->debug_mode)),
            velocity_quiver_data(std::make_unique<VelocityQuiverData>(2 * properties_gui->domain_node_count)),
            window_title(std::make_unique<std::string>(window_title)),
            properties_changed(false),
            fps_buffer(std::make_unique<FPSBuffer>(2000)),
            algorithm_handler(std::make_unique<A>())
            {
                glfwSetErrorCallback(glfw_error_callback);
                if (!glfwInit()) {throw exceptions::Exception("Failed to initialize GLFW.");}
                
                // GL 3.0 + GLSL 130
                const char* glsl_version = "#version 130";
                glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
                glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
                glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
                glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

                monitor = std::make_unique<Monitor>();

                glfw_window = glfwCreateWindow(monitor->video_mode->width, monitor->video_mode->height, window_title.c_str(), nullptr, nullptr);
                if (glfw_window == nullptr) {throw exceptions::Exception("Failed to create GLFW window.");}

                glfwMakeContextCurrent(glfw_window);
                glfwGetMonitorContentScale(monitor->primary_monitor, &(monitor->monitor_x_scale), &(monitor->monitor_x_scale));
                glfwGetFramebufferSize(glfw_window, &(monitor->display_width), &(monitor->display_height));

                // Setup Platform/Renderer backends
                create_contexts();
                ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);
                ImGui_ImplOpenGL3_Init(glsl_version);

                set_light_style();

                // IO
                ImGuiIO& io = ImGui::GetIO();
                io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
                io.Fonts->AddFontFromFileTTF("../fonts/DroidSans.ttf", 2 * sqrt(monitor->monitor_x_scale * monitor->monitor_x_scale) * 9.0f);

                // ImPlot style
                ImPlotStyle& implot_style = ImPlot::GetStyle();
                implot_style.PlotPadding = ImVec2(monitor->monitor_x_scale * 20, monitor->monitor_y_scale * 20);

                // ImPlot colormaps
                add_solid_colormap();
            };

            void run()
            {
                // Timer initializations
                SimpleTimer timer;
                SimpleTimer timer_framerate;

                // Eternal loop of imaging magic
                while (!glfwWindowShouldClose(glfw_window))
                {
                    glfwPollEvents();

                    // Start the Dear ImGui frame
                    ImGui_ImplOpenGL3_NewFrame();
                    ImGui_ImplGlfw_NewFrame();
                    ImGui::NewFrame();

                    monitor->viewport = std::make_unique<ImGuiViewport>(*ImGui::GetMainViewport());

                    menu_bar();

                    simulation_status_window();

                    framerate_window();

                    properties_window();

                    progress->progress = algorithm_handler->get_progress();
                    if(timer_framerate.elapsed() > 0.25)
                    {
                        
                        progress->frametime_backend = algorithm_handler->get_last_frametime();
                        progress->framerate_backend = 1 / (progress->frametime_backend * 0.001);

                        progress->frametime_frontend = ImGui::GetIO().DeltaTime * 1000.0;
                        progress->framerate_frontend = 1.0 / ImGui::GetIO().DeltaTime;
                        timer_framerate.restart();
                    }

                    if(simulation_control->is_simulation_active && !simulation_control->is_paused)
                    {
                        if(windows->enable_velocity_quiver)
                        {
                            unsigned int dnode_index = 0;
                            unsigned int node_index = 0;
                            unsigned int velocity_value_index = 0;
                            double base_x = 0;
                            double base_y = 0;
                            double offset_x = 0;
                            double offset_y = 0;

                            velocity_quiver_data->x_values->assign(2 * algorithm_handler->get_horizontal_nodes() * algorithm_handler->get_vertical_nodes(), 0);
                            velocity_quiver_data->y_values->assign(2 * algorithm_handler->get_horizontal_nodes() * algorithm_handler->get_vertical_nodes(), 0);

                            for(int y = 1; y < algorithm_handler->get_vertical_nodes() - 1; ++y)
                            {
                                for(int x = 1; x < algorithm_handler->get_horizontal_nodes() - 1; ++x)
                                {
                                    dnode_index = core::access::get_node_index(x-1, y-1, algorithm_handler->get_horizontal_nodes()-2);
                                    node_index = core::access::get_node_index(x, y, algorithm_handler->get_horizontal_nodes());

                                    velocity_value_index = 
                                        core::access::results::get_result_index(
                                            core::access::get_node_index(x, y, algorithm_handler->get_horizontal_nodes()), 
                                            algorithm_handler->get_horizontal_nodes()
                                        );

                                    if(algorithm_handler->get_absolute_velocities().at(velocity_value_index) > 1e-15)
                                    {
                                        base_x = 0.5 + x - 1;
                                        base_y = 0.5 + y - 1;
                                        
                                        (*velocity_quiver_data->x_values)[2 * dnode_index] = base_x;
                                        (*velocity_quiver_data->y_values)[2 * dnode_index] = base_y;

                                        offset_x = base_x + 0.5 * (1.0 / algorithm_handler->get_absolute_velocities().at(velocity_value_index)) 
                                                                * algorithm_handler->get_x_velocities().at(velocity_value_index);

                                        offset_y = base_y + 0.5 * (1.0 / algorithm_handler->get_absolute_velocities().at(velocity_value_index)) 
                                                                * algorithm_handler->get_y_velocities().at(velocity_value_index);

                                        (*velocity_quiver_data->x_values)[2 * dnode_index + 1] = offset_x;
                                        (*velocity_quiver_data->y_values)[2 * dnode_index + 1] = offset_y;
                                    }
                                }  
                            }
                        }

                        // Check if simulation is finished
                        if(algorithm_handler->is_finished())
                        {
                            progress->progress = 0;
                            simulation_control->is_simulation_active = false;
                            algorithm_handler->pause();
                        }
                    }

                    density_window();
                    velocity_window();
                    debug_message();
                    render();
                }

                algorithm_handler->pause();
                if(simulation_control->is_simulation_active) algorithm_handler->block_until_finished();
            
                destroy();
            }
        };

    } // ! namespace gui

} // ! namespace lbm

#endif // ! LBM_GUI_HPP
