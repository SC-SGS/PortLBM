/**
 * @file        lbm_gui.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains all declarations and some implementations of the GUI
 *              for the lattice Boltzmann simulation developed in my Bachelor thesis.
 * 
 * @version     2.0
 * 
 * @date        December 2024
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_GUI_HPP
#define LBM_GUI_HPP

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

// LBM
#include "simple_timer.hpp"
#include "gui_constants.hpp"
#include "../file_interaction/file_interaction.hpp"
#include "../execution/lbm_sycl_executor.hpp"
#include "../core/simulation.hpp"
#include "../exceptions/exceptions.hpp"

// Standard library
#include <memory>

namespace lbm
{

    namespace gui
    {

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
            explicit Windows();

            bool show_properties;
            bool show_simulation_status;
            bool show_read_from_file_window;
            bool show_density;
            bool show_velocity;
            bool enable_live_visualization;
            bool enable_velocity_quiver;
            float menu_bar_size;
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

            int current_iter;
            double progress;
            double framerate;
            double frametime;
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
            video_mode(nullptr),
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

        /**
         * @brief This structure contains all important data for the GUI.
         * 
         */
        struct Gui
        {
            std::unique_ptr<Windows> windows;
            std::unique_ptr<SimulationControl> simulation_control;
            std::unique_ptr<Progress> progress;
            std::unique_ptr<Monitor> monitor;
            std::unique_ptr<Colormaps> colormaps;
            std::unique_ptr<core::Properties> properties_stored;
            std::unique_ptr<core::Properties> properties_buffered;
            std::unique_ptr<VelocityQuiverData> velocity_quiver_data;
            std::unique_ptr<std::string> window_title;
            std::unique_ptr<execution::SYCLExecutor> executor;

            bool properties_changed;

            explicit Gui(const std::string &&window_title);

            void run();
        };

        /**
         * @brief This namespace contains helper functions for setting up the style of ImGui and ImPlot.
         */
        namespace styles
        {
            /**
             * @brief Sets the style of ImGui and ImPlot to light colors.
             */
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

            /**
             * @brief Sets the style of ImGui and ImPlot to dark colors.
             */
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

            /**
             * @brief Sets the style of ImGui and ImPlot to the classical ImGui style.
             */
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

        } // ! namespace styles

        /**
         * @brief This namespace contains various items used by the GUI.
         */
        namespace items
        {

            /**
             * @brief This namespace contains important buttons used to control the simulation.
             */
            namespace buttons
            {
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
                 * @param[in, out]  gui         reference to the structure containing all GUI data
                 * @param[in, out]  executor    a reference to the executor used to execute the simulation is set up with 
                 *                              an according default value
                 */
                inline void run_button
                (
                    gui::Gui &gui,
                    std::unique_ptr<execution::SYCLExecutor> &executor
                )
                {
                    ImGui::PushID(0);
                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.333f, 0.6f, 0.6f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.333f, 0.75f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.333f, 0.9f, 0.9f));

                    if (ImGui::Button("run", ImVec2(1.0/3 * ImGui::GetWindowSize().x*0.5f, 0.0f)))
                    {
                        if(!gui.simulation_control->is_simulation_active)
                        {
                            executor.reset();
                            executor = std::make_unique<execution::SYCLExecutor>();
                        }
                        gui.simulation_control->is_paused = false;
                        gui.simulation_control->is_simulation_active = true;
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
                 * @param[in, out]  gui reference to the structure containing all GUI data
                 */
                inline void pause_button(gui::Gui &gui)
                {
                    ImGui::PushID(1);
                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.166f, 0.6f, 0.6f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.166f, 0.75f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.166f, 0.9f, 0.9f));
                    if(ImGui::Button("pause", ImVec2(1.0/3 * ImGui::GetWindowSize().x*0.5f, 0.0f)))                           
                    {
                        gui.simulation_control->is_paused = true;
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::PopID();
                }

                /**
                 * @brief Creates an abort button that is capable of aborting an active simulation.
                 *        All progess counters are reset to zero.
                 *        Values are not re-initialized such that the last progress remains visible unless the user performs any changes.
                 *        TODO: Currently, simulation results are only written to a CSV file if the simulation executes entirely without
                 *        interruptions or crashes. In the future, simulation results may be written to a CSV file right when they are available.
                 *        
                 *        Pressing this button when no simulation is active, i.e., after the start of the program, 
                 *        after a completed simulation or after the abortion of a simulation run has no effect.
                 *       
                 *        Pressing this button when a simulation is active aborts it and resets all progress but keeps the last result.
                 * 
                 * @param[in, out]  gui reference to the structure containing all GUI data
                 */
                inline void abort_button(gui::Gui &gui)
                {
                    ImGui::PushID(2);
                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.00f, 0.6f, 0.6f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.0f, 0.75f, 0.75f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.0f, 0.9f, 0.9f));
                    if (ImGui::Button("abort", ImVec2(1.0/3 * ImGui::GetWindowSize().x*0.5f, 0.0f)) && gui.simulation_control->is_simulation_active)                           
                    {
                        gui.simulation_control->is_paused = false;
                        gui.simulation_control->is_simulation_active = false;
                        gui.progress->current_iter = 0;
                        gui.progress->progress = 0;
                    }    
                    ImGui::PopStyleColor(3);
                    ImGui::PopID();
                }

            } // ! namespace buttons


            /**
             * @brief   Creates the menu bar of the GUI.
             *          Its height is stored in the specified Windows object.
             * 
             * @param[in, out]  gui reference to the structure containing all GUI data
             */
            inline void menu_bar(gui::Gui &gui)
            {
                if (ImGui::BeginMainMenuBar()) 
                {
                    if (ImGui::BeginMenu("Windows"))
                    {
                        if (ImGui::MenuItem("Properties", NULL, gui.windows->show_properties)) 
                        {
                            gui.windows->show_properties = !gui.windows->show_properties;
                        }
                        if (ImGui::MenuItem("Simulation status", NULL, gui.windows->show_simulation_status)) 
                        { 
                            gui.windows->show_simulation_status = !gui.windows->show_simulation_status;
                        }
                        if (ImGui::MenuItem("Density", NULL, gui.windows->show_density)) 
                        { 
                            gui.windows->show_density = !gui.windows->show_density;
                        }
                        if (ImGui::MenuItem("Velocity", NULL, gui.windows->show_velocity)) 
                        { 
                            gui.windows->show_velocity = !gui.windows->show_velocity;
                        }
                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Theme"))
                    {
                        if (ImGui::MenuItem("Light")) 
                        {
                            styles::set_light_style();
                        }
                        if (ImGui::MenuItem("Dark")) 
                        { 
                            styles::set_dark_style();
                        }
                        if (ImGui::MenuItem("Classic")) 
                        { 
                            styles::set_classic_style();
                        }
                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Load from file"))
                    {
                        if (ImGui::MenuItem("Show window", NULL, gui.windows->show_read_from_file_window)) 
                        {
                            gui.windows->show_read_from_file_window = !gui.windows->show_read_from_file_window;
                        }
                        ImGui::EndMenu();
                    }
                }
                gui.windows->menu_bar_size = ImGui::GetWindowHeight();
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

            /**
             * @brief Creates an ImGui Combo for the selection of the algorithm.
             * 
             * @param settings a reference to the settings object responsible for setting up all necessary simulation properties
             * @param gui_algorithmic a reference to an object containing all data specifying the algorithm used for the simulation
             */
            inline void algorithm_selection
            (
                core::Properties &properties_buffer,
                bool &changed
            )
            {
                ImGui::SeparatorText("Algorithmic");
                bool is_selected;

                if (ImGui::BeginCombo("Algorithm", properties_buffer.algorithm.c_str()))
                {
                    for (int n = 0; n < lbm::gui::constants::algorithms.size(); n++)
                    {
                        is_selected = (properties_buffer.algorithm == lbm::gui::constants::algorithms[n]); 

                        if (ImGui::Selectable(lbm::gui::constants::algorithms[n].data(), is_selected))
                        {
                            properties_buffer.algorithm = lbm::gui::constants::algorithms[n];
                            changed = true;
                        }
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus(); 
                        }   
                    }
                    ImGui::EndCombo();
                }
            }

            /**
             * @brief Creates an ImGui Combo for the selection of the data layout (and the corresponding access pattern).
             * 
             * @param settings a reference to the settings object responsible for setting up all necessary simulation properties
             * @param gui_algorithmic a reference to an object containing all data specifying the algorithm used for the simulation
             */
            inline void data_layout_selection
            (
                core::Properties &properties_buffer,
                bool &changed
            )
            {
                if (ImGui::BeginCombo("Data layout", properties_buffer.data_layout.c_str())) 
                {
                    for (int n = 0; n < lbm::gui::constants::data_layouts.size(); n++)
                    {
                        bool is_selected = (properties_buffer.data_layout == lbm::gui::constants::data_layouts[n]); 
                        if (ImGui::Selectable(lbm::gui::constants::data_layouts[n].data(), is_selected))
                        {
                            properties_buffer.data_layout = lbm::gui::constants::data_layouts[n];
                            changed = true;
                        }
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }      
                    }
                    ImGui::EndCombo();
                }
            }

            /**
             * @brief Creates a checkbox governing whether or not simulation results should be stored in a csv file
             *        and a text input field where the filename is specified.
             *        Up to 200 characters are allowed.
             * 
             * @param settings a reference to the settings object responsible for setting up all necessary simulation properties
             * @param gui.simulation_control a reference to an object responsible for tracking control data for the simulation 
             * @param gui_simulation_data a reference to an object containing all data on which the simulation operates
             */
            inline void save_results_selection
            (
                bool &results_to_csv,
                gui::Gui &gui
            )
            {
                if(ImGui::Checkbox("Save results", &gui.simulation_control->results_to_csv))
                {
                    results_to_csv = gui.simulation_control->results_to_csv;
                }

                ImGui::SameLine();
                ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
                ImGui::SameLine();

                ImGui::InputTextWithHint("File name", "enter_name.csv", gui.simulation_control->result_file_name, IM_ARRAYSIZE(gui.simulation_control->result_file_name));
            }

            /**
             * @brief Creates various input fields for properties regarding the simulation and its domain.
             *        The density and velocity plots are automatically adapting to new domain extents.
             *        
             *        Notice that any previous simulation results are replaced by fedault values when resizing.
             * 
             *        TODO: Notice that the current specification does not entirely prevent faulty setups.
             * 
             * @param settings a reference to the settings object responsible for setting up all necessary simulation properties
             * @param current_sim_data a reference to the tuple storing the current simulation results
             */
            inline void properties_simulation_and_domain(core::Properties &properties_buffer, bool &changed)
            {
                ImGui::SeparatorText("Simulation and domain");

                if
                (
                    ImGui::InputUnsignedInt("Time steps", &(properties_buffer.time_steps), 1, 10) |
                    ImGui::InputUnsignedInt("Horizontal nodes", &(properties_buffer.horizontal_nodes), 10, 100) |
                    ImGui::InputUnsignedInt("Vertical nodes", &(properties_buffer.vertical_nodes), 10, 100)
                )
                {
                    changed = true;
                }
            }

            /**
             * @brief Creates input fields for various values related to general fluid properties.
             * 
             * @param settings a reference to the settings object responsible for setting up all necessary simulation properties 
             */
            inline void properties_fluid(core::Properties &properties_buffer, bool &changed)
            {
                ImGui::SeparatorText("Fluid properties");

                if
                (
                    ImGui::InputDouble("Inlet density", &(properties_buffer.inlet_density), 0.01f, 0.1f) |
                    ImGui::InputDouble("Outlet density", &(properties_buffer.outlet_density), 0.01f, 0.1f) |
                    ImGui::InputDouble("Inlet velocity x", &(properties_buffer.inlet_velocity_x), 0.01f, 0.1f) |
                    ImGui::InputDouble("Inlet velocity y", &(properties_buffer.inlet_velocity_y), 0.01f, 0.1f) |
                    ImGui::InputDouble("Outlet velocity x", &(properties_buffer.outlet_velocity_x), 0.01f, 0.1f) |
                    ImGui::InputDouble("Outlet velocity y", &(properties_buffer.outlet_velocity_y), 0.01f, 0.1f) |
                    ImGui::InputDouble("Relaxation time", &(properties_buffer.relaxation_time), 0.01, 0.1)
                )
                {
                    changed = true;
                }
            }

            /**
             * @brief Provides information on the simulation status and displays the current progress using a progress bar.
             * 
             * @param gui.simulation_control a reference to an object responsible for tracking control data for the simulation  
             * @param gui_progress reference to an object containing all data on the current progress of a simulation
             */
            inline void show_simulation_status(gui::Gui &gui)
            {
                if(!gui.simulation_control->is_simulation_active)
                {
                    ImGui::Text("Ready to start simulation.");
                    ImGui::ProgressBar(0.0, ImVec2(0.9 * ImGui::GetWindowWidth(), 0.0f));
                }
                else if(gui.simulation_control->is_paused)
                {
                    ImGui::Text("Simulation is paused.");
                    ImGui::ProgressBar(gui.progress->progress, ImVec2(0.9 * ImGui::GetWindowWidth(), 0.0f));
                }
                else
                {
                    ImGui::Text("Simulation is running.");
                    ImGui::ProgressBar(gui.progress->progress, ImVec2(0.9 * ImGui::GetWindowWidth(), 0.0f));
                }
            }

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
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus(); 

                        }
                    }
                    ImGui::EndCombo();
                }
            }

        } // ! namespace items

        /**
         * @brief This namespace contains helper methods for interactions with contexts.
         * 
         */
        namespace contexts
        {
            /**
             * @brief Creates the contexts for ImGui and ImPlot.
             */
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
            inline void destroy(GLFWwindow *window)
            {
                ImGui_ImplOpenGL3_Shutdown();
                ImGui_ImplGlfw_Shutdown();
                ImPlot::DestroyContext();
                ImGui::DestroyContext();
                glfwDestroyWindow(window);
                glfwTerminate();
            }
        } // ! namespace contexts

        /**
         * @brief This namespace contains the specification of all windows that can be shown.
         */
        namespace windows
        {
            /**
             * @brief Creates a window showing status information and means to control simulations.
             *        The window contains the following buttons from the buttons namespace: run, pause, and abort.
             *        Furthermore, it informs the user about whether a simulation is currently active and what its status is.
             *        It also shows the framerate at which new density and velocity plots are created. 
             *        The window can intentionally not be relocated.
             * 
             * @param executor a reference to the executor used to execute the simulation
             */
            inline void simulation_status_window
            (
                gui::Gui &gui,
                std::unique_ptr<execution::SYCLExecutor> &executor
            )
            {
                if (gui.windows->show_simulation_status)
                {
                    ImGui::SetNextWindowSize
                    (
                        { 
                            1 * gui.monitor->viewport->WorkSize.x / 4, 
                            gui.monitor->viewport->WorkSize.y / 5
                        }
                    );
                    ImGui::SetNextWindowPos({0, gui.windows->menu_bar_size});

                    if(ImGui::Begin("Simulation", &gui.windows->show_simulation_status, ImGuiWindowFlags_NoResize))
                    {
                        ImGui::SeparatorText("Control");

                        items::buttons::run_button(gui, executor);
                        ImGui::SameLine();
                        items::buttons::pause_button(gui);
                        ImGui::SameLine();
                        items::buttons::abort_button(gui);

                        ImGui::SeparatorText("Status");
                        items::show_simulation_status(gui);

                        ImGui::SeparatorText("Framerate");
                        ImGui::Text("%.1f FPS, %.3f ms/frame ", gui.progress->framerate, gui.progress->frametime);
                    }                        
                    ImGui::End();
                }
            }

            /**
            * @brief Creates a window allowing to set various properties of the algorithm and the domain used in a simulation.
            *        The window can intentionally not be relocated.
            * 
            * @param gui.monitor a constant reference to an object storing data related to the primary monitor and the viewport
            */
            inline void properties_window
            (
                std::unique_ptr<execution::SYCLExecutor> &executor,
                std::unique_ptr<core::Properties> &properties_buffer,
                bool &changed,
                gui::Gui &gui
            )
            {
                if (gui.windows->show_properties)
                {    
                    ImGui::SetNextWindowSize
                    (
                        {
                            1 * gui.monitor->viewport->WorkSize.x / 4, 
                            4 * gui.monitor->viewport->WorkSize.y / 5
                        }
                    );

                    ImGui::SetNextWindowPos
                    (
                        {
                            0, 
                            gui.windows->menu_bar_size + gui.monitor->viewport->WorkSize.y / 5
                        }
                    );

                    if(ImGui::Begin("Properties", &gui.windows->show_properties, ImGuiWindowFlags_NoResize))
                    {
                        ImGui::PushItemWidth(ImGui::GetWindowWidth() / 2);

                        ImGui::BeginDisabled(gui.simulation_control->is_simulation_active);
                        
                        items::algorithm_selection(*properties_buffer, changed);
                        items::data_layout_selection(*properties_buffer, changed);
                        items::save_results_selection(properties_buffer->results_to_csv, gui);
                        ImGui::Checkbox("Live visualization", &gui.windows->enable_live_visualization);
                        items::properties_simulation_and_domain(*properties_buffer, changed);
                        items::properties_fluid(*properties_buffer, changed);
            
                        ImGui::SeparatorText("");
                        if (ImGui::Button("Create configuration file", ImVec2(ImGui::GetWindowSize().x * 0.45f, 0)))
                        {
                            //properties_buffer->to_json();
                            file_interaction::properties_to_json(*properties_buffer);
                            changed = false;
                            //initialize_executor(properties_buffer->algorithm, properties_buffer->data_layout, executor);
                            executor = std::make_unique<execution::SYCLExecutor>();
                        }

                        ImGui::BeginDisabled(!changed);
                        if (ImGui::Button("Undo changes", ImVec2(ImGui::GetWindowSize().x * 0.45f, 0)))
                        {
                            //properties_buffer = std::make_unique<core::Properties>(*executor->properties);
                        }
                        ImGui::EndDisabled();  

                        ImGui::EndDisabled();   

                        if(gui.simulation_control->is_simulation_active)
                        {
                            ImGui::Text("The properties of an active simulation cannot be changed.");
                        } 
                    }   
                    ImGui::End();                    
                }
            }

            /**
            * @brief Work in progress: Creates the window showing control options for simulation data read from a csv file.
            * 
            * @param window_settings 
            */
            void read_from_file_window(gui::Gui &gui);

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
            void density_window
            (
                const execution::SYCLExecutor &sycl_executor,
                gui::Gui &gui
            );

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
            void velocity_window
            (
                const execution::SYCLExecutor &sycl_executor,
                gui::Gui &gui
            );

        } // ! namespace windows

        /**
        * @brief This namespace contains helper methods related to OpenGL and GLFW.
        * 
        */
        namespace rendering
        {

            /**
            * @brief Helper function displaying an error callback message on the console.
            * 
            * @param error the integer error id
            * @param description char array containing an error message
            */
            inline void glfw_error_callback(int error, const char* description)
            {
                fprintf(stderr, "GLFW Error %d: %s\n", error, description);
            }

            /**
            * @brief Renders the contents of the specified window.
            * 
            * @param window the window whose contents are drawn
            * @param gui.monitor the primary monitor on which the window is shown
            */
            inline void render(GLFWwindow *window, gui::Gui &gui)
            {
                ImGui::Render();
                glViewport(0, 0, gui.monitor->display_width, gui.monitor->display_height);
                glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                glfwSwapBuffers(window);
            }
        } // ! namespace rendering

        /**
        * @brief Utility namespace of helper functions that were found to be unsuited elsewhere
        * 
        */
        namespace misc
        {
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
        } // ! namespace misc

    } // ! namespace gui

} // ! namespace lbm

#endif
