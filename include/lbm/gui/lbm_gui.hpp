/**
 * @file        lbm_gui.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains all declarations and some implementations of the GUI
 *              for the lattice Boltzmann simulation developed in my Bachelor thesis.
 *              The visualization framework was initially designed with the help of the HPX implementation
 *              from my SimTech project work (https://github.com/MarcelGraf0710/Task-based-Lattice-Boltzmann).
 *              However, it was designed to be as independent as possible from it such that it can easily be
 *              reused in the GPU-based lattice Boltzmann simulation or any other implementation with similar
 *              data structures. Nevertheless, some methods may require adaptions in the future to suit the
 *              needs of the individual implementation. Any interaction with the GUI is supposed to be carried out
 *              through Executors inheriting from the equally named class defined in executor.hpp.
 * 
 * @version     1.0
 * 
 * @date        2024-08-28
 * 
 * @copyright   Copyright (c) 2024
 * 
 */

#ifndef LBM_GUI_HPP
#define LBM_GUI_HPP

#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"

#include "../implot/implot.h"
#include "../implot/implot_internal.h"

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>

#include "../include/file_interaction.hpp"
#include "../include/parallel_two_lattice.hpp"
#include "../include/simulation.hpp"

#include "executor.hpp"
#include "simple_timer.hpp"

#include <memory>

/**
 * @brief Convenience typedef to avoid ambiguous names.
 * 
 */
typedef access_function access_function_t;

/**
 * @brief Convenience typedef to avoid ambiguous names.
 * 
 */
typedef border_swap_information border_swap_information_t;

/**
 * @brief An array containing all default colormaps offered by ImPlot.
 */
const ImPlotColormap_ COLOR_MAPS [16] = 
{
    ImPlotColormap_Deep, ImPlotColormap_Dark, ImPlotColormap_Pastel, ImPlotColormap_Paired, ImPlotColormap_Viridis, ImPlotColormap_Plasma,
    ImPlotColormap_Hot, ImPlotColormap_Cool, ImPlotColormap_Pink, ImPlotColormap_Jet, ImPlotColormap_Twilight, ImPlotColormap_RdBu, ImPlotColormap_BrBG,
    ImPlotColormap_PiYG, ImPlotColormap_Spectral, ImPlotColormap_Greys
};

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
struct GuiWindows
{
    explicit GuiWindows();

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
 * 
 */
struct GuiSimulationControl
{
    explicit GuiSimulationControl();

    bool is_paused;
    bool is_simulation_active;
    bool results_to_csv;
    char result_file_name [200];
};

/**
 * @brief This struct keeps track of the progress of the simulation and the framerate.
 * 
 */
struct GuiProgress
{
    explicit GuiProgress();

    int current_iter;
    double progress;
    double framerate;
    double frametime;
};

/**
 * @brief This struct contains:
 * 
 *        - all data on which the simulation operates
 *        
 *        - the name of the csv file to which simulation results are written
 * 
 *        - a vector buffering the simulation results of all time steps
 * 
 *        - the result of the current iteration (or a default value)
 */
struct GuiSimulationData
{
    explicit GuiSimulationData();

    std::vector<sim_data_tuple> all_results; 
    sim_data_tuple current_result;
    char result_file_name[200];
    bool results_to_csv;
    std::vector<double> distribution_values_0;
    std::vector<double> distribution_values_1;
    std::vector<unsigned int> nodes;
    std::vector<unsigned int> fluid_nodes;
    std::vector<bool> phase_information;
    border_swap_information_t border_swap_information;
    access_function_t access_function;
    std::shared_ptr<std::vector<double>> x_velocities;
    std::shared_ptr<std::vector<double>> y_velocities;
    std::shared_ptr<std::vector<double>> absolute_velocities;
};

/**
 * @brief This struct contains information on the primary monitor and the viewport.
 * 
 */
struct GuiMonitor
{
    explicit GuiMonitor();

    GLFWmonitor *primary_monitor;
    const GLFWvidmode *video_mode;
    float video_mode_width;
    float video_mode_height;
    float monitor_x_scale;
    float monitor_y_scale;
    int display_width;
    int display_height;
    ImGuiViewport viewport;
    ImVec2 viewport_work_size;
};

/**
 * @brief This struct contains information for setting up the algorithm.
 * 
 */
struct GuiAlgorithmic
{
    explicit GuiAlgorithmic();

    const std::vector<std::string> &algorithms;
    std::string current_algorithm;
    const std::vector<std::string> &data_layouts;
    std::string current_data_layout;
};

/**
 * @brief This struct contains information on the colormaps used in the density and velocity plot.
 * 
 */
struct GuiColormaps
{
    explicit GuiColormaps();

    ImPlotColormap density_colormap;
    double density_colormap_lower_scale;
    double density_colormap_upper_scale;
    ImPlotColormap velocity_colormap;
    double velocity_colormap_lower_scale;
    double velocity_colormap_upper_scale;
};

/**
 * @brief This struct contains the x and y values of the velocity quiver.
 * 
 */
struct GuiVelocityQuiverData
{
    explicit GuiVelocityQuiverData();

    std::unique_ptr<std::vector<double>> x_values;
    std::unique_ptr<std::vector<double>> y_values;
};





struct GuiVisualizationData
{
    explicit GuiVisualizationData();
    int vertical_nodes;
    int horizontal_nodes;
    std::shared_ptr<std::vector<double>> density_values;
    std::shared_ptr<std::vector<double>> absolute_velocity_values;
    std::shared_ptr<std::vector<double>> x_velocity_values;
    std::shared_ptr<std::vector<double>> y_velocity_values;
};







/**
 * @brief This namespace contains helper functions for setting up the style of ImGui and ImPlot.
 * 
 */
namespace gui_styles
{
    /**
     * @brief Sets the style of ImGui and ImPlot to light colors.
     */
    inline void set_light_style()
    {
        ImGui::StyleColorsLight();
        ImVec4 clear_color = ImVec4(0.7f, 0.7f, 0.7f, 1.00f);
        glClearColor(
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
        glClearColor(
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
        glClearColor(
            clear_color.x * clear_color.w, 
            clear_color.y * clear_color.w, 
            clear_color.z * clear_color.w, 
            clear_color.w
        );
        glClear(GL_COLOR_BUFFER_BIT);
        ImPlot::StyleColorsClassic();
    }
}

/**
 * @brief This namespace contains various items used by the GUI.
 */
namespace gui_items
{

    /**
     * @brief This namespace contains important buttons used to control the simulation.
     */
    namespace gui_buttons
    {
        /**
         * @brief Creates a run button that is capable of launching a new simulation or resuming a paused simulation.
         *        Simulation values are initialized with placeholder values.
         *        
         *        Pressing this button when no simulation is active, i.e., after the start of the program, 
         *        after a completed simulation or after the abortion of a simulation run enacts the launch of a new simulation run.
         *       
         *        Pressing this button when a simulation is active but paused resumes work on it.
         * 
         *        Pressing this button when a simulation is active and not paused has no effect.
         * 
         * @param settings a constant reference to the settings object responsible for setting up all necessary simulation properties
         * @param gui_simulation_control a reference to an object responsible for tracking control data for the simulation
         * @param gui_simulation_data a reference to an object containing all data on which the simulation operates
         * @param executor a reference to the executor used to execute the simulation is set up with an according default value
         */
        template <typename ContainerType, typename ResultsType>
        inline void run_button
        (
            const Settings &settings,
            GuiSimulationControl &gui_simulation_control,
            ContainerType &container_data,
            Executor<ContainerType, ResultsType> &executor
        )
        {
            ImGui::PushID(0);
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.333f, 0.6f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.333f, 0.75f, 0.75f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.333f, 0.9f, 0.9f));

            if (ImGui::Button("run", ImVec2(1.0/3 * ImGui::GetWindowSize().x*0.5f, 0.0f)))
            {
                if(!gui_simulation_control.is_simulation_active)
                {
                    setup_global_variables(settings);
                    executor.initialize(container_data);
                }
                gui_simulation_control.is_paused = false;
                gui_simulation_control.is_simulation_active = true;
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
         * @param gui_simulation_control a reference to an object responsible for tracking control data for the simulation
         */
        inline void pause_button
        (
            GuiSimulationControl &gui_simulation_control
        )
        {
            ImGui::PushID(1);
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.166f, 0.6f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.166f, 0.75f, 0.75f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.166f, 0.9f, 0.9f));
            if(ImGui::Button("pause", ImVec2(1.0/3 * ImGui::GetWindowSize().x*0.5f, 0.0f)))                           
            {
                gui_simulation_control.is_paused = true;
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
         * @param gui_simulation_control a reference to an object responsible for tracking control data for the simulation
         * @param gui_progress reference to an object containing all data on the current progress of a simulation
         */
        inline void abort_button
        (
            GuiSimulationControl &gui_simulation_control,
            GuiProgress &gui_progess
        )
        {
            ImGui::PushID(2);
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.00f, 0.6f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.0f, 0.75f, 0.75f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.0f, 0.9f, 0.9f));
            if (ImGui::Button("abort", ImVec2(1.0/3 * ImGui::GetWindowSize().x*0.5f, 0.0f)))                           
            {
                gui_simulation_control.is_paused = false;
                gui_simulation_control.is_simulation_active = false;
                gui_progess.current_iter = 0;
                gui_progess.progress = 0;
            }    
            ImGui::PopStyleColor(3);
            ImGui::PopID();
        }
    }
    /**
     * @brief Creates the menu bar of the GUI.
     *        Its height is stored in the specified GuiWindows object.
     * 
     * @param window_config a struct containing all information on the GUI windows
     */
    inline void menu_bar(GuiWindows &window_config)
    {
        if (ImGui::BeginMainMenuBar()) 
        {
            if (ImGui::BeginMenu("Windows"))
            {
                if (ImGui::MenuItem("Properties", NULL, window_config.show_properties)) 
                {
                    window_config.show_properties = !window_config.show_properties;
                }
                if (ImGui::MenuItem("Simulation status", NULL, window_config.show_simulation_status)) 
                { 
                    window_config.show_simulation_status = !window_config.show_simulation_status;
                }
                if (ImGui::MenuItem("Density", NULL, window_config.show_density)) 
                { 
                    window_config.show_density = !window_config.show_density;
                }
                if (ImGui::MenuItem("Velocity", NULL, window_config.show_velocity)) 
                { 
                    window_config.show_velocity = !window_config.show_velocity;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Theme"))
            {
                if (ImGui::MenuItem("Light")) 
                {
                    gui_styles::set_light_style();
                }
                if (ImGui::MenuItem("Dark")) 
                { 
                    gui_styles::set_dark_style();
                }
                if (ImGui::MenuItem("Classic")) 
                { 
                    gui_styles::set_classic_style();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Load from file"))
            {
                if (ImGui::MenuItem("Show window", NULL, window_config.show_read_from_file_window)) 
                {
                    window_config.show_read_from_file_window = !window_config.show_read_from_file_window;
                }
                ImGui::EndMenu();
            }
        }
        window_config.menu_bar_size = ImGui::GetWindowHeight();
        ImGui::EndMainMenuBar();
    }

    /**
     * @brief Creates a gray question mark showing a tooltip when hovered.
     * 
     * @param description The description shown when hovering as a char array 
     *                    (or C-style string by calling c_str() on a std::string)
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
        Settings &settings,
        GuiAlgorithmic &gui_algorithmic
    )
    {
        ImGui::SeparatorText("Algorithmic");

        if (ImGui::BeginCombo("Algorithm", gui_algorithmic.current_algorithm.c_str()))
        {
            for (int n = 0; n < size(gui_algorithmic.algorithms); n++)
            {
                bool is_selected = (gui_algorithmic.current_algorithm == gui_algorithmic.algorithms[n]); 

                if (ImGui::Selectable(gui_algorithmic.algorithms[n].c_str(), is_selected))
                {
                    gui_algorithmic.current_algorithm = gui_algorithmic.algorithms[n];
                    settings.algorithm = gui_algorithmic.current_algorithm;
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
        Settings &settings,
        GuiAlgorithmic &gui_algorithmic
    )
    {
        if (ImGui::BeginCombo("Data layout", gui_algorithmic.current_data_layout.c_str())) 
        {
            for (int n = 0; n < size(gui_algorithmic.data_layouts); n++)
            {
                bool is_selected = (gui_algorithmic.current_data_layout == gui_algorithmic.data_layouts[n]); 
                if (ImGui::Selectable(gui_algorithmic.data_layouts[n].c_str(), is_selected))
                {
                    gui_algorithmic.current_data_layout = gui_algorithmic.data_layouts[n];
                    settings.access_pattern = gui_algorithmic.current_data_layout;
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
     * @param gui_simulation_control a reference to an object responsible for tracking control data for the simulation 
     * @param gui_simulation_data a reference to an object containing all data on which the simulation operates
     */
    inline void save_results_selection
    (
        int &results_to_csv,
        GuiSimulationControl &gui_simulation_control
    )
    {
        if(ImGui::Checkbox("Save results", &gui_simulation_control.results_to_csv))
        {
            results_to_csv = gui_simulation_control.results_to_csv ? 1 : 0;
        }

        ImGui::SameLine();
        ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() / 20,0));
        ImGui::SameLine();

        ImGui::InputTextWithHint("File name", "enter_name.csv", gui_simulation_control.result_file_name, IM_ARRAYSIZE(gui_simulation_control.result_file_name));
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
    inline void properties_simulation_and_domain(Settings &settings, GuiVisualizationData &gui_visualization_data)
    {
        ImGui::SeparatorText("Simulation and domain");

        if (ImGui::InputInt("Time steps", &(settings.time_steps), 1, 10))
        {
            if (settings.time_steps < 1)
            {
                settings.time_steps = 1;
            }
        }
        if(ImGui::InputInt("Horizontal nodes", &(settings.horizontal_nodes), 10, 100))
        {
            if (settings.horizontal_nodes < 0)
            {
                settings.horizontal_nodes = 0;
            }
            settings.total_node_count = settings.horizontal_nodes * settings.vertical_nodes;
            gui_visualization_data.horizontal_nodes = settings.horizontal_nodes;
            gui_visualization_data.density_values->assign(settings.total_node_count, 0);
            gui_visualization_data.absolute_velocity_values->assign(settings.total_node_count, 0);
            gui_visualization_data.x_velocity_values->assign(settings.total_node_count, 0);
            gui_visualization_data.y_velocity_values->assign(settings.total_node_count, 0);
        }
        if(ImGui::InputInt("Vertical nodes", &(settings.vertical_nodes), 10, 100))
        {
            if (settings.vertical_nodes < 0)
            {
                settings.vertical_nodes = 0;
            }
            settings.total_node_count = settings.horizontal_nodes * settings.vertical_nodes;
            gui_visualization_data.vertical_nodes = settings.vertical_nodes;
            gui_visualization_data.density_values->assign(settings.total_node_count, 0);
            gui_visualization_data.absolute_velocity_values->assign(settings.total_node_count, 0);
            gui_visualization_data.x_velocity_values->assign(settings.total_node_count, 0);
            gui_visualization_data.y_velocity_values->assign(settings.total_node_count, 0);

        }
        ImGui::InputInt("Vertical nodes excluding buffers", &(settings.vertical_nodes_excluding_buffers), 10, 100);
        ImGui::InputInt("Total nodes excluding buffers", &(settings.total_nodes_excluding_buffers), 10, 100);
        ImGui::InputInt("Total node count", &(settings.total_node_count), 10, 100);
        ImGui::InputInt("Subdomain height", &(settings.subdomain_height), 10, 100);
        ImGui::InputInt("Subdomain count", &(settings.subdomain_count), 10, 100);
        ImGui::InputInt("Buffer count", &(settings.buffer_count), 10, 100);  
    }

    /**
     * @brief Creates two input fields for two values related to the shift algorithm.
     * 
     * @param settings a reference to the settings object responsible for setting up all necessary simulation properties
     */
    inline void properties_shift_algorithm(Settings &settings)
    {
        ImGui::SeparatorText("Shift algorithm");

        ImGui::InputInt("Shift offset", &(settings.shift_offset), 10, 100);
        ImGui::InputInt("Shift distribution value count", &(settings.shift_distribution_value_count), 10, 100);   
    }

    /**
     * @brief Creates input fields for various values related to general fluid properties.
     * 
     * @param settings a reference to the settings object responsible for setting up all necessary simulation properties 
     */
    inline void properties_fluid(Settings &settings)
    {
        ImGui::SeparatorText("Fluid properties");

        ImGui::InputDouble("Inlet density", &(settings.inlet_density), 0.01f, 0.1f);
        ImGui::InputDouble("Outlet density", &(settings.outlet_density), 0.01f, 0.1f);
        ImGui::InputDouble("Inlet velocity x", &(settings.inlet_velocity[0]), 0.01f, 0.1f);
        ImGui::InputDouble("Inlet velocity y", &(settings.inlet_velocity[1]), 0.01f, 0.1f);
        ImGui::InputDouble("Outlet velocity x", &(settings.outlet_velocity[0]), 0.01f, 0.1f);
        ImGui::InputDouble("Outlet velocity y", &(settings.outlet_velocity[1]), 0.01f, 0.1f);
        ImGui::InputDouble("Relaxation time", &(settings.relaxation_time), 0.01, 0.1);
    }

    /**
     * @brief Provides information on the simulation status and displays the current progress using a progress bar.
     * 
     * @param gui_simulation_control a reference to an object responsible for tracking control data for the simulation  
     * @param gui_progress reference to an object containing all data on the current progress of a simulation
     */
    inline void show_simulation_status
    (
        GuiSimulationControl &gui_simulation_control,
        GuiProgress &gui_progress
    )
    {
        if(!gui_simulation_control.is_simulation_active)
        {
            ImGui::Text("Ready to start simulation.");
            ImGui::ProgressBar(0.0, ImVec2(0.9 * ImGui::GetWindowWidth(), 0.0f));
        }
        else if(gui_simulation_control.is_paused)
        {
            ImGui::Text("Simulation is paused.");
            ImGui::ProgressBar(gui_progress.progress, ImVec2(0.9 * ImGui::GetWindowWidth(), 0.0f));
        }
        else
        {
            ImGui::Text("Simulation is running.");
            ImGui::ProgressBar(gui_progress.progress, ImVec2(0.9 * ImGui::GetWindowWidth(), 0.0f));
        }
    }

    /**
     * @brief Creates an ImGui Combo for the selection of a colormap for the plot with the specified title.
     * 
     * @param colormap a reference to the currently selected colormap
     * @param plot_title the title of the plot for which the colormap is chosen
     */
    inline void colormap_picker
    (
        ImPlotColormap &colormap,
        const char *plot_title
    )
    {
        ImPlot::ColormapIcon(colormap);
        ImGui::SameLine();

        if (ImGui::BeginCombo("Colormap", ImPlot::GetColormapName(colormap)))
        {
            for (int n = 0; n < IM_ARRAYSIZE(COLOR_MAPS); n++)
            {
                bool is_selected = (colormap == COLOR_MAPS[n]); 
                ImPlot::ColormapIcon(COLOR_MAPS[n]);
                ImGui::SameLine();
                if (ImGui::Selectable(ImPlot::GetColormapName(COLOR_MAPS[n]), is_selected))
                {
                    colormap = COLOR_MAPS[n];
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
}

/**
 * @brief This namespace contains helper methods for interactions with contexts.
 * 
 */
namespace gui_contexts
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
}

/**
 * @brief This namespace contains the specification of all windows that can be shown.
 * 
 */
namespace gui_windows
{
    /**
     * @brief Creates a window showing status information and means to control simulations.
     *        The window contains the following buttons from the gui_buttons namespace: run, pause, and abort.
     *        Furthermore, it informs the user about whether a simulation is currently active and what its status is.
     *        It also shows the framerate at which new density and velocity plots are created. 
     *        The window can intentionally not be relocated.
     * 
     * @param settings a constant reference to the settings object responsible for setting up all necessary simulation properties
     * @param gui_monitor a constant reference to an object storing data related to the primary monitor and the viewport
     * @param gui_windows a reference to an object storing data on which windows are shown
     * @param gui_simulation_control a reference to an object responsible for tracking control data for the simulation  
     * @param gui_algorithmic a reference to an object containing all data specifying the algorithm used for the simulation
     * @param gui_simulation_data a reference to an object containing all data on which the simulation operates
     * @param gui_progress reference to an object containing all data on the current progress of a simulation
     * @param executor a reference to the executor used to execute the simulation
     */
    template <typename ContainerType, typename ResultsType>
    inline void simulation_status_window
    (
        const Settings &settings,
        const GuiMonitor &gui_monitor,
        GuiWindows &gui_windows, 
        GuiSimulationControl &gui_simulation_control,
        GuiAlgorithmic &gui_algorithmic,
        ContainerType &container_data,
        GuiProgress &gui_progess,
        Executor<ContainerType, ResultsType> &executor
    )
    {
        if (gui_windows.show_simulation_status)
        {
            ImGui::SetNextWindowSize
            (
                { 
                    1 * gui_monitor.viewport_work_size.x / 4, 
                    gui_monitor.viewport_work_size.y / 5
                }
            );
            ImGui::SetNextWindowPos({0,gui_windows.menu_bar_size});

            if(ImGui::Begin("Simulation", &gui_windows.show_simulation_status, ImGuiWindowFlags_NoResize))
            {
                ImGui::SeparatorText("Control");

                gui_items::gui_buttons::run_button(settings, gui_simulation_control, container_data, executor);
                ImGui::SameLine();
                gui_items::gui_buttons::pause_button(gui_simulation_control);
                ImGui::SameLine();
                gui_items::gui_buttons::abort_button(gui_simulation_control, gui_progess);

                ImGui::SeparatorText("Status");
                gui_items::show_simulation_status(gui_simulation_control, gui_progess);

                ImGui::SeparatorText("Framerate");
                ImGui::Text("%.1f FPS, %.3f ms/frame ", gui_progess.framerate, gui_progess.frametime);
            }                        
            ImGui::End();
        }
    }

    /**
     * @brief Creates a window allowing to set various properties of the algorithm and the domain used in a simulation.
     *        The window can intentionally not be relocated.
     * 
     * @param gui_monitor a constant reference to an object storing data related to the primary monitor and the viewport
     * @param settings a reference to the settings object responsible for setting up all necessary simulation properties
     * @param gui_windows a reference to an object storing data on which windows are shown
     * @param gui_simulation_control a reference to an object responsible for tracking control data for the simulation  
     * @param gui_algorithmic a reference to an object containing all data specifying the algorithm used for the simulation
     * @param gui_simulation_data a reference to an object containing all data on which the simulation operates
     */
    void properties_window
    (
        const GuiMonitor &gui_monitor,
        Settings &settings,
        GuiWindows &gui_windows,
        GuiSimulationControl &gui_simulation_control,
        GuiAlgorithmic &gui_algorithmic,
        GuiVisualizationData &gui_visualization_data
    );

    /**
     * @brief Work in progress: Creates the window showing control options for simulation data read from a csv file.
     * 
     * @param window_settings 
     */
    void read_from_file_window(const GuiMonitor &gui_monitor, GuiWindows &window_settings);

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
     * @param density_data an array of doubles containing the density values of all nodes
     *                     (can be retrieved from a std::vector by calling .data() on it)
     * @param settings a constant reference to the settings object responsible for setting up all necessary simulation properties
     * @param gui_monitor a constant reference to an object storing data related to the primary monitor and the viewport
     * @param gui_simulation_control a constant reference to an object responsible for tracking control data for the simulation  
     * @param gui_algorithmic a constant reference to an object containing all data specifying the algorithm used for the simulation
     * @param gui_simulation_data a constant reference to an object containing all data on which the simulation operates
     * @param gui_progress a constant reference to an object containing all data on the current progress of a simulation
     * @param gui_windows a reference to an object storing data on which windows are shown
     * @param gui_colormaps a reference to an object containing data on the colormaps used in plots
     */
    void density_window
    (
        const GuiMonitor &gui_monitor,
        const GuiSimulationControl &gui_simulation_control,
        const GuiAlgorithmic &gui_algorithmic,
        const GuiVisualizationData &gui_visualization_data,
        const GuiProgress &gui_progress,
        GuiWindows &gui_windows, 
        GuiColormaps &gui_colormaps
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
     * @param settings a constant reference to the settings object responsible for setting up all necessary simulation properties
     * @param gui_monitor a constant reference to an object storing data related to the primary monitor and the viewport
     * @param gui_simulation_control a constant reference to an object responsible for tracking control data for the simulation  
     * @param gui_algorithmic a constant reference to an object containing all data specifying the algorithm used for the simulation
     * @param gui_simulation_data a constant reference to an object containing all data on which the simulation operates
     * @param gui_progress a constant reference to an object containing all data on the current progress of a simulation
     * @param gui_windows a reference to an object storing data on which windows are shown
     * @param gui_velocity_quiver_data a reference to an object containing all data for the quiver plot
     * @param gui_colormaps a reference to an object containing data on the colormaps used in plots
     */
    void velocity_window
    (
        const GuiMonitor &gui_monitor,
        const GuiSimulationControl &gui_simulation_control,
        //const GuiSimulationData &gui_simulation_data,
        const GuiVisualizationData &gui_visualization_data,
        const GuiProgress &gui_progress,
        GuiWindows &gui_windows, 
        GuiVelocityQuiverData &gui_velocity_quiver_data,
        GuiColormaps &gui_colormaps
    );
}

/**
 * @brief This namespace contains helper methods related to OpenGL and GLFW.
 * 
 */
namespace gui_rendering
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
     * @param gui_monitor the primary monitor on which the window is shown
     */
    inline void render(GLFWwindow *window, GuiMonitor &gui_monitor)
    {
        ImGui::Render();
        glViewport(0, 0, gui_monitor.display_width, gui_monitor.display_height);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
}

/**
 * @brief Utility namespace of helper functions that were found to be unsuited elsewhere
 * 
 */
namespace gui_misc
{
    /**
     * @brief Prints debug information on important simulation data to the console.
     * 
     * @param distribution_values a constant reference to a vector containing all distribution values
     * @param nodes a constant reference to a vector containing the indices of all nodes
     * @param fluid_nodes a constant reference to a vector containing the indices of all fluid nodes
     * @param phase_information a constant reference to a vector containing the phase information of all nodes
     */
    inline void debug_prints
    (
        const std::vector<double> &distribution_values,
        const std::vector<unsigned int> &nodes,
        const std::vector<unsigned int> &fluid_nodes,
        const std::vector<bool> &phase_information
    )
    {
        /* Illustration of the phase information */
        std::cout << "Illustration of lattice: " << std::endl;
        to_console::print_phase_vector(phase_information);
        std::cout << std::endl;
        /* Overview */
        std::cout << "Enumeration of all nodes within the lattice: " << std::endl;
        to_console::print_vector(nodes);
        std::cout << std::endl;

        std::cout << "Enumeration of all fluid nodes within the simulation domain: " << std::endl;
        to_console::print_vector(fluid_nodes, HORIZONTAL_NODES - 2);
        std::cout << std::endl;

        std::cout << "Initial distributions:" << std::endl;
        to_console::print_distribution_values(distribution_values, ACCESS_FUNCTION);
        std::cout << std::endl;
    }

    /**
     * @brief Adds the colormap used to distinguish solid and fluid nodes.
     *        It is meant to be used in a heatmap on top of another displaying the densities and velocities.
     *        Solid nodes are black regardless of the chosen color theme.
     */
    inline void add_solid_colormap()
    {
        ImVector<ImVec4> custom;
        custom.push_back(ImVec4(0,0,0,1));
        custom.push_back(ImVec4(1,1,1,0));
        ImPlot::AddColormap("SOLID_MASK",custom.Data,custom.Size,true);
    }
}

/**
 * @brief 
 * 
 * @tparam Ex 
 * @return int 
 */
template <class Ex>
int run(const std::string &window_title)
{
    glfwSetErrorCallback(gui_rendering::glfw_error_callback);
    if (!glfwInit())
        return 1;
    #if defined(IMGUI_IMPL_OPENGL_ES2)
        // GL ES 2.0 + GLSL 100
        const char* glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    #elif defined(__APPLE__)
        // GL 3.2 + GLSL 150
        const char* glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
    #else
        // GL 3.0 + GLSL 130
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
    #endif
    
    GuiWindows gui_windows_config;
    GuiSimulationData gui_simulation_data;
    GuiVisualizationData gui_visualization_data;
    GuiSimulationControl gui_simulation_control;
    GuiProgress gui_progress;
    GuiMonitor gui_monitor;
    GuiAlgorithmic gui_algorithmic;
    GuiColormaps gui_colormaps;
    GuiVelocityQuiverData gui_velocity_quiver_data;

    Ex executor(gui_simulation_data);

    gui_monitor.primary_monitor = glfwGetPrimaryMonitor();
    gui_monitor.video_mode = glfwGetVideoMode(gui_monitor.primary_monitor);

    gui_monitor.video_mode_width = gui_monitor.video_mode->width;
    gui_monitor.video_mode_height = gui_monitor.video_mode->height;

    std::cout << "Structs \n";

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(gui_monitor.video_mode_width, gui_monitor.video_mode_height, window_title.c_str(), nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    //glfwSwapInterval(1); // Enable vsync

    gui_contexts::create_contexts();
    gui_styles::set_light_style();
    glfwGetMonitorContentScale(gui_monitor.primary_monitor, &gui_monitor.monitor_x_scale, &gui_monitor.monitor_x_scale);

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.Fonts->AddFontFromFileTTF("../fonts/DroidSans.ttf", 2 * sqrt(gui_monitor.monitor_x_scale * gui_monitor.monitor_x_scale) * 9.0f);

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    glfwGetFramebufferSize(window, &gui_monitor.display_width, &gui_monitor.display_height);

    // Set defaults
    static Settings settings = get_default_settings();
    setup_global_variables(settings);

    // Timer initializations
    SimpleTimer timer;
    SimpleTimer timer_framerate;

    // Simulation variable initializations
    gui_simulation_data.distribution_values_0 = std::vector<double>(0, settings.total_node_count * DIRECTION_COUNT);
    gui_simulation_data.nodes = std::vector<unsigned int>(0, settings.total_node_count);
    gui_simulation_data.fluid_nodes = std::vector<unsigned int>(0, settings.total_node_count);
    gui_simulation_data.phase_information = std::vector<bool>(false, settings.total_node_count);
    gui_simulation_data.current_result = std::make_tuple(std::vector<velocity>(settings.total_node_count, {0,0}), std::vector<double>(settings.total_node_count,0));
    gui_simulation_data.absolute_velocities->assign(settings.total_node_count, 0);
    gui_simulation_data.x_velocities->assign(settings.total_node_count, 0);
    gui_simulation_data.y_velocities->assign(settings.total_node_count, 0);
    gui_velocity_quiver_data.x_values->assign(2 * settings.total_node_count, 0);
    gui_velocity_quiver_data.y_values->assign(2 * settings.total_node_count, 0);

    gui_visualization_data.horizontal_nodes = settings.horizontal_nodes;
    gui_visualization_data.vertical_nodes = settings.vertical_nodes;
    gui_visualization_data.absolute_velocity_values = gui_simulation_data.absolute_velocities;
    gui_visualization_data.x_velocity_values = gui_simulation_data.x_velocities;
    gui_visualization_data.y_velocity_values = gui_simulation_data.y_velocities;
    gui_visualization_data.density_values->assign(settings.total_node_count, 0);

    if (settings.algorithm != "sequential_shift" && settings.algorithm != "parallel_shift")
    {
        if (settings.access_pattern == "collision")
        {
            gui_simulation_data.access_function = lbm_access::collision;
        }
        else if (settings.access_pattern == "stream")
        {
            gui_simulation_data.access_function = lbm_access::stream;
        }
        else
        {
            gui_simulation_data.access_function = lbm_access::bundle;
        }
    }

    gui_misc::add_solid_colormap();

    ImPlotStyle& implot_style = ImPlot::GetStyle();
    implot_style.PlotPadding = ImVec2(gui_monitor.monitor_x_scale * 20, gui_monitor.monitor_y_scale * 20);

    gui_colormaps.density_colormap_lower_scale = settings.inlet_density;
    gui_colormaps.density_colormap_upper_scale = 1.25 * settings.inlet_density;
    gui_colormaps.velocity_colormap_upper_scale = sqrt(pow(settings.inlet_velocity[0], 2) + pow(settings.inlet_velocity[1], 2));  

    std::cout << "Maybe entering loop \n"; 

    // Eternal loop of imaging magic
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        gui_monitor.viewport = *ImGui::GetMainViewport();
        gui_monitor.viewport_work_size = gui_monitor.viewport.WorkSize;

        gui_items::menu_bar(gui_windows_config);

        gui_windows::read_from_file_window(gui_monitor, gui_windows_config);

        gui_windows::properties_window
        (
            gui_monitor,
            settings,
            gui_windows_config,
            gui_simulation_control,
            gui_algorithmic,
            gui_visualization_data
        );

        gui_windows::simulation_status_window
        (
            settings,
            gui_monitor,
            gui_windows_config,
            gui_simulation_control,
            gui_algorithmic,
            gui_simulation_data,
            gui_progress,
            executor
        );

        if(gui_simulation_control.is_simulation_active && !gui_simulation_control.is_paused)
        {
            if(executor.is_ready())
            {
                if(timer_framerate.elapsed() > 0.25)
                {
                    gui_progress.framerate = 1 / timer.elapsed();
                    gui_progress.frametime = 1000.0f / gui_progress.framerate;
                    timer_framerate.restart();
                }
                
                timer.restart();
                gui_simulation_data.current_result = *executor.get();

                executor.execute(gui_simulation_data);
                gui_progress.current_iter++;
                gui_progress.progress = (double)gui_progress.current_iter / settings.time_steps;

                if(gui_simulation_control.results_to_csv)
                {
                    gui_simulation_data.all_results.push_back(gui_simulation_data.current_result);
                }

                for(int i = 0; i < std::get<1>(gui_simulation_data.current_result).size(); ++i)
                {
                    (*(gui_visualization_data.x_velocity_values))[i] = (std::get<0>(gui_simulation_data.current_result))[i][0];
                    (*(gui_visualization_data.y_velocity_values))[i] = std::get<0>(gui_simulation_data.current_result)[i][1];
                    (*(gui_visualization_data.absolute_velocity_values))[i] = sqrt(pow((*(gui_simulation_data.x_velocities))[i], 2) + pow((*(gui_simulation_data.y_velocities))[i], 2));
                    (*(gui_visualization_data.density_values))[i] = std::get<1>(gui_simulation_data.current_result)[i];
                }

                if(gui_windows_config.enable_velocity_quiver)
                {
                    gui_velocity_quiver_data.x_values->assign(2 * std::get<1>(gui_simulation_data.current_result).size(), 0);
                    gui_velocity_quiver_data.y_values->assign(2 * std::get<1>(gui_simulation_data.current_result).size(), 0);

                    for(int y = 0; y < settings.vertical_nodes; ++y)
                    {
                        for(int x = 0; x < settings.horizontal_nodes; ++x)
                        {
                            unsigned int node_index = lbm_access::get_node_index(x,y);
                            if(gui_simulation_data.absolute_velocities->at(node_index) > 1e-15)
                            {
                                double base_x = 0.5 + x;
                                double base_y = 0.5 + y;

                                (*gui_velocity_quiver_data.x_values)[2 * node_index] = base_x;
                                (*gui_velocity_quiver_data.y_values)[2 * node_index] = base_y;

                                double offset_x = base_x + 0.5 * (1.0 / gui_simulation_data.absolute_velocities->at(node_index)) * gui_simulation_data.x_velocities->at(node_index);
                                double offset_y = base_y + 0.5 * (1.0 / gui_simulation_data.absolute_velocities->at(node_index)) * gui_simulation_data.y_velocities->at(node_index);

                                (*gui_velocity_quiver_data.x_values)[2 * node_index + 1] = offset_x;
                                (*gui_velocity_quiver_data.y_values)[2 * node_index + 1] = offset_y;
                            }
                        }  
                    }
                }

                // Check if simulation is finished
                if(gui_progress.current_iter == settings.time_steps)
                {
                    gui_progress.current_iter = 0;
                    gui_progress.progress = 0;
                    gui_simulation_control.is_simulation_active = false;

                    if(gui_simulation_control.results_to_csv)
                    {
                        std::string filename{gui_simulation_control.result_file_name};
                        if(filename.size() < 4)
                        {
                            filename.append(".csv");
                        }
                        std::string end{filename.end() - 4, filename.end()};
                        if(end != ".csv")
                        {
                            filename.append(".csv");
                        }
                        sim_data_to_csv(gui_simulation_data.all_results, "../results/" + filename);
                    }
                }
            }
        }

        gui_windows::density_window
        (
            gui_monitor, 
            gui_simulation_control, 
            gui_algorithmic, 
            gui_visualization_data, 
            gui_progress, 
            gui_windows_config, 
            gui_colormaps
        );

        gui_windows::velocity_window
        (
            gui_monitor, 
            gui_simulation_control, 
            gui_visualization_data, 
            gui_progress, 
            gui_windows_config, 
            gui_velocity_quiver_data,
            gui_colormaps
        );

        gui_rendering::render(window, gui_monitor);
    }

    gui_contexts::destroy(window);

    return 0;
}

#endif
