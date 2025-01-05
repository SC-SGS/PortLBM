/**
 * @file        gui_constants.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains constants for the GUI.
 * 
 * @version     1.1
 * 
 * @date        January 2025
 * 
 * @copyright   Copyright (c) 2024
 */

#ifndef GUI_CONSTANTS_HPP
#define GUI_CONSTANTS_HPP

// ImPlot
#include "../../external/implot/implot.h"

// Standary library
#include <array>
#include <string_view>

namespace lbm
{

    namespace gui
    {

        /**
         * @brief This namespace contains various `constexpr` used throughout the project.
         */
        namespace constants
        {

            constexpr std::array<std::string_view, 3> algorithms = {"gpu-two-lattice", "gpu-two-lattice-linear", "gpu-swap"};
            constexpr std::array<std::string_view, 3> data_layouts = {"collision", "stream", "bundle"};
            constexpr std::array<std::string_view, 8> scenarios = 
                {"Hagen-Poiseuille", "walls", "circle", "square", "wing", "skyscraper", "porous", "plate"};
            
            /**
             * @brief An array containing all default colormaps offered by ImPlot.
             */
            constexpr std::array<ImPlotColormap_, 16> color_maps = 
            {
                ImPlotColormap_Deep, 
                ImPlotColormap_Dark, 
                ImPlotColormap_Pastel, 
                ImPlotColormap_Paired, 
                ImPlotColormap_Viridis, 
                ImPlotColormap_Plasma,
                ImPlotColormap_Hot, 
                ImPlotColormap_Cool, 
                ImPlotColormap_Pink, 
                ImPlotColormap_Jet, 
                ImPlotColormap_Twilight, 
                ImPlotColormap_RdBu, 
                ImPlotColormap_BrBG,
                ImPlotColormap_PiYG, 
                ImPlotColormap_Spectral, 
                ImPlotColormap_Greys
            };

        } // ! namespace constants

    } // ! namespace gui

} // ! namespace lbm

#endif // ! GUI_CONSTANTS_HPP
