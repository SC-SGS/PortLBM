/**
 * @file        file_interaction.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations of two functions for retrieving the properties of a 
 *              simulation from a JSON file and for storing properties to a JSON file.
 * 
 * @version     1.3
 * 
 * @date        March 2025
 * 
 * @copyright   Copyright (c) Marcel Graf
 */

#ifndef LBM_FILE_INTERACTION_HPP
#define LBM_FILE_INTERACTION_HPP

// Dependencies on LBM core features
#include "../core/simulation.hpp"

// C++ JSON integration by N. Lohmann
#include <nlohmann/json.hpp>

// Standard library
#include <fstream>
#include <iostream>

namespace lbm
{
    /**
     * @brief This namespace contains function for exchanging information between `Properties` objects and JSON files.
     */
    namespace file_interaction
    {
        /**
         * @brief   Outputs a JSON file to the specified path based on the specified `Properties` object.
         *          The path must be supplied explicitly. Use `properties.settings_path` when the
         *          object was originally loaded via `json_to_properties`.
         *
         * @param[in]   properties  source of the values written to the file
         * @param[in]   path        destination path for the JSON file (required)
         */
        void properties_to_json
        (
            const core::Properties &properties,
            const std::string &path
        );

        /**
         * @brief   Creates a `core::Properties` object from an input JSON file.
         *          The returned object's `settings_path` field is set to `path` so that
         *          callers can later invoke `properties_to_json` without tracking the path
         *          separately.
         *
         * @param[in]   path    path to the JSON settings file (required — no default)
         * @param[in]   offset  value added to horizontal/vertical node counts (e.g. to strip ghost nodes)
         *
         * @return  a `Properties` object with `settings_path` populated
         */
        core::Properties json_to_properties(const std::string &path, const int offset = 0);

    } // ! namespace file_interaction

} // ! namespace lbm

#endif // ! LBM_FILE_INTERACTION_HPP
