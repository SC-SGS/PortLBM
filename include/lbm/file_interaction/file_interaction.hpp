/**
 * @file        file_interaction.hpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This header file contains the declarations of two functions for retrieving the properties of a simulation
 *              from a JSON file and for storing properties to a JSON file.
 * 
 * @version     1.1
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 */

#ifndef FILE_INTERACTION_HPP
#define FILE_INTERACTION_HPP

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
         * @brief Outputs a JSON file to the specified path based on the specified `Properties` object.
         * 
         * @param[in] properties the properties are read from this object    
         * @param[in] path       the path where the JSON file is stored
         */
        void properties_to_json
        (
            const core::Properties &properties,
            const std::string &path = "../settings/domain.json"
        );

        /**
         * @brief Creates a `Properties` object based on an input JSON file.
         * 
         * @param[in] path the path to the JSON file
         * 
         * @return a unique pointer to a `Properties` object
         */
        std::unique_ptr<core::Properties> json_to_properties(const std::string &path = "../settings/domain.json");

    } // !namespace file_interaction

} // !namespace lbm

#endif