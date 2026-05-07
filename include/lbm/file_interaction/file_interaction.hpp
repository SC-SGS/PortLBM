/**
 * @brief       This header file contains the declarations of two functions for retrieving the properties of a
 *              simulation from a JSON file and for storing properties to a JSON file.
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 *              Copyright (c) 2026 Alexander Strack
 */

#ifndef LBM_FILE_INTERACTION_HPP
#define LBM_FILE_INTERACTION_HPP

// Dependencies on LBM core features
#include "../core/properties.hpp"

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
 * @brief   Writes a `Properties` object back to the JSON file it was loaded from.
 *          The destination path is taken from `properties.settings_path`, which is
 *          set automatically by `json_to_properties()`.
 *
 * @param[in]   properties  source of the values written to the file
 */
void properties_to_json(const core::Properties &properties);

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

}  // namespace file_interaction

}  // namespace lbm

#endif  // ! LBM_FILE_INTERACTION_HPP
