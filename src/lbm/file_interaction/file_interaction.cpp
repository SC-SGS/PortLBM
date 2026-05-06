/**
 * @file        file_interaction.cpp
 *
 * @author      Marcel Graf
 *
 * @brief       This source file contains the definitions of two functions for retrieving the properties of a
 *              simulation from a JSON file and for storing properties to a JSON file.
 *
 * @version     1.5
 *
 * @date        March 2025
 *
 * @copyright   Copyright (c) Marcel Graf
 */

#include "../../../include/lbm/file_interaction/file_interaction.hpp"

void lbm::file_interaction::properties_to_json(const lbm::core::Properties &properties, const std::string &path)
{
    nlohmann::json file_data;

    file_data["algorithmic"]["workGroupSize"] = properties.work_group_size;
    file_data["algorithmic"]["debugMode"] = properties.debug_mode;
    file_data["algorithmic"]["timeSteps"] = properties.time_steps;
    file_data["algorithmic"]["dataLayout"] = properties.data_layout;
    file_data["algorithmic"]["algorithm"] = properties.algorithm;
    file_data["algorithmic"]["frameUpdateInterval"] = properties.frame_update_interval;

    file_data["domain"]["scenario"] = properties.scenario;
    file_data["domain"]["verticalNodes"] = properties.vertical_nodes;
    file_data["domain"]["horizontalNodes"] = properties.horizontal_nodes;

    file_data["physical"]["inletVelocity"]["x"] = properties.inlet_velocity_x;
    file_data["physical"]["inletVelocity"]["y"] = properties.inlet_velocity_y;
    file_data["physical"]["inletDensity"] = properties.inlet_density;
    file_data["physical"]["outletVelocity"]["x"] = properties.outlet_velocity_x;
    file_data["physical"]["outletVelocity"]["y"] = properties.outlet_velocity_y;
    file_data["physical"]["outletDensity"] = properties.outlet_density;
    file_data["physical"]["relaxationTime"] = properties.relaxation_time;

    std::ofstream file(path);
    file << std::setw(4) << file_data;
    file.close();
}

lbm::core::Properties lbm::file_interaction::json_to_properties(const std::string &path, const int offset)
{
    std::ifstream file(path);
    nlohmann::json data = nlohmann::json::parse(file);
    file.close();

    lbm::core::Properties result(
        // Algorithmic properties
        data.at("algorithmic").at("algorithm").get<std::string>(),
        data.at("algorithmic").at("dataLayout").get<std::string>(),
        data.at("algorithmic").at("debugMode").get<bool>(),
        data.at("algorithmic").at("workGroupSize").get<unsigned int>(),
        data.at("algorithmic").at("timeSteps").get<unsigned int>(),
        data.at("algorithmic").at("frameUpdateInterval").get<unsigned int>(),
        // Domain properties
        data.at("domain").at("scenario").get<std::string>(),
        data.at("domain").at("verticalNodes").get<unsigned int>() + offset,
        data.at("domain").at("horizontalNodes").get<unsigned int>() + offset,
        // Physical
        data.at("physical").at("inletVelocity").at("x").get<real_type>(),
        data.at("physical").at("inletVelocity").at("y").get<real_type>(),
        data.at("physical").at("inletDensity").get<real_type>(),
        data.at("physical").at("outletVelocity").at("x").get<real_type>(),
        data.at("physical").at("outletVelocity").at("y").get<real_type>(),
        data.at("physical").at("outletDensity").get<real_type>(),
        data.at("physical").at("relaxationTime").get<real_type>());

    // Record the source path so callers can round-trip without tracking it separately.
    result.settings_path = path;
    return result;
}
