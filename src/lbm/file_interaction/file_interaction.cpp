/**
 * @file        file_interaction.cpp
 * 
 * @author      Marcel Graf
 * 
 * @brief       This source file contains the definitions of two functions for retrieving the properties of a simulation
 *              from a JSON file and for storing properties to a JSON file.
 * 
 * @version     1.1
 * 
 * @date        November 2024
 * 
 * @copyright   Copyright (c) 2024
 */


#include "../../../include/lbm/file_interaction/file_interaction.hpp"

void lbm::file_interaction::properties_to_json
(
    const lbm::core::Properties &properties,
    const std::string &path
)
{
    nlohmann::json file_data;

    file_data["algorithmic"]["debugMode"] = properties.debug_mode;
    file_data["algorithmic"]["resultsToCsv"] = properties.results_to_csv;
    file_data["algorithmic"]["relaxationTime"] = properties.relaxation_time;
    file_data["algorithmic"]["timeSteps"] = properties.time_steps;
    file_data["algorithmic"]["dataLayout"] = properties.data_layout;
    file_data["algorithmic"]["algorithm"] = properties.algorithm;

    file_data["domain"]["verticalNodes"] = properties.vertical_nodes - 2;
    file_data["domain"]["horizontalNodes"]= properties.horizontal_nodes - 2;

    file_data["inlets"]["velocity"]["x"] = properties.inlet_velocity_x;
    file_data["inlets"]["velocity"]["y"] = properties.inlet_velocity_y;
    file_data["inlets"]["density"] = properties.inlet_density;

    file_data["outlets"]["velocity"]["x"] = properties.outlet_velocity_x;
    file_data["outlets"]["velocity"]["y"] = properties.outlet_velocity_y;
    file_data["outlets"]["density"] = properties.outlet_density;

    std::ofstream file(path);
    file << std::setw(4) << file_data;
    file.close();
}

std::shared_ptr<lbm::core::Properties> lbm::file_interaction::json_to_properties(const std::string &path)
{
    std::ifstream file(path);
    nlohmann::json data = nlohmann::json::parse(file);
    file.close();

    return std::make_shared<lbm::core::Properties>
    (
        lbm::core::Properties
        (
            // Algorithmic properties
            data.at("algorithmic").at("algorithm").get<std::string>(),
            data.at("algorithmic").at("dataLayout").get<std::string>(),
            data.at("algorithmic").at("debugMode").get<bool>(),
            data.at("algorithmic").at("resultsToCsv").get<bool>(),
            data.at("algorithmic").at("relaxationTime").get<double>(),
            data.at("algorithmic").at("timeSteps").get<unsigned int>(),
            // Domain properties
            data.at("domain").at("verticalNodes").get<unsigned int>(),
            data.at("domain").at("horizontalNodes").get<unsigned int>(),
            // Inlets
            data.at("inlets").at("velocity").at("x").get<double>(),
            data.at("inlets").at("velocity").at("y").get<double>(),
            data.at("inlets").at("density").get<double>(),
            // Outlets
            data.at("outlets").at("velocity").at("x").get<double>(),
            data.at("outlets").at("velocity").at("y").get<double>(),
            data.at("outlets").at("density").get<double>()
        )
    );
}

