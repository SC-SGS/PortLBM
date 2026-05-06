/**
 * @file        properties.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       Simulation parameters. This header has no SYCL dependency and
 *              is safe to include in any context that does not require a device.
 *
 * @version     1.0
 *
 * @date        May 2025
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 */

#ifndef LBM_PROPERTIES_HPP
#define LBM_PROPERTIES_HPP

#include "constants.hpp"
#include <string>

namespace lbm
{

namespace core
{

/**
 * @brief   All configurable parameters of a simulation run.
 *
 * Instances can be created entirely in C++ without a JSON file by calling the
 * explicit constructor directly.  Use `file_interaction::json_to_properties()`
 * when settings come from disk.
 *
 * Call `validate()` after construction to confirm that every field holds a
 * physically and algorithmically legal value.
 */
struct Properties
{
    /* ---- Algorithmic options ---- */

    /**
     * @brief   LBM kernel variant.
     *
     * Valid values: `"gpu-two-lattice"`, `"gpu-two-lattice-linear"`,
     * `"gpu-two-lattice-buffered"`, `"gpu-swap"`.
     */
    std::string algorithm;

    /**
     * @brief   Memory layout for the distribution-function arrays.
     *
     * Valid values: `"stream"`, `"collision"`, `"bundle"`.
     */
    std::string data_layout;

    /// When `true`, verbose per-iteration console output is produced.
    bool debug_mode;

    /// Number of SYCL work-items per work-group.
    unsigned int work_group_size;

    /// Total number of LBM time steps to execute.
    unsigned int time_steps;

    /// Macroscopic observables are copied to the host every this many steps.
    unsigned int frame_update_interval;

    /* ---- Domain options ---- */

    /**
     * @brief   Geometry scenario used to initialise solid nodes.
     *
     * Valid values: `"Hagen-Poiseuille"`, `"walls"`, `"circle"`, `"square"`,
     * `"plate"`, `"skyscraper"`, `"wing"`, `"porous"`.
     */
    std::string scenario;

    /// Vertical node count including the two ghost-node rows.
    unsigned int vertical_nodes;

    /// Horizontal node count including the two ghost-node columns.
    unsigned int horizontal_nodes;

    /// Total node count (ghost nodes included, no buffer nodes).
    unsigned int total_unexpanded_node_count;

    /// Node count of the interior domain (ghost nodes excluded).
    unsigned int domain_node_count;

    /* ---- Physical parameters ---- */

    real_type inlet_velocity_x;
    real_type inlet_velocity_y;
    real_type inlet_density;

    real_type outlet_velocity_x;
    real_type outlet_velocity_y;
    real_type outlet_density;

    real_type relaxation_time;

    /**
     * @brief   Path to the JSON file from which this object was loaded, or
     *          empty when constructed programmatically.
     *          Set automatically by `file_interaction::json_to_properties()`.
     */
    std::string settings_path;

    /**
     * @brief   Constructs a `Properties` object from individual parameters.
     *
     * The `vertical_nodes` and `horizontal_nodes` arguments are the *inner*
     * extents; two ghost-node layers are added automatically, so the stored
     * values are `inner + 2`.
     *
     * @param algorithm             LBM kernel variant (see field documentation)
     * @param data_layout           memory layout (see field documentation)
     * @param debug_mode            enable verbose console output
     * @param work_group_size       SYCL work-items per work-group
     * @param time_steps            total number of LBM time steps
     * @param frame_update_interval host-copy interval in steps
     * @param scenario              geometry scenario (see field documentation)
     * @param vertical_nodes        inner vertical node count (ghost nodes added by ctor)
     * @param horizontal_nodes      inner horizontal node count (ghost nodes added by ctor)
     * @param inlet_velocity_x      x-component of inlet velocity
     * @param inlet_velocity_y      y-component of inlet velocity
     * @param inlet_density         inlet density
     * @param outlet_velocity_x     x-component of outlet velocity
     * @param outlet_velocity_y     y-component of outlet velocity
     * @param outlet_density        outlet density
     * @param relaxation_time       BGK relaxation time τ
     */
    explicit Properties(
        std::string algorithm,
        std::string data_layout,
        bool debug_mode,
        unsigned int work_group_size,
        unsigned int time_steps,
        unsigned int frame_update_interval,
        std::string scenario,
        unsigned int vertical_nodes,
        unsigned int horizontal_nodes,
        real_type inlet_velocity_x,
        real_type inlet_velocity_y,
        real_type inlet_density,
        real_type outlet_velocity_x,
        real_type outlet_velocity_y,
        real_type outlet_density,
        real_type relaxation_time);

    /**
     * @brief   Validates all fields and throws on any constraint violation.
     *
     * Checked constraints:
     *  - `vertical_nodes >= 4` and `horizontal_nodes >= 4` (inner domain ≥ 2×2)
     *  - `time_steps >= 1`
     *  - `1 <= frame_update_interval <= time_steps`
     *  - `work_group_size >= 1`
     *  - `inlet_density > 0` and `outlet_density > 0`
     *  - `relaxation_time > 0`
     *  - `algorithm` is one of the four known values
     *  - `data_layout` is one of the three known values
     *  - `scenario` is one of the eight known values
     *
     * @throws  lbm::exceptions::json::PropertyArgumentException
     */
    void validate() const;

    /**
     * @brief   Returns a human-readable summary of all parameters.
     */
    std::string to_string() const;
};

}  // namespace core

}  // namespace lbm

#endif  // LBM_PROPERTIES_HPP
