/**
 * @file        domain.hpp
 *
 * @author      Marcel Graf
 *
 * @brief       Simulation domain dimensions, derived from Properties. No SYCL
 *              dependency — safe to include anywhere Properties is available.
 *
 * @version     1.0
 *
 * @date        May 2025
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 */

#ifndef LBM_DOMAIN_HPP
#define LBM_DOMAIN_HPP

#include "properties.hpp"
#include <string>

namespace lbm
{

namespace core
{

/**
 * @brief   Simulation domain dimensions derived from `Properties` for a
 *          particular algorithm.
 *
 * Depending on the chosen algorithm, the domain may be padded to an even
 * multiple of the work-group size and/or decomposed into sub-domains.
 * Construct via `Domain(properties)` — it chooses the right layout
 * automatically based on `properties.algorithm`.
 */
class Domain
{
  public:
    /// Total node count including buffer nodes (algorithm-dependent).
    unsigned int total_node_count;

    /// Total horizontal node count including buffers.
    unsigned int horizontal_nodes;

    /// Total vertical node count including buffers.
    unsigned int vertical_nodes;

    /// Vertical nodes per sub-domain.
    unsigned int subdomain_vertical_nodes;

    /// Horizontal nodes per sub-domain.
    unsigned int subdomain_horizontal_nodes;

    /// Number of sub-domains in the vertical direction.
    unsigned int subdomain_count_vertical;

    /// Number of sub-domains in the horizontal direction.
    unsigned int subdomain_count_horizontal;

    /**
     * @brief   Initialises the domain according to `properties`.
     *
     * The sub-domain layout is chosen based on `properties.algorithm` and
     * `properties.work_group_size`.  For the swap algorithm, `work_group_size`
     * may be corrected if it falls below the minimum of 6.
     *
     * @param[in,out]   properties  source parameters; the swap variant may
     *                              update `work_group_size` and rewrite the
     *                              settings JSON if `settings_path` is set
     */
    explicit Domain(core::Properties &properties);

  private:
    void setup_for_decomposed_two_lattice(
        unsigned int unexpanded_horizontal_nodes, unsigned int unexpanded_vertical_nodes, size_t work_group_size);

    void setup_for_buffered_two_lattice(
        unsigned int unexpanded_horizontal_nodes, unsigned int unexpanded_vertical_nodes, size_t work_group_size);

    void setup_for_swap(unsigned int unexpanded_horizontal_nodes,
                        unsigned int unexpanded_vertical_nodes,
                        size_t work_group_size,
                        const std::string &settings_path);

    explicit Domain(unsigned int total_node_count,
                    unsigned int vertical_nodes,
                    unsigned int horizontal_nodes,
                    unsigned int subdomain_vertical_nodes,
                    unsigned int subdomain_horizontal_nodes,
                    unsigned int subdomain_count_vertical,
                    unsigned int subdomain_count_horizontal);
};

}  // namespace core

}  // namespace lbm

#endif  // LBM_DOMAIN_HPP
