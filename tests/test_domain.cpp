/**
 * @brief   Unit tests for lbm::core::Domain (plan item 4.3).
 *
 *          Covers the domain-layout math for each algorithm variant:
 *            - lptl (Linear Pull Two-Lattice)       — no decomposition; exact node counts
 *            - nptl (Non-linear Pull Two-Lattice)   — decomposed layout; structural invariants
 *            - npol (Non-linear Pull One-Lattice)   — buffered layout; structural invariants
 *            - nsol (Non-linear Swap One-Lattice)   — swap layout; structural invariants
 *
 *          For the decomposed/buffered/swap algorithms the exact expanded
 *          extents depend on subdomain arithmetic that is sensitive to grid
 *          size and work-group size, so we verify structural invariants
 *          (total == h * v, domain not smaller than requested) rather than
 *          hard-coded values.  For lptl (no expansion) we can assert exact values.
 * @copyright   Copyright (c) 2026 Alexander Strack
 */

#include <catch2/catch_test_macros.hpp>

#include "lbm/core/domain.hpp"
#include "lbm/core/properties.hpp"

// ---------------------------------------------------------------------------
// Helper: build a minimal valid Properties for a given algorithm
// ---------------------------------------------------------------------------

static lbm::core::Properties make_props(const std::string &algorithm,
                                        unsigned int inner_v,
                                        unsigned int inner_h,
                                        unsigned int wg)
{
    return lbm::core::Properties(
        algorithm, "stream", false, wg,
        /*time_steps=*/10, /*frame_interval=*/10,
        "Hagen-Poiseuille",
        inner_v, inner_h,
        0.0, 0.0, 1.2, 0.0, 0.0, 1.0, 0.6);
}

// ---------------------------------------------------------------------------
// lptl — Linear Pull Two-Lattice  (no subdomain decomposition)
// ---------------------------------------------------------------------------

TEST_CASE("Domain: lptl preserves ghost-extended extents",
          "[domain][linear]")
{
    // The linear algorithm does not call any setup_for_* method, so Domain
    // stores the ghost-extended extents from Properties directly.
    //
    // inner_v=10, inner_h=6  →  stored_v=12, stored_h=8
    auto props = make_props("lptl", 10, 6, 64);

    lbm::core::Domain dom(props);

    CHECK(dom.vertical_nodes   == 12u);
    CHECK(dom.horizontal_nodes == 8u);
    CHECK(dom.total_node_count == 12u * 8u);

    // No decomposition: single subdomain spanning the whole grid
    CHECK(dom.subdomain_count_vertical   == 1u);
    CHECK(dom.subdomain_count_horizontal == 1u);
    CHECK(dom.subdomain_vertical_nodes   == 12u);
    CHECK(dom.subdomain_horizontal_nodes == 8u);
}

// ---------------------------------------------------------------------------
// nptl — Non-linear Pull Two-Lattice  (decomposed, non-buffered)
// ---------------------------------------------------------------------------

TEST_CASE("Domain: nptl structural invariants", "[domain][decomposed]")
{
    SECTION("work-group size 4 (power of 4 → square 2×2 subdomains)")
    {
        // inner 10×10 → stored 12×12
        auto props = make_props("nptl", 10, 10, 4);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 12u);
        CHECK(dom.vertical_nodes   >= 12u);
        CHECK(dom.subdomain_horizontal_nodes == 2u);
        CHECK(dom.subdomain_vertical_nodes   == 2u);
    }

    SECTION("work-group size 8 (power of 2, not 4 → rectangular 2×4 subdomains)")
    {
        auto props = make_props("nptl", 10, 10, 8);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 12u);
        CHECK(dom.vertical_nodes   >= 12u);
        // subdomain_h * subdomain_v == work_group_size
        CHECK(dom.subdomain_horizontal_nodes * dom.subdomain_vertical_nodes == 8u);
    }

    SECTION("work-group size 7 (odd → stripe subdomains 1×7)")
    {
        auto props = make_props("nptl", 10, 10, 7);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.subdomain_vertical_nodes   == 1u);
        CHECK(dom.subdomain_horizontal_nodes == 7u);
    }
}

// ---------------------------------------------------------------------------
// npol — Non-linear Pull One-Lattice
// ---------------------------------------------------------------------------

TEST_CASE("Domain: npol structural invariants", "[domain][buffered]")
{
    SECTION("work-group size 4")
    {
        auto props = make_props("npol", 10, 10, 4);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 12u);
        CHECK(dom.vertical_nodes   >= 12u);
        CHECK(dom.subdomain_horizontal_nodes * dom.subdomain_vertical_nodes == 4u);
    }

    SECTION("work-group size 16 (power of 4 → square 4×4 subdomains)")
    {
        auto props = make_props("npol", 20, 20, 16);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.subdomain_horizontal_nodes == 4u);
        CHECK(dom.subdomain_vertical_nodes   == 4u);
    }
}

// ---------------------------------------------------------------------------
// nsol — Non-linear Swap One-Lattice
// ---------------------------------------------------------------------------

TEST_CASE("Domain: nsol structural invariants", "[domain][swap]")
{
    // wg must be >= 6 to avoid the JSON-correction branch
    SECTION("work-group size 8")
    {
        auto props = make_props("nsol", 10, 10, 8);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 1u);
        CHECK(dom.vertical_nodes   >= 1u);
    }

    SECTION("work-group size 64")
    {
        auto props = make_props("nsol", 30, 30, 64);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
    }
}
