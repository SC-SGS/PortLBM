/**
 * @file    test_domain.cpp
 *
 * @brief   Unit tests for lbm::core::Domain (plan item 4.3).
 *
 *          Covers the domain-layout math for each algorithm variant:
 *            - gpu-two-lattice-linear  — no decomposition; exact node counts
 *            - gpu-two-lattice         — decomposed layout; structural invariants
 *            - gpu-two-lattice-buffered — buffered layout; structural invariants
 *            - gpu-swap                — swap layout; structural invariants
 *
 *          For the decomposed/buffered/swap algorithms the exact expanded
 *          extents depend on subdomain arithmetic that is sensitive to grid
 *          size and work-group size, so we verify structural invariants
 *          (total == h * v, domain not smaller than requested) rather than
 *          hard-coded values.  For the linear variant (no expansion) we can
 *          assert exact values.
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
// gpu-two-lattice-linear  (no subdomain decomposition)
// ---------------------------------------------------------------------------

TEST_CASE("Domain: gpu-two-lattice-linear preserves ghost-extended extents",
          "[domain][linear]")
{
    // The linear algorithm does not call any setup_for_* method, so Domain
    // stores the ghost-extended extents from Properties directly.
    //
    // inner_v=10, inner_h=6  →  stored_v=12, stored_h=8
    auto props = make_props("gpu-two-lattice-linear", 10, 6, 64);

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
// gpu-two-lattice  (decomposed, non-buffered)
// ---------------------------------------------------------------------------

TEST_CASE("Domain: gpu-two-lattice structural invariants", "[domain][decomposed]")
{
    SECTION("work-group size 4 (power of 4 → square 2×2 subdomains)")
    {
        // inner 10×10 → stored 12×12
        auto props = make_props("gpu-two-lattice", 10, 10, 4);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 12u);
        CHECK(dom.vertical_nodes   >= 12u);
        CHECK(dom.subdomain_horizontal_nodes == 2u);
        CHECK(dom.subdomain_vertical_nodes   == 2u);
    }

    SECTION("work-group size 8 (power of 2, not 4 → rectangular 2×4 subdomains)")
    {
        auto props = make_props("gpu-two-lattice", 10, 10, 8);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 12u);
        CHECK(dom.vertical_nodes   >= 12u);
        // subdomain_h * subdomain_v == work_group_size
        CHECK(dom.subdomain_horizontal_nodes * dom.subdomain_vertical_nodes == 8u);
    }

    SECTION("work-group size 7 (odd → stripe subdomains 1×7)")
    {
        auto props = make_props("gpu-two-lattice", 10, 10, 7);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.subdomain_vertical_nodes   == 1u);
        CHECK(dom.subdomain_horizontal_nodes == 7u);
    }
}

// ---------------------------------------------------------------------------
// gpu-two-lattice-buffered
// ---------------------------------------------------------------------------

TEST_CASE("Domain: gpu-two-lattice-buffered structural invariants", "[domain][buffered]")
{
    SECTION("work-group size 4")
    {
        auto props = make_props("gpu-two-lattice-buffered", 10, 10, 4);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 12u);
        CHECK(dom.vertical_nodes   >= 12u);
        CHECK(dom.subdomain_horizontal_nodes * dom.subdomain_vertical_nodes == 4u);
    }

    SECTION("work-group size 16 (power of 4 → square 4×4 subdomains)")
    {
        auto props = make_props("gpu-two-lattice-buffered", 20, 20, 16);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.subdomain_horizontal_nodes == 4u);
        CHECK(dom.subdomain_vertical_nodes   == 4u);
    }
}

// ---------------------------------------------------------------------------
// gpu-swap
// ---------------------------------------------------------------------------

TEST_CASE("Domain: gpu-swap structural invariants", "[domain][swap]")
{
    // wg must be >= 6 to avoid the JSON-correction branch
    SECTION("work-group size 8")
    {
        auto props = make_props("gpu-swap", 10, 10, 8);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
        CHECK(dom.horizontal_nodes >= 1u);
        CHECK(dom.vertical_nodes   >= 1u);
    }

    SECTION("work-group size 64")
    {
        auto props = make_props("gpu-swap", 30, 30, 64);
        lbm::core::Domain dom(props);

        CHECK(dom.total_node_count == dom.horizontal_nodes * dom.vertical_nodes);
    }
}
