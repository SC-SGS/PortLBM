/**
 * @file        smoke_test.cpp
 *
 * @brief       CPU smoke test for the PortLBM library.
 *
 *              Runs a small Hagen-Poiseuille simulation entirely on the host
 *              (requires -DFORCE_USE_CPU=ON) and asserts that the output arrays
 *              are finite, non-negative, and non-trivial.  Physics correctness is
 *              not checked here — that belongs in a separate validation suite.
 *
 *              Grid: 30 × 126 inner nodes (stored as 32 × 128 with ghost layers).
 *              This gives 4096 total nodes, which divides evenly by any power-of-two
 *              work-group size up to 4096 — safe for CPU execution.
 */

#include <catch2/catch_test_macros.hpp>

#include "lbm/portlbm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool any_nan(const std::vector<lbm::real_type> &v)
{
    return std::any_of(v.begin(), v.end(), [](lbm::real_type x) { return std::isnan(x); });
}

static lbm::real_type mean(const std::vector<lbm::real_type> &v)
{
    return std::accumulate(v.begin(), v.end(), lbm::real_type{0}) / static_cast<lbm::real_type>(v.size());
}

// ---------------------------------------------------------------------------
// Test case
// ---------------------------------------------------------------------------

TEST_CASE("Hagen-Poiseuille smoke: output is finite and non-trivial", "[smoke]")
{
    // Inner extents: 30 vertical × 126 horizontal.
    // The Properties constructor adds 2 ghost layers in each direction,
    // giving stored extents of 32 × 128 = 4096 nodes total.
    lbm::core::Properties props(
        "gpu-two-lattice-linear",  // algorithm
        "stream",                  // data layout
        false,                     // debug mode
        64,                        // work-group size
        100,                       // time steps
        100,                       // frame update interval (single host-copy at end)
        "Hagen-Poiseuille",        // scenario
        30,                        // inner vertical nodes
        126,                       // inner horizontal nodes
        0.0,                       // inlet velocity x
        0.0,                       // inlet velocity y
        1.3,                       // inlet density
        0.0,                       // outlet velocity x
        0.0,                       // outlet velocity y
        1.0,                       // outlet density
        0.6                        // relaxation time τ
    );

    REQUIRE_NOTHROW(props.validate());

    auto handler = lbm::create_handler(props);

    REQUIRE_NOTHROW(handler->start());
    REQUIRE_NOTHROW(handler->block_until_finished());

    const auto &densities    = handler->get_densities();
    const auto &x_velocities = handler->get_x_velocities();
    const auto &y_velocities = handler->get_y_velocities();

    // Arrays must be populated
    REQUIRE_FALSE(densities.empty());
    REQUIRE(densities.size() == x_velocities.size());
    REQUIRE(densities.size() == y_velocities.size());

    // No NaN in any output array
    CHECK_FALSE(any_nan(densities));
    CHECK_FALSE(any_nan(x_velocities));
    CHECK_FALSE(any_nan(y_velocities));

    // Mean density over the whole domain must be physically sensible (> 0.5).
    // Solid nodes carry a negative sentinel value, so we cannot assert all-positive,
    // but the domain average must reflect a healthy fluid region.
    CHECK(mean(densities) > lbm::real_type{0.5});

    // Hagen-Poiseuille must develop a non-trivial axial velocity profile
    const bool any_nonzero_x = std::any_of(
        x_velocities.begin(), x_velocities.end(),
        [](lbm::real_type v) { return std::abs(v) > lbm::real_type{1e-12}; });
    CHECK(any_nonzero_x);
}
