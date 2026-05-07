/**
 * @brief   Regression test: Hagen-Poiseuille velocity profile (plan item 4.6).
 *
 *          Runs a pressure-driven channel flow to steady state and verifies
 *          that the simulated x-velocity profile matches the analytical
 *          parabolic (Poiseuille) solution.
 *
 *          Domain
 *          -------
 *          inner_v = 14 vertical fluid nodes, inner_h = 30 horizontal nodes.
 *          Stored as 16 × 32 = 512 nodes; work-group size 16 divides evenly.
 *
 *          Convergence
 *          -----------
 *          τ = 1.0  →  ν = c_s² (τ − 0.5) = (1/6)
 *          Diffusion time scale ≈ 5 × H² / ν = 5 × 196 / (1/6) ≈ 5880 steps.
 *          We run 8 000 steps to be comfortably past steady state.
 *
 *          Assertions
 *          -----------
 *          Rows 0 and inner_v−1 of the result array are the wall-adjacent nodes
 *          that the bounceback scheme zeroes — they are excluded from all checks.
 *
 *          For the N = inner_v − 2 = 12 interior fluid rows the theoretical
 *          u_max/u_mean ratio (half-way bounceback) is ~1.48, approaching 1.5
 *          as N → ∞.  The test asserts the ratio is in [1.35, 1.55] — wide
 *          enough to be indifferent to the exact wall-placement convention while
 *          still catching non-parabolic profiles.
 *
 *          Additionally the test verifies:
 *            - All interior fluid-node x-velocities are positive (flow in +x).
 *            - The profile is left–right symmetric within 1 % of u_max.
 *            - The velocity peak is at the channel centre row.
 * @copyright   Copyright (c) 2026 Alexander Strack
 */

#include <catch2/catch_test_macros.hpp>

#include "lbm/portlbm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the x-velocity column profile at the given 0-indexed column.
static std::vector<lbm::real_type> column_profile(
    const std::vector<lbm::real_type> &x_vel,
    unsigned int inner_h,
    unsigned int inner_v,
    unsigned int col)
{
    std::vector<lbm::real_type> profile(inner_v);
    for (unsigned int row = 0; row < inner_v; ++row)
        profile[row] = x_vel[row * inner_h + col];
    return profile;
}

// ---------------------------------------------------------------------------
// Regression test
// ---------------------------------------------------------------------------

TEST_CASE("Hagen-Poiseuille regression: parabolic velocity profile", "[regression]")
{
    // ---- Simulation setup --------------------------------------------------

    lbm::core::Properties props(
        "lptl",  // algorithm
        "stream",                  // data layout
        false,                     // debug mode
        16,                        // work-group size (512 / 16 = 32 groups)
        8000,                      // time steps (well past diffusion time scale)
        8000,                      // frame update interval (single copy at end)
        "Hagen-Poiseuille",        // scenario
        14,                        // inner vertical nodes  (stored: 16)
        30,                        // inner horizontal nodes (stored: 32)
        0.0,                       // inlet velocity x
        0.0,                       // inlet velocity y
        1.01,                      // inlet density
        0.0,                       // outlet velocity x
        0.0,                       // outlet velocity y
        1.0,                       // outlet density
        1.0                        // relaxation time τ = 1.0  →  ν = 1/6
    );

    REQUIRE_NOTHROW(props.validate());

    auto handler = lbm::create_handler(props);
    handler->start();
    handler->block_until_finished();

    // ---- Retrieve inner-domain dimensions ----------------------------------

    const unsigned int inner_h = handler->get_horizontal_nodes() - 2u;  // 30
    const unsigned int inner_v = handler->get_vertical_nodes()   - 2u;  // 14

    const auto &x_vel = handler->get_x_velocities();
    REQUIRE(x_vel.size() == inner_h * inner_v);

    // ---- Extract the profile at the mid-channel column ---------------------
    //
    // Results are stored row-major in the inner domain:
    //   index = col + row * inner_h   (both 0-indexed)
    //
    // The centre column is the least disturbed by inlet/outlet boundary
    // effects and therefore the best place to measure the developed profile.

    const unsigned int centre_col = inner_h / 2u;
    const auto profile = column_profile(x_vel, inner_h, inner_v, centre_col);

    // Rows 0 and inner_v-1 are wall-adjacent nodes zeroed by the bounceback
    // scheme.  All checks operate on the interior fluid slice only.
    const unsigned int fluid_first = 1u;
    const unsigned int fluid_last  = inner_v - 2u;   // inclusive
    const unsigned int fluid_count = fluid_last - fluid_first + 1u;  // 12

    // ---- 1. All interior fluid velocities must be positive -----------------

    for (unsigned int row = fluid_first; row <= fluid_last; ++row)
    {
        INFO("row " << row << " has x-velocity " << profile[row]);
        CHECK(profile[row] > lbm::real_type{0});
    }

    // ---- 2. Profile must be symmetric about the channel centre -------------
    //
    // Compare each fluid row against its mirror; tolerance: 1 % of u_max.

    const lbm::real_type u_max = *std::max_element(
        profile.begin() + fluid_first,
        profile.begin() + fluid_last + 1u);

    REQUIRE(u_max > lbm::real_type{0});

    for (unsigned int i = fluid_first; i <= inner_v / 2u; ++i)
    {
        const unsigned int mirror = inner_v - 1u - i;
        const lbm::real_type diff = std::abs(profile[i] - profile[mirror]);
        INFO("symmetry mismatch at rows " << i << " and " << mirror
             << ": diff = " << diff << ", u_max = " << u_max);
        CHECK(diff / u_max < lbm::real_type{0.01});
    }

    // ---- 3. Peak must be at the centre row ---------------------------------

    const auto peak_it = std::max_element(
        profile.begin() + fluid_first,
        profile.begin() + fluid_last + 1u);
    const unsigned int peak_row =
        static_cast<unsigned int>(peak_it - profile.begin());

    // Allow ±1 row around the geometric centre of the fluid region
    const unsigned int fluid_centre = (fluid_first + fluid_last) / 2u;
    CHECK(peak_row >= fluid_centre - 1u);
    CHECK(peak_row <= fluid_centre + 1u);

    // ---- 4. Parabolic ratio: u_max / u_mean_fluid --------------------------
    //
    // For the continuous limit the ratio is exactly 1.5.  For the 12-row
    // discrete channel with half-way bounceback it is ~1.48.  The range
    // [1.35, 1.55] is wide enough to be indifferent to the exact wall-
    // placement convention while still catching non-parabolic profiles.

    const lbm::real_type u_mean_fluid =
        std::accumulate(
            profile.begin() + fluid_first,
            profile.begin() + fluid_last + 1u,
            lbm::real_type{0})
        / static_cast<lbm::real_type>(fluid_count);

    const lbm::real_type ratio = u_max / u_mean_fluid;

    INFO("u_max = " << u_max << ", u_mean_fluid = " << u_mean_fluid
         << ", ratio = " << ratio << " (expected 1.35 – 1.55)");

    CHECK(ratio > lbm::real_type{1.35});
    CHECK(ratio < lbm::real_type{1.55});
}
