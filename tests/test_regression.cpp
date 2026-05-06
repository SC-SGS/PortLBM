/**
 * @file    test_regression.cpp
 *
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
 *          For any parabola u(y) = A·y·(H−y) the ratio u_max/u_mean is exactly
 *          1.5, independent of wall position or absolute pressure.  This is the
 *          primary regression invariant and is checked within 3 %.
 *
 *          Additionally the test verifies:
 *            - All fluid-node x-velocities are positive (flow is in +x).
 *            - The profile is left–right symmetric within 1 % of u_max.
 *            - The velocity peak is at the channel centre row.
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
        "gpu-two-lattice-linear",  // algorithm
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

    // ---- 1. All fluid velocities must be positive --------------------------

    for (unsigned int row = 0; row < inner_v; ++row)
    {
        INFO("row " << row << " has x-velocity " << profile[row]);
        CHECK(profile[row] > lbm::real_type{0});
    }

    // ---- 2. Profile must be symmetric about the channel centre -------------
    //
    // Tolerance: 1 % of the peak velocity.

    const lbm::real_type u_max =
        *std::max_element(profile.begin(), profile.end());

    REQUIRE(u_max > lbm::real_type{0});

    for (unsigned int i = 0; i < inner_v / 2u; ++i)
    {
        const lbm::real_type diff = std::abs(profile[i] - profile[inner_v - 1u - i]);
        INFO("symmetry mismatch at rows " << i << " and " << (inner_v - 1u - i)
             << ": diff = " << diff << ", u_max = " << u_max);
        CHECK(diff / u_max < lbm::real_type{0.01});
    }

    // ---- 3. Peak must be at the centre row ---------------------------------

    const auto peak_it = std::max_element(profile.begin(), profile.end());
    const unsigned int peak_row =
        static_cast<unsigned int>(peak_it - profile.begin());

    // Allow ±1 row around the geometric centre
    CHECK(peak_row >= inner_v / 2u - 1u);
    CHECK(peak_row <= inner_v / 2u + 1u);

    // ---- 4. Parabolic ratio: u_max / u_mean = 1.5  (±3 %) -----------------
    //
    // For u(y) = A·y·(H−y) the mean over [0,H] is A·H²/6 and the peak is
    // A·H²/4, giving an exact ratio of 1.5 regardless of A or H.  This holds
    // for a channel with no-slip walls even when the lattice resolution is
    // modest, as long as the flow is fully developed.

    const lbm::real_type u_mean =
        std::accumulate(profile.begin(), profile.end(), lbm::real_type{0})
        / static_cast<lbm::real_type>(inner_v);

    const lbm::real_type ratio = u_max / u_mean;

    INFO("u_max = " << u_max << ", u_mean = " << u_mean
         << ", ratio = " << ratio << " (expected ≈ 1.5)");

    CHECK(ratio > lbm::real_type{1.5 * 0.97});   // lower bound: 1.455
    CHECK(ratio < lbm::real_type{1.5 * 1.03});   // upper bound: 1.545
}
