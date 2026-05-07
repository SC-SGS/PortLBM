/**
 * @brief   Unit tests for lbm::core::access (plan item 4.2).
 *
 *          Covers:
 *            - CollisionAccessor::at  — AoS layout: index = 9*node + dir
 *            - StreamAccessor::at     — SoA layout: index = N*dir + node
 *            - BundleAccessor::at     — bundled-SoA: groups of 3 directions
 *            - get_node_index         — row-major (x, y) → linear
 *            - get_neighbor           — D2Q9 direction offsets
 *            - get_result_index       — three overloads stripping ghost nodes
 *            - power_functions        — is_power_of_4, which_power_of_4,
 *                                       which_power_of_2
 * @copyright   Copyright (c) 2026 Alexander Strack
 */

#include <catch2/catch_test_macros.hpp>

#include "lbm/core/access.hpp"
#include "lbm/core/simulation.hpp"  // power_functions

using namespace lbm::core::access;
using namespace lbm::core::power_functions;

// ---------------------------------------------------------------------------
// CollisionAccessor
// ---------------------------------------------------------------------------

TEST_CASE("CollisionAccessor: AoS layout index", "[access][collision]")
{
    // Formula: 9 * node + direction
    // The third argument (total_buffered_node_count) is ignored.
    constexpr unsigned int N = 999;  // unused sentinel

    CHECK(CollisionAccessor::at(0, 0, N) == 0);
    CHECK(CollisionAccessor::at(0, 8, N) == 8);
    CHECK(CollisionAccessor::at(1, 0, N) == 9);
    CHECK(CollisionAccessor::at(1, 8, N) == 17);
    CHECK(CollisionAccessor::at(5, 3, N) == 48);

    // Round-trip: recover node and direction from the flat index
    for (unsigned int node = 0; node < 4; ++node)
    {
        for (unsigned int dir = 0; dir < 9; ++dir)
        {
            unsigned int idx = CollisionAccessor::at(node, dir, N);
            CHECK(idx / 9 == node);
            CHECK(idx % 9 == dir);
        }
    }
}

// ---------------------------------------------------------------------------
// StreamAccessor
// ---------------------------------------------------------------------------

TEST_CASE("StreamAccessor: SoA layout index", "[access][stream]")
{
    // Formula: total_buffered_node_count * direction + node
    constexpr unsigned int total = 100;

    CHECK(StreamAccessor::at(0, 0, total) == 0);
    CHECK(StreamAccessor::at(1, 0, total) == 1);
    CHECK(StreamAccessor::at(0, 1, total) == 100);
    CHECK(StreamAccessor::at(5, 3, total) == 305);

    // Round-trip
    for (unsigned int node = 0; node < 5; ++node)
    {
        for (unsigned int dir = 0; dir < 9; ++dir)
        {
            unsigned int idx = StreamAccessor::at(node, dir, total);
            CHECK(idx / total == dir);
            CHECK(idx % total == node);
        }
    }
}

// ---------------------------------------------------------------------------
// BundleAccessor
// ---------------------------------------------------------------------------

TEST_CASE("BundleAccessor: bundled-SoA layout index", "[access][bundle]")
{
    // Formula: 3*(dir/3)*total + (dir%3) + 3*node
    // Directions 0-2 map to bundle 0, 3-5 to bundle 1, 6-8 to bundle 2.
    constexpr unsigned int total = 100;

    // Node 0, all 9 directions
    CHECK(BundleAccessor::at(0, 0, total) == 0);    // bundle 0, offset 0
    CHECK(BundleAccessor::at(0, 1, total) == 1);    // bundle 0, offset 1
    CHECK(BundleAccessor::at(0, 2, total) == 2);    // bundle 0, offset 2
    CHECK(BundleAccessor::at(0, 3, total) == 300);  // bundle 1, offset 0
    CHECK(BundleAccessor::at(0, 4, total) == 301);  // bundle 1, offset 1
    CHECK(BundleAccessor::at(0, 5, total) == 302);  // bundle 1, offset 2
    CHECK(BundleAccessor::at(0, 6, total) == 600);  // bundle 2, offset 0
    CHECK(BundleAccessor::at(0, 7, total) == 601);  // bundle 2, offset 1
    CHECK(BundleAccessor::at(0, 8, total) == 602);  // bundle 2, offset 2

    // Node 1: each index shifts by 3
    CHECK(BundleAccessor::at(1, 0, total) == 3);
    CHECK(BundleAccessor::at(1, 3, total) == 303);
    CHECK(BundleAccessor::at(1, 6, total) == 603);

    // Directions within the same bundle are adjacent; directions in different
    // bundles are separated by total*3 positions.
    for (unsigned int node = 0; node < 3; ++node)
    {
        for (unsigned int dir = 0; dir < 9; ++dir)
        {
            unsigned int expected =
                3u * (dir / 3u) * total + (dir % 3u) + 3u * node;
            CHECK(BundleAccessor::at(node, dir, total) == expected);
        }
    }
}

// ---------------------------------------------------------------------------
// get_node_index
// ---------------------------------------------------------------------------

TEST_CASE("get_node_index: row-major (x,y) to linear", "[access][node_index]")
{
    // Formula: x + y * horizontal_nodes
    constexpr unsigned int H = 10;

    CHECK(get_node_index(0, 0, H) == 0);
    CHECK(get_node_index(1, 0, H) == 1);
    CHECK(get_node_index(9, 0, H) == 9);
    CHECK(get_node_index(0, 1, H) == 10);
    CHECK(get_node_index(3, 2, H) == 23);

    for (unsigned int y = 0; y < 4; ++y)
        for (unsigned int x = 0; x < H; ++x)
            CHECK(get_node_index(x, y, H) == x + y * H);
}

// ---------------------------------------------------------------------------
// get_neighbor
// ---------------------------------------------------------------------------

TEST_CASE("get_neighbor: D2Q9 direction offsets", "[access][neighbor]")
{
    // D2Q9 direction encoding:
    //   dir / 3 - 1  gives the y offset: -1 for dirs 0-2, 0 for 3-5, +1 for 6-8
    //   dir % 3 - 1  gives the x offset: -1 for dirs 0,3,6; 0 for 1,4,7; +1 for 2,5,8
    constexpr unsigned int H = 10;
    constexpr unsigned int N = 55;  // a node in the middle of a large grid

    // Self (rest direction = 4)
    CHECK(get_neighbor(N, 4, H) == N);

    // Cardinal directions
    CHECK(get_neighbor(N, 5, H) == N + 1);   // right  (+x)
    CHECK(get_neighbor(N, 3, H) == N - 1);   // left   (-x)
    CHECK(get_neighbor(N, 7, H) == N + H);   // up     (+y)
    CHECK(get_neighbor(N, 1, H) == N - H);   // down   (-y)

    // Diagonals
    CHECK(get_neighbor(N, 8, H) == N + H + 1);  // up-right
    CHECK(get_neighbor(N, 6, H) == N + H - 1);  // up-left
    CHECK(get_neighbor(N, 2, H) == N - H + 1);  // down-right
    CHECK(get_neighbor(N, 0, H) == N - H - 1);  // down-left
}

// ---------------------------------------------------------------------------
// get_result_index
// ---------------------------------------------------------------------------

TEST_CASE("get_result_index (x, y, horiz): strips ghost layer", "[access][result_index]")
{
    // Formula: (x-1) + (y-1) * (horiz - 2)
    // Ghost nodes are at row/column 0 and (horiz/vert - 1).
    // Inner nodes start at (1,1).
    constexpr unsigned int H = 10;

    CHECK(get_result_index(1u, 1u, H) == 0u);             // first inner node
    CHECK(get_result_index(2u, 1u, H) == 1u);             // second node, first row
    CHECK(get_result_index(8u, 1u, H) == 7u);             // last inner node, first row (H-2 = 8 inner cols)
    CHECK(get_result_index(1u, 2u, H) == H - 2u);         // first node, second row
    CHECK(get_result_index(2u, 2u, H) == H - 2u + 1u);
}

TEST_CASE("get_result_index (x, y, horiz, domain_count, step): time-series offset",
          "[access][result_index]")
{
    // Formula: (x-1) + (y-1)*(horiz-2) + step * domain_node_count
    constexpr unsigned int H = 10;
    constexpr unsigned int domain_count = 64;

    CHECK(get_result_index(1u, 1u, H, domain_count, 0u) == 0u);
    CHECK(get_result_index(1u, 1u, H, domain_count, 1u) == domain_count);
    CHECK(get_result_index(2u, 1u, H, domain_count, 2u) == 1u + 2u * domain_count);
}

TEST_CASE("get_result_index (node_index, horiz): linear form", "[access][result_index]")
{
    // Formula: ((node_index - H) / H) * (H-2) + (node_index - 1) % H
    // The first inner row begins at linear index H (row y=1).
    constexpr unsigned int H = 10;

    // Node at (1, 1) has linear index H + 1
    CHECK(get_result_index(H + 1u, H) == 0u);

    // Node at (2, 1) has linear index H + 2
    CHECK(get_result_index(H + 2u, H) == 1u);

    // Node at (1, 2) has linear index 2*H + 1
    CHECK(get_result_index(2u * H + 1u, H) == H - 2u);
}

// ---------------------------------------------------------------------------
// power_functions
// ---------------------------------------------------------------------------

TEST_CASE("is_power_of_4", "[power_functions]")
{
    // 1 is odd — returns false
    CHECK_FALSE(is_power_of_4(1));

    CHECK(is_power_of_4(4));
    CHECK(is_power_of_4(16));
    CHECK(is_power_of_4(64));
    CHECK(is_power_of_4(256));

    CHECK_FALSE(is_power_of_4(2));
    CHECK_FALSE(is_power_of_4(8));
    CHECK_FALSE(is_power_of_4(32));
    CHECK_FALSE(is_power_of_4(6));
}

TEST_CASE("which_power_of_4: returns exponent or 0", "[power_functions]")
{
    // 1 is odd — returns 0
    CHECK(which_power_of_4(1) == 0);

    CHECK(which_power_of_4(4)   == 1);
    CHECK(which_power_of_4(16)  == 2);
    CHECK(which_power_of_4(64)  == 3);
    CHECK(which_power_of_4(256) == 4);

    // Powers of 2 that are not powers of 4 return 0
    CHECK(which_power_of_4(2)  == 0);
    CHECK(which_power_of_4(8)  == 0);
    CHECK(which_power_of_4(32) == 0);
}

TEST_CASE("which_power_of_2: returns exponent or 0", "[power_functions]")
{
    // 1 is odd — returns 0
    CHECK(which_power_of_2(1) == 0);

    CHECK(which_power_of_2(2)  == 1);
    CHECK(which_power_of_2(4)  == 2);
    CHECK(which_power_of_2(8)  == 3);
    CHECK(which_power_of_2(16) == 4);
    CHECK(which_power_of_2(32) == 5);
    CHECK(which_power_of_2(64) == 6);

    // Non-powers of 2 return 0
    CHECK(which_power_of_2(3)  == 0);
    CHECK(which_power_of_2(6)  == 0);
    CHECK(which_power_of_2(12) == 0);
    CHECK(which_power_of_2(24) == 0);
}
