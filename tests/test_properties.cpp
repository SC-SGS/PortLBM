/**
 * @brief   Unit tests for lbm::core::Properties (plan item 4.4).
 *
 *          Covers:
 *            - Constructor field storage (ghost-layer accounting, counts)
 *            - validate(): accepts all valid combinations
 *            - validate(): rejects each class of invalid input
 *            - json_to_properties(): parses the bundled settings/settings.json
 *            - Round-trip: Properties → JSON → Properties preserves all fields
 *
 *          The path to the settings file is injected at compile time via the
 *          PORTLBM_SETTINGS_JSON_PATH macro (defined in tests/CMakeLists.txt).
 * @copyright   Copyright (c) 2026 Alexander Strack
 */

#include <catch2/catch_test_macros.hpp>

#include "lbm/core/properties.hpp"
#include "lbm/exceptions/exceptions.hpp"
#include "lbm/file_interaction/file_interaction.hpp"

#include <filesystem>
#include <string>

// ---------------------------------------------------------------------------
// Helper: a known-good Properties
// ---------------------------------------------------------------------------

static lbm::core::Properties valid_props()
{
    return lbm::core::Properties(
        "lptl", "stream", false,
        /*work_group_size=*/64,
        /*time_steps=*/100, /*frame_interval=*/100,
        "Hagen-Poiseuille",
        /*inner_v=*/30, /*inner_h=*/126,
        0.0, 0.0, 1.3, 0.0, 0.0, 1.0, 0.6);
}

// ---------------------------------------------------------------------------
// Constructor field storage
// ---------------------------------------------------------------------------

TEST_CASE("Properties constructor: field storage and ghost-layer accounting",
          "[properties][constructor]")
{
    auto p = valid_props();

    // Strings are stored as-is
    CHECK(p.algorithm   == "lptl");
    CHECK(p.data_layout == "stream");
    CHECK(p.scenario    == "Hagen-Poiseuille");
    CHECK(p.debug_mode  == false);

    // Ghost layers: stored = inner + 2
    CHECK(p.vertical_nodes   == 32u);   // 30 + 2
    CHECK(p.horizontal_nodes == 128u);  // 126 + 2

    // Node counts
    CHECK(p.domain_node_count           == 30u * 126u);         // inner only
    CHECK(p.total_unexpanded_node_count == 32u * 128u);         // with ghosts

    // Physical parameters
    CHECK(p.inlet_density   == 1.3);
    CHECK(p.outlet_density  == 1.0);
    CHECK(p.relaxation_time == 0.6);
    CHECK(p.work_group_size == 64u);
    CHECK(p.time_steps      == 100u);

    // Programmatically constructed Properties have no settings path
    CHECK(p.settings_path.empty());
}

// ---------------------------------------------------------------------------
// validate(): accepts valid inputs
// ---------------------------------------------------------------------------

TEST_CASE("Properties::validate(): valid inputs do not throw", "[properties][validate]")
{
    SECTION("all four algorithms")
    {
        for (const auto &alg : {"nptl", "lptl",
                                "npol", "nsol"})
        {
            auto p = valid_props();
            p.algorithm = alg;
            CHECK_NOTHROW(p.validate());
        }
    }

    SECTION("all three data layouts")
    {
        for (const auto &layout : {"stream", "collision", "bundle"})
        {
            auto p = valid_props();
            p.data_layout = layout;
            CHECK_NOTHROW(p.validate());
        }
    }

    SECTION("all eight scenarios")
    {
        for (const auto &sc : {"Hagen-Poiseuille", "walls", "circle", "square",
                               "plate", "skyscraper", "wing", "porous"})
        {
            auto p = valid_props();
            p.scenario = sc;
            CHECK_NOTHROW(p.validate());
        }
    }

    SECTION("minimum valid node counts (inner = 2 → stored = 4)")
    {
        // Re-construct so ghost accounting is correct
        lbm::core::Properties p(
            "lptl", "stream", false, 1, 1, 1,
            "Hagen-Poiseuille", 2, 2, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5);
        CHECK_NOTHROW(p.validate());
    }
}

// ---------------------------------------------------------------------------
// validate(): rejects invalid inputs
// ---------------------------------------------------------------------------

TEST_CASE("Properties::validate(): invalid inputs throw PropertyArgumentException",
          "[properties][validate][invalid]")
{
    using Exc = lbm::exceptions::json::PropertyArgumentException;

    SECTION("vertical_nodes below minimum (inner = 1 → stored = 3 < 4)")
    {
        lbm::core::Properties p(
            "lptl", "stream", false, 64, 100, 100,
            "Hagen-Poiseuille", 1, 30, 0.0, 0.0, 1.3, 0.0, 0.0, 1.0, 0.6);
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("horizontal_nodes below minimum (inner = 1 → stored = 3 < 4)")
    {
        lbm::core::Properties p(
            "lptl", "stream", false, 64, 100, 100,
            "Hagen-Poiseuille", 30, 1, 0.0, 0.0, 1.3, 0.0, 0.0, 1.0, 0.6);
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("time_steps == 0")
    {
        auto p = valid_props();
        p.time_steps = 0;
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("work_group_size == 0")
    {
        auto p = valid_props();
        p.work_group_size = 0;
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("inlet_density == 0")
    {
        auto p = valid_props();
        p.inlet_density = 0.0;
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("inlet_density negative")
    {
        auto p = valid_props();
        p.inlet_density = -1.0;
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("outlet_density == 0")
    {
        auto p = valid_props();
        p.outlet_density = 0.0;
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("relaxation_time == 0")
    {
        auto p = valid_props();
        p.relaxation_time = 0.0;
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("unknown algorithm string")
    {
        auto p = valid_props();
        p.algorithm = "not-an-algorithm";
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("unknown data_layout string")
    {
        auto p = valid_props();
        p.data_layout = "not-a-layout";
        CHECK_THROWS_AS(p.validate(), Exc);
    }

    SECTION("unknown scenario string")
    {
        auto p = valid_props();
        p.scenario = "not-a-scenario";
        CHECK_THROWS_AS(p.validate(), Exc);
    }
}

// ---------------------------------------------------------------------------
// json_to_properties: parses the bundled settings.json
// ---------------------------------------------------------------------------

TEST_CASE("json_to_properties: parses bundled settings/settings.json",
          "[properties][json]")
{
    const std::string path = PORTLBM_SETTINGS_JSON_PATH;
    REQUIRE(std::filesystem::exists(path));

    lbm::core::Properties p = lbm::file_interaction::json_to_properties(path);

    CHECK(p.algorithm        == "nptl");
    CHECK(p.data_layout      == "stream");
    CHECK(p.scenario         == "Hagen-Poiseuille");
    CHECK(p.debug_mode       == false);
    CHECK(p.time_steps       == 10000u);
    CHECK(p.work_group_size  == 1024u);
    CHECK(p.inlet_density    > 0.0);
    CHECK(p.outlet_density   > 0.0);
    CHECK(p.relaxation_time  > 0.0);
    CHECK(p.settings_path    == path);

    CHECK_NOTHROW(p.validate());
}

// ---------------------------------------------------------------------------
// Round-trip: Properties → JSON file → Properties
// ---------------------------------------------------------------------------

TEST_CASE("json round-trip: properties_to_json → json_to_properties",
          "[properties][json][round-trip]")
{
    auto orig = valid_props();
    // Give it a temporary path so properties_to_json has somewhere to write
    const auto tmp = std::filesystem::temp_directory_path() / "portlbm_test_roundtrip.json";
    orig.settings_path = tmp.string();

    lbm::file_interaction::properties_to_json(orig, tmp.string());
    REQUIRE(std::filesystem::exists(tmp));

    // Read back with offset=0 (the stored node counts already include ghosts;
    // json_to_properties adds the offset to the values it reads, which were
    // stored without ghosts, so use the default offset=0).
    lbm::core::Properties loaded = lbm::file_interaction::json_to_properties(tmp.string());

    CHECK(loaded.algorithm        == orig.algorithm);
    CHECK(loaded.data_layout      == orig.data_layout);
    CHECK(loaded.scenario         == orig.scenario);
    CHECK(loaded.debug_mode       == orig.debug_mode);
    CHECK(loaded.time_steps       == orig.time_steps);
    CHECK(loaded.work_group_size  == orig.work_group_size);
    CHECK(loaded.inlet_density    == orig.inlet_density);
    CHECK(loaded.outlet_density   == orig.outlet_density);
    CHECK(loaded.relaxation_time  == orig.relaxation_time);

    std::filesystem::remove(tmp);
}
