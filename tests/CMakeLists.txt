# Tests for the PortLBM library.
# Enabled via -DPORTLBM_BUILD_TESTS=ON.
# Requires -DFORCE_USE_CPU=ON for execution in headless / CI environments.

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.6.0
  GIT_SHALLOW    TRUE
)

set(CATCH_INSTALL_DOCS    OFF CACHE INTERNAL "")
set(CATCH_INSTALL_EXTRAS  OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

# ---- Helper: absolute path to the bundled settings file --------------------

# Passed as a compile definition so test_properties.cpp can locate it at
# runtime without hard-coding repository-layout assumptions.
set(PORTLBM_SETTINGS_JSON
    "${CMAKE_SOURCE_DIR}/settings/settings.json"
    CACHE FILEPATH "Path to the bundled settings/settings.json used in tests")

# ---- Integration smoke test ------------------------------------------------

add_executable(portlbm_smoke_tests
  smoke_test.cpp
)

target_link_libraries(portlbm_smoke_tests
  PRIVATE
    portlbm_core
    Catch2::Catch2WithMain
)

add_sycl_to_target(TARGET portlbm_smoke_tests)
catch_discover_tests(portlbm_smoke_tests)

# ---- Unit tests (SYCL-free logic) ------------------------------------------

add_executable(portlbm_unit_tests
  test_access.cpp
  test_domain.cpp
  test_properties.cpp
)

target_link_libraries(portlbm_unit_tests
  PRIVATE
    portlbm_core
    Catch2::Catch2WithMain
)

target_compile_definitions(portlbm_unit_tests
  PRIVATE
    PORTLBM_SETTINGS_JSON_PATH="${PORTLBM_SETTINGS_JSON}"
)

# Unit tests use portlbm_core which was built with SYCL, so the linker needs
# the SYCL runtime — add_sycl_to_target handles that without recompiling tests.
add_sycl_to_target(TARGET portlbm_unit_tests)
catch_discover_tests(portlbm_unit_tests)
