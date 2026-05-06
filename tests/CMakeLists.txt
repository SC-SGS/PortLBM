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

# ---- Smoke-test executable -------------------------------------------------

add_executable(portlbm_tests
  smoke_test.cpp
)

target_link_libraries(portlbm_tests
  PRIVATE
    portlbm_core
    Catch2::Catch2WithMain
)

# The test binary inherits SYCL compile options from portlbm_core (PUBLIC),
# but still needs add_sycl_to_target so AdaptiveCpp emits the runtime glue.
add_sycl_to_target(TARGET portlbm_tests)

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(portlbm_tests)
