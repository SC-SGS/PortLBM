# The top-level include/ directory is already added as a PUBLIC include root in
# the root CMakeLists.txt via $<BUILD_INTERFACE:...> / $<INSTALL_INTERFACE:...>,
# so consumers can write  #include <lbm/portlbm.hpp>  etc. without any extra
# per-subdirectory entries.
#
# This file only handles special cases:
#
#   - Vendored external headers (ImGui, ImPlot) are marked PRIVATE so their
#     symbols do not leak into the installed interface seen by consumers.
#
#   - lbm/gui is PUBLIC when visualization is enabled because its types appear
#     in the library's public API.

if(WITH_VISUALIZATION)
  # External GUI vendor headers — PRIVATE (not part of the installed API)
  target_include_directories(portlbm_core
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui"
      "${CMAKE_CURRENT_SOURCE_DIR}/external/implot"
  )
  # lbm/gui is part of the public API when WITH_VISUALIZATION is ON
  target_include_directories(portlbm_core
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/lbm/gui"
  )
endif()
