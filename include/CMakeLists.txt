# The top-level include/ directory is already added as a PUBLIC include root in
# the root CMakeLists.txt via $<BUILD_INTERFACE:...> / $<INSTALL_INTERFACE:...>,
# so consumers can write  #include <lbm/portlbm.hpp>  or  #include <lbm/gui/lbm_gui.hpp>
# etc. without any extra per-subdirectory entries here.
#
# This file only handles one special case:
#
#   - Vendored external headers (ImGui, ImPlot) are marked PRIVATE so their
#     symbols do not leak into the installed interface seen by consumers.

if(WITH_VISUALIZATION)
  # External GUI vendor headers — PRIVATE (not part of the installed API)
  target_include_directories(portlbm_core
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui"
      "${CMAKE_CURRENT_SOURCE_DIR}/external/implot"
  )
endif()
