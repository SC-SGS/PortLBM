# Sources compiled into the portlbm_core library.
# NOTE: benchmark.cpp is intentionally excluded here — it is application-level
# code that is compiled only into the parallel_lbm executable (controlled by
# BENCHMARK_MODE in the root CMakeLists.txt).

set(portlbm_core_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/console/console_output.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/core/simulation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/exceptions/exceptions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/file_interaction/file_interaction.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/core/timer.cpp"
)

if(WITH_VISUALIZATION)
  list(APPEND portlbm_core_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/lbm/gui/lbm_gui.cpp"

    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_demo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_draw.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_impl_glfw.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_impl_opengl3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_tables.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui.cpp"

    "${CMAKE_CURRENT_SOURCE_DIR}/external/implot/implot_demo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/implot/implot_items.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/implot/implot.cpp"
  )
endif()

message(STATUS "portlbm_core sources:")
foreach(src ${portlbm_core_sources})
  message("   ${src}")
endforeach()

target_sources(portlbm_core PRIVATE ${portlbm_core_sources})
