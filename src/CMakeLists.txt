set(source_files "${CMAKE_CURRENT_SOURCE_DIR}/lbm/console/console_output.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/core/access.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/core/simulation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/exceptions/exceptions.cpp"  
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/file_interaction/file_interaction.cpp"  
)

if(WITH_VISUALIZATION)

set(source_files ${source_files}
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/execution/sycl_executor.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/gui/lbm_gui.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/lbm/execution/simple_timer.cpp"
)

endif(WITH_VISUALIZATION)

message(STATUS "Added sources:")

foreach(source ${source_files})
  message("   ${source}")
endforeach()

target_sources(${PROJECT_NAME} PUBLIC ${source_files})