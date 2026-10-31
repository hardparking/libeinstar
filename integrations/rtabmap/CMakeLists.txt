# Out-of-tree RTAB-Map camera, scanning app and synthetic-data tools.
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d imgcodecs)
find_package(RTABMap REQUIRED)
if(NOT TARGET einstar_depth)
  message(FATAL_ERROR "EINSTAR_BUILD_RTABMAP needs einstar::depth (OpenCV)")
endif()

# Use RTAB-Map's own CameraEinstar when it was built with libeinstar;
# otherwise compile the copy kept here (byte-identical to the corelib files).
set(RTABMAP_HAS_EINSTAR FALSE)
foreach(_dir ${RTABMap_INCLUDE_DIRS})
  if(EXISTS "${_dir}/rtabmap/core/Version.h")
    file(STRINGS "${_dir}/rtabmap/core/Version.h" _einstar_define REGEX "^#define RTABMAP_EINSTAR")
    if(_einstar_define)
      set(RTABMAP_HAS_EINSTAR TRUE)
    endif()
  endif()
endforeach()

add_library(einstar_rtabmap INTERFACE)
if(RTABMAP_HAS_EINSTAR)
  message(STATUS "RTAB-Map provides CameraEinstar")
  target_link_libraries(einstar_rtabmap INTERFACE rtabmap::core)
else()
  add_library(einstar_rtabmap_camera STATIC CameraEinstar.cpp)
  target_include_directories(einstar_rtabmap_camera BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(einstar_rtabmap_camera PUBLIC RTABMAP_EINSTAR)
  target_link_libraries(einstar_rtabmap_camera PUBLIC einstar einstar_depth rtabmap::core ${OpenCV_LIBS})
  target_link_libraries(einstar_rtabmap INTERFACE einstar_rtabmap_camera)
endif()

add_executable(einstar-rtabmap einstar_rtabmap.cpp)
target_link_libraries(einstar-rtabmap PRIVATE einstar_rtabmap)

add_executable(einstar-synth einstar_synth.cpp)
target_link_libraries(einstar-synth PRIVATE einstar_depth ${OpenCV_LIBS})
