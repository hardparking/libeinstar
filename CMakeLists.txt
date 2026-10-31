cmake_minimum_required(VERSION 3.20)
project(libeinstar VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(EINSTAR_BUILD_TOOLS "Build command-line tools" ON)
option(EINSTAR_BUILD_TESTS "Build unit tests" ON)
option(EINSTAR_WITH_OPENCV "Build einstar::depth (IR-stereo depth, needs OpenCV)" ON)
option(EINSTAR_BUILD_RTABMAP "Build the out-of-tree RTAB-Map camera and tools" OFF)

find_path(LIBUSB_INCLUDE_DIR libusb.h PATH_SUFFIXES libusb-1.0 REQUIRED)
find_library(LIBUSB_LIBRARY NAMES usb-1.0 libusb-1.0 REQUIRED)

set(EINSTAR_WARNINGS -Wall -Wextra -Wpedantic)

# Pure protocol code: no I/O, unit-testable without hardware.
add_library(einstar_core STATIC
  src/protocol.cpp
  src/stream.cpp
  src/calibration.cpp
  src/exposure.cpp)
target_include_directories(einstar_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(einstar_core PRIVATE ${EINSTAR_WARNINGS})
set_target_properties(einstar_core PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME core)
add_library(einstar::core ALIAS einstar_core)

add_library(einstar STATIC src/device.cpp)
target_link_libraries(einstar PUBLIC einstar_core PRIVATE ${LIBUSB_LIBRARY})
target_include_directories(einstar SYSTEM PRIVATE ${LIBUSB_INCLUDE_DIR})
target_compile_options(einstar PRIVATE ${EINSTAR_WARNINGS})
set_target_properties(einstar PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME einstar)
add_library(einstar::einstar ALIAS einstar)
set(EINSTAR_EXPORTED_TARGETS einstar_core einstar)
set(EINSTAR_HAS_DEPTH OFF)

if(EINSTAR_WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgproc calib3d)
  if(OpenCV_FOUND)
    add_library(einstar_depth STATIC src/depth.cpp)
    target_link_libraries(einstar_depth PUBLIC einstar_core ${OpenCV_LIBS})
    target_include_directories(einstar_depth SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_options(einstar_depth PRIVATE ${EINSTAR_WARNINGS})
    set_target_properties(einstar_depth PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME depth)
    add_library(einstar::depth ALIAS einstar_depth)
    list(APPEND EINSTAR_EXPORTED_TARGETS einstar_depth)
    set(EINSTAR_HAS_DEPTH ON)
  else()
    message(STATUS "OpenCV not found: einstar::depth disabled")
  endif()
endif()

if(EINSTAR_BUILD_TOOLS)
  add_executable(einstar-probe tools/einstar-probe.cpp)
  target_link_libraries(einstar-probe PRIVATE einstar)
  add_executable(einstar-calib tools/einstar-calib.cpp)
  target_link_libraries(einstar-calib PRIVATE einstar_core)
  install(TARGETS einstar-probe einstar-calib RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  if(TARGET einstar_depth)
    find_package(OpenCV REQUIRED COMPONENTS imgcodecs)
    add_executable(einstar-depth tools/einstar-depth.cpp)
    target_link_libraries(einstar-depth PRIVATE einstar_depth ${OpenCV_LIBS})
    add_executable(einstar-rgbd tools/einstar-rgbd.cpp)
    target_link_libraries(einstar-rgbd PRIVATE einstar einstar_depth ${OpenCV_LIBS})
  endif()
endif()

if(EINSTAR_BUILD_TESTS)
  enable_testing()
  add_executable(einstar-tests tests/test_main.cpp tests/test_protocol.cpp tests/test_stream.cpp
                               tests/test_calibration.cpp tests/test_device.cpp)
  target_link_libraries(einstar-tests PRIVATE einstar)
  add_test(NAME einstar-tests COMMAND einstar-tests)
  if(TARGET einstar_depth)
    find_package(OpenCV REQUIRED COMPONENTS imgcodecs)
    add_executable(einstar-depth-test tests/test_depth.cpp)
    target_link_libraries(einstar-depth-test PRIVATE einstar_depth ${OpenCV_LIBS})
    add_test(NAME einstar-depth-test COMMAND einstar-depth-test)
  endif()
endif()

if(EINSTAR_BUILD_RTABMAP)
  add_subdirectory(integrations/rtabmap)
endif()

# Installable CMake package: find_package(einstar) -> einstar::einstar,
# einstar::core and (when built) einstar::depth.
install(TARGETS ${EINSTAR_EXPORTED_TARGETS} EXPORT einstarTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/einstar DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT einstarTargets NAMESPACE einstar:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einstar)
configure_package_config_file(cmake/einstarConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/einstarConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einstar)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/einstarConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/einstarConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/einstarConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einstar)
