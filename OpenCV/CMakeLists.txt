cmake_minimum_required(VERSION 3.10)

project("OpenCV practice" LANGUAGES CXX)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/build_opencv)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(AffineTransformations src/AffineTransformations.cpp)

target_link_libraries(AffineTransformations PRIVATE ${OpenCV_LIBS})
