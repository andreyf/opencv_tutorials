cmake_minimum_required(VERSION 2.8.12)

project(gaussian_smoothing_filter)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gaussian_smoothing_filter main.cpp)
target_link_libraries(gaussian_smoothing_filter ${OpenCV_LIBS})
