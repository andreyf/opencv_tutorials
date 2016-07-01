cmake_minimum_required(VERSION 2.8.12)

project(median_filter)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(median_filter main.cpp)
target_link_libraries(median_filter ${OpenCV_LIBS})
