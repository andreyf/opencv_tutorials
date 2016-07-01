cmake_minimum_required(VERSION 2.8.12)

project(bilateral_filter)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bilateral_filter main.cpp)
target_link_libraries(bilateral_filter ${OpenCV_LIBS})
