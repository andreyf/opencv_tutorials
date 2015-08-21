cmake_minimum_required(VERSION 2.8.12)

project(opencv_cardetect)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_cardetect cardetect.cpp)
target_link_libraries(opencv_cardetect ${OpenCV_LIBS})