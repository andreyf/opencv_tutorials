cmake_minimum_required(VERSION 2.8.12)

project(cardetect)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cardetect cardetect.cpp)
target_link_libraries(cardetect ${OpenCV_LIBS})