cmake_minimum_required(VERSION 2.8.12)

project(watershed)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(watershed main.cpp)
target_link_libraries(watershed ${OpenCV_LIBS})