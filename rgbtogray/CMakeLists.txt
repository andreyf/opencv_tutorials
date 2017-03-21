cmake_minimum_required(VERSION 2.8.12)

project(rgbtogray)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(rgbtogray main.cpp)
target_link_libraries(rgbtogray ${OpenCV_LIBS})