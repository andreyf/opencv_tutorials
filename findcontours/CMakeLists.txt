cmake_minimum_required(VERSION 2.8.12)

project(apple)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(apple main.cpp)
target_link_libraries(apple ${OpenCV_LIBS})