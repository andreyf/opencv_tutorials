cmake_minimum_required(VERSION 2.8.12)

project(findcontours)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(findcontours main.cpp)
target_link_libraries(findcontours ${OpenCV_LIBS})