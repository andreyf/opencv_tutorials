cmake_minimum_required(VERSION 2.8.12)

project(blobdetector)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(blobdetector main.cpp)
target_link_libraries(blobdetector ${OpenCV_LIBS})