cmake_minimum_required (VERSION 3.5)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated" )

add_subdirectory ( findcontours )
add_subdirectory ( watershed_and_distance_transform )
add_subdirectory ( car_detection/cardetect )
add_subdirectory ( generalized_hough_transform )
add_subdirectory ( opencv_filters/bilateral_filter )
add_subdirectory ( opencv_filters/gaussian_smoothing_filter )
add_subdirectory ( opencv_filters/homogeneous_smoothing_filter )
add_subdirectory ( opencv_filters/median_filter )
add_subdirectory ( blobdetector )

