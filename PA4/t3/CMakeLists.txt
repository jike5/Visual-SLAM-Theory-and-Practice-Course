CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( GAUSSNEWTON )

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "/usr/include/eigen3" )

add_executable(gaussnewton gaussnewton.cpp)

target_link_libraries(gaussnewton ${OpenCV_LIBRARIES})