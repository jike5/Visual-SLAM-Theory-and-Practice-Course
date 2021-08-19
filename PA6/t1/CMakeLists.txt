CMAKE_MINIMUM_REQUIRED( VERSION 3.5 )
PROJECT( OPTICAL_FLOW)

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})