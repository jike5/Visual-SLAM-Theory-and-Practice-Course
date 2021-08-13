CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT(COMPUTEORB)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(computeORB computeORB.cpp)
target_link_libraries(computeORB ${OpenCV_LIBS})