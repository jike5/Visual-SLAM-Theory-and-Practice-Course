CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( Disparity )
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")

MESSAGE(STATUS "Include: " ${Pangolin_INCLUDE_DIRS})
MESSAGE(STATUS "Linrary: " ${Pangolin_LIBRARIES})
MESSAGE(STATUS "Linrary: " ${OpenCV_LIBRARIES})
MESSAGE(STATUS "Linrary: " ${OpenCV_LIBS})

add_executable(disparity disparity.cpp)

target_link_libraries(disparity ${OpenCV_LIBS})
target_link_libraries(disparity ${Pangolin_LIBRARIES})