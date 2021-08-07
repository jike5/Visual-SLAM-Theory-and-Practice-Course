CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( DrawTrajectory )
SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")

FIND_PACKAGE(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

FIND_PACKAGE(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_executable(draw_trajectory draw_trajectory.cpp)
target_link_libraries(draw_trajectory ${Sophus_LIBRARIES})
target_link_libraries(draw_trajectory ${Pangolin_LIBRARIES})
