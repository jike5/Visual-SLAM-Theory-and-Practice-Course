CMAKE_MINIMUM_REQUIRED( VERSION 3.5 )
PROJECT( TRACKALIGNMENT )
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3")

include_directories("/usr/include/eigen3")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(tracksvd TrackAlignment_SVD.cpp)
target_link_libraries(tracksvd ${Sophus_LIBRARIES})
target_link_libraries(tracksvd ${Pangolin_LIBRARIES})
