CMAKE_MINIMUM_REQUIRED( VERSION 3.5 )
PROJECT( CALCUL_DISPARITY )

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(calcul_disparity calcul_disparity.cpp)
target_link_libraries(calcul_disparity ${OpenCV_LIBS})
target_link_libraries(calcul_disparity ${Sophus_LIBRARIES})
target_link_libraries(calcul_disparity ${Pangolin_LIBRARIES})