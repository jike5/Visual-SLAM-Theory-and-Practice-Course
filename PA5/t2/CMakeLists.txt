CMAKE_MINIMUM_REQUIRED( VERSION 3.5 )
PROJECT( E2Rt )
find_package( Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_executable(e2rt E2Rt.cpp)
target_link_libraries(e2rt ${Sophus_LIBRARIES})