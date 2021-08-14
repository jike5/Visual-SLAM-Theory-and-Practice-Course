CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_executable(t1_hello t1_hello.cpp)
target_link_libraries(t1_hello ${CERES_LIBRARIES})

add_executable(t2_powell t2_powell.cpp)
target_link_libraries(t2_powell ${CERES_LIBRARIES})