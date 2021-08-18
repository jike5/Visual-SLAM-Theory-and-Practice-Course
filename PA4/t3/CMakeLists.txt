cmake_minimum_required(VERSION 2.8)
project(GN)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# opencv
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

include_directories( "/usr/include/eigen3" )

add_executable(gaussnewton gaussnewton.cpp)
target_link_libraries(gaussnewton ${OpenCV_LIBRARIES})

add_executable(gng2o gaussnewton_g2o.cpp)
target_link_libraries(gng2o ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})