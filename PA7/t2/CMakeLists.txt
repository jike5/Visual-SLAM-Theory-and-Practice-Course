CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( DIRECTBA )

set(CMAKE_CXX_STANDARD 11 )
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)
FIND_PACKAGE(Pangolin REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse g2o_types_sba)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS}  ${OpenCV_INCLUDE_DIRS})

add_executable(directBA directBA.cpp)
target_link_libraries(directBA ${G2O_LIBS} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
