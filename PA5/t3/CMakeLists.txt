CMAKE_MINIMUM_REQUIRED( VERSION 3.5 )
PROJECT( GN-BA )

find_package(Sophus REQUIRED)

include_directories("/usr/include/eigen3")

add_executable(gn_ba GN-BA.cpp)
target_link_libraries(gn_ba ${Sophus_LIBRARIES})
message(STATUS "Sophus" ${Sophus_LIBRARIES})