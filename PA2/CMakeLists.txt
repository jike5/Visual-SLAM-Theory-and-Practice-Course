cmake_minimum_required(VERSION 2.8)
project( PA2)

set( CMAKE_BUILD_TYPE "Release")
include_directories("/usr/include/eigen3")

add_executable(PA2_1 PA2_1_5.cpp )
add_executable(PA2_2 PA2_2.cpp)