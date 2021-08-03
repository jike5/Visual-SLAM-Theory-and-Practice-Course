cmake_minimum_required( VERSION 3.5)

project(hello)

set(CMAKE_BUILD_TYPE Debug)

include_directories("include")

add_library(libHello src/hello.cpp)

add_executable(hello main.cpp)

target_link_libraries(hello libHello)
