cmake_minimum_required(VERSION 3.20)
project(pcl_test)

# 设置编译模式
set( CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_FLAGS   "-std=c++14")

find_package(PCL  REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_test src/addpoints.cpp)

target_link_libraries (pcl_test ${PCL_LIBRARIES} ${OpenCV_LIBS})
