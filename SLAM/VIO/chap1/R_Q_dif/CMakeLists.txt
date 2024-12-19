cmake_minimum_required(VERSION 3.10)
project(R_Q_dif)

find_package(Sophus REQUIRED)
#find_package(Eigen3 REQUIRED)
include_directories("/usr/local/include/eigen3" ${EIGEN3_INCLUDE_DIR})

message(Eigen: ${EIGEN3_INCLUDE_DIR})

add_executable(R_Q_dif R_Q_dif.cpp)

target_link_libraries(R_Q_dif ${Sophus_LIBRARIES} fmt)