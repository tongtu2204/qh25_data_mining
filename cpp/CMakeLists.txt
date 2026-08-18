cmake_minimum_required(VERSION 3.15)

project(jacobi_pca_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3)
endif()

add_executable(
    jacobi_serial
    jacobi_serial.cpp
)

find_package(OpenMP REQUIRED)

add_executable(
    jacobi_openmp
    jacobi_openmp.cpp
)

target_link_libraries(
    jacobi_openmp
    PRIVATE
    OpenMP::OpenMP_CXX
)