cmake_minimum_required(VERSION 3.18)
project(ImageFilterGPU-Okab LANGUAGES CXX CUDA)

# Project info
set(PROJECT_VERSION "1.0.0")
set(PROJECT_DESCRIPTION "GPU-accelerated image filtering with CUDA")

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# CUDA architectures (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Source files
set(SOURCES
    src/main.cpp
    src/ImageProcessor.cpp
    src/CudaKernels.cu
)

# Create executable
add_executable(ImageFilterGPU-Okab ${SOURCES})

# Link libraries
target_link_libraries(ImageFilterGPU-Okab ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# CUDA properties
set_property(TARGET ImageFilterGPU-Okab PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Build info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")