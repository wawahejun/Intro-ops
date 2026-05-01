cmake_minimum_required(VERSION 3.22)

project(operator_runtime_training LANGUAGES CXX)

option(CAMP_ENABLE_NVIDIA "Build NVIDIA CUDA backend" ON)
option(CAMP_ENABLE_TILELANG "Enable TileLang Python backend metadata" ON)
option(CAMP_ENABLE_METAX "Build MetaX backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/GenerateOperators.cmake)

if(CAMP_ENABLE_NVIDIA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  include(cmake/cuda_helpers.cmake)
endif()

add_subdirectory(ops)
