cmake_minimum_required(VERSION 3.22)

project(operator_runtime_training LANGUAGES CXX)

option(CAMP_ENABLE_NVIDIA "Build NVIDIA CUDA backend" ON)
option(CAMP_ENABLE_TILELANG "Enable TileLang Python backend metadata" ON)
option(CAMP_ENABLE_METAX "Build MetaX backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(CAMP_ENABLE_NVIDIA AND CAMP_ENABLE_METAX)
  message(FATAL_ERROR "CAMP_ENABLE_NVIDIA and CAMP_ENABLE_METAX cannot both be ON; build separate variants")
endif()

if(CAMP_ENABLE_METAX)
  find_package(MetaX REQUIRED)
  set(ENV{CUCC_PATH} "${MetaX_CUBRIDGE_ROOT}")
  set(ENV{CUDA_PATH} "${MetaX_CUBRIDGE_ROOT}")
  list(APPEND CMAKE_MODULE_PATH "${MetaX_CMAKE_MODULE_DIR}")
  set(CMAKE_MACA_SOURCE_FILE_EXTENSIONS maca;cu;cc;cpp;c)
endif()

if(CAMP_ENABLE_NVIDIA OR CAMP_ENABLE_METAX)
  if(CAMP_ENABLE_METAX)
    enable_language(MACA)
  else()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    include(cmake/cuda_helpers.cmake)
  endif()
endif()

if(CAMP_ENABLE_METAX)
  set(CAMP_METAX_COMPILE_OPTIONS
    -D__MACA_NO_HALF_OPERATORS__
    -D__FAST_BLOCK_RED__
  )
endif()

add_subdirectory(ops)
