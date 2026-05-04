# Convention: operator CUDA sources auto-discovered by directory layout:
#   ops/<operator>/nvidia/*.cu  ->  compiled when CAMP_ENABLE_NVIDIA is ON
# After adding a new operator, re-run cmake to pick up new files.

set(CAMP_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/common/status.cc
)

set(CAMP_NVIDIA_SOURCES "")
if(CAMP_ENABLE_NVIDIA)
  file(GLOB_RECURSE CAMP_NVIDIA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*/nvidia/*.cu
  )
endif()

set(CAMP_OPERATOR_SOURCES ${CAMP_COMMON_SOURCES} ${CAMP_NVIDIA_SOURCES})

add_library(camp_ops SHARED ${CAMP_OPERATOR_SOURCES})
target_include_directories(camp_ops PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_compile_definitions(camp_ops PRIVATE
  $<$<BOOL:${CAMP_ENABLE_NVIDIA}>:CAMP_ENABLE_NVIDIA=1>
  $<$<BOOL:${CAMP_ENABLE_METAX}>:CAMP_ENABLE_METAX=1>
)

if(CAMP_ENABLE_NVIDIA)
  set_target_properties(camp_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
