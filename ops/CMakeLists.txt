include("${CMAKE_BINARY_DIR}/generated/operators.cmake")

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

