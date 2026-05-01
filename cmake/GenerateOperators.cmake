set(CAMP_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${CAMP_GENERATED_DIR}")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CAMP_GENERATED_OPERATORS_CMAKE "${CAMP_GENERATED_DIR}/operators.cmake")
set(CAMP_GENERATED_REGISTRY_PY "${CAMP_GENERATED_DIR}/operator_registry.py")
set(CAMP_GENERATED_TEST_MANIFEST "${CAMP_GENERATED_DIR}/operator_test_manifest.json")

execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_operator_artifacts.py"
          --ops-root "${CMAKE_CURRENT_SOURCE_DIR}/ops"
          --out-dir "${CAMP_GENERATED_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RESULT_VARIABLE CAMP_GENERATOR_RESULT
)

if(NOT CAMP_GENERATOR_RESULT EQUAL 0)
  message(FATAL_ERROR "operator artifact generation failed")
endif()

