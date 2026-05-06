#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-metax}"
MACA_PATH="${MACA_PATH:-/opt/maca}"
CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
CUDA_PATH="${CUDA_PATH:-${CUCC_PATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="${1:-build}"

export MACA_PATH
export CUCC_PATH
export CUDA_PATH
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${LD_LIBRARY_PATH:-}"

configure() {
  mkdir -p "${BUILD_DIR}"
  "${PYTHON_BIN}" -m cmake -G Ninja "${ROOT}" \
    -B "${BUILD_DIR}" \
    -DCAMP_ENABLE_NVIDIA=OFF \
    -DCAMP_ENABLE_METAX=ON
}

build() {
  "${PYTHON_BIN}" -m cmake --build "${BUILD_DIR}" -- -v
}

test_pytest() {
  env CAMP_BUILD_DIR="${BUILD_DIR}" "${PYTHON_BIN}" -m pytest "${ROOT}/tests" -v --backend metax
}

test_run_ops() {
  env CAMP_BUILD_DIR="${BUILD_DIR}" "${PYTHON_BIN}" "${ROOT}/tests/run_ops.py" --op all --backend metax --mode all
}

test_examples() {
  env CAMP_BUILD_DIR="${BUILD_DIR}" "${PYTHON_BIN}" "${ROOT}/examples/01_copy.py" --backend metax
  env CAMP_BUILD_DIR="${BUILD_DIR}" "${PYTHON_BIN}" "${ROOT}/examples/02_vector_add.py" --backend metax
  env CAMP_BUILD_DIR="${BUILD_DIR}" "${PYTHON_BIN}" "${ROOT}/examples/03_reduce_sum.py" --backend metax
  env CAMP_BUILD_DIR="${BUILD_DIR}" "${PYTHON_BIN}" "${ROOT}/examples/04_softmax.py" --backend metax
}

case "${MODE}" in
  configure)
    configure
    ;;
  build)
    configure
    build
    ;;
  test)
    test_pytest
    test_run_ops
    test_examples
    ;;
  all)
    configure
    build
    test_pytest
    test_run_ops
    test_examples
    ;;
  clean)
    rm -rf "${BUILD_DIR}"
    ;;
  *)
    echo "usage: $0 {configure|build|test|all|clean}" >&2
    exit 2
    ;;
esac
