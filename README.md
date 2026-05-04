# Operator Runtime Training Camp

This repository is a training-oriented GPU operator runtime. It is intentionally
small, but its workflow mirrors production operator libraries:

1. Create a directory under `ops/<op>/` with backend implementations.
2. The build system auto-discovers sources by directory convention.
3. Implement a backend-specific descriptor lifecycle (create, workspace, execute, destroy).
4. Expose a Python API with out-of-place, out-variant, and prepared execution.
5. Validate correctness against PyTorch and benchmark steady-state execution.


## Operators

| Operator | NVIDIA C++ | TileLang | MetaX |
| --- | --- | --- | --- |
| `copy` | runnable | runnable when TileLang is installed | stub |
| `vector_add` | runnable | runnable when TileLang is installed | stub |
| `reduce_sum` | runnable, row-wise fp32 | runnable when TileLang is installed | stub |
| `softmax` | runnable, row-wise fp32 | runnable when TileLang is installed | stub |

## Setup

```bash
pip install -r requirements.txt
```

## Build

```bash
mkdir -p build
cd build
cmake .. -DCAMP_ENABLE_NVIDIA=ON -DCAMP_ENABLE_METAX=OFF
cmake --build . -j$(nproc)
```

## Validate

```bash
python tests/run_ops.py --op copy --backend nvidia --mode all
CAMP_BUILD_DIR=build pytest tests/ -v --backend nvidia
pytest tests/ -v --backend tilelang
python tests/run_ops.py --op all --backend nvidia --mode bench
```

The TileLang backend requires the `tilelang` Python package.

## Adding a New Operator

1. Create `ops/<name>/nvidia/<name>_cuda.h` with the C API (4 functions: create, workspace, execute, destroy).
2. Create `ops/<name>/nvidia/<name>_cuda.cu` with the CUDA implementation.
3. Create `python/operator_runtime/ops/<name>.py` using `ctypes_bindings.bind_*` functions.
4. Create `tests/cases/<name>.py` with `correctness_cases()`, `api_error_cases()`, and `benchmark_cases()`.
5. Create `tests/ops/test_<name>.py` and `tests/bench/<name>.py`.
6. Re-run `cmake ..` in the build directory (the glob will pick up the new `.cu` file).
7. Register the public API in `python/operator_runtime/__init__.py`.

No YAML, no code generation, no registration step.

## Production Mapping

| Training concept | Production equivalent |
| --- | --- |
| directory convention `ops/<op>/nvidia/*.cu` | build system auto-discovery / operator registry |
| C header `ops/<op>/nvidia/<op>_cuda.h` | reviewed operator API contract |
| descriptor lifecycle | create, workspace, execute, destroy |
| `tests/cases/<op>.py` | correctness, layout, and API contract coverage |
| `PerformanceResult` | profiler report row with latency, bytes, flops, bandwidth |
| eager TileLang kernel | puzzle-stage kernel using `T.empty(...)` return values |
| lazy TileLang `out_idx` template | TileOPs-style kernel factory and output-position contract |
