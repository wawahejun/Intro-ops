# Operator Runtime Training Camp

This repository is a training-oriented GPU operator runtime. It is intentionally
small, but its workflow mirrors production operator libraries:

1. Create a directory under `ops/<op>/` with backend implementations.
2. The build system auto-discovers sources by directory convention.
3. Implement a backend-specific descriptor lifecycle (create, workspace, execute, destroy).
4. Expose a Python API with out-of-place, out-variant, and prepared execution.
5. Validate correctness against PyTorch and benchmark steady-state execution.

## Python Layout

```text
python/
  operator_runtime/
    backend.py
    ops/
    _internal/
  operator_runtime_testing/
```

- `operator_runtime.ops` contains public operator bindings.
- `operator_runtime._internal` contains private FFI/runtime plumbing.
- `operator_runtime_testing` contains test-only helpers such as assertions and benchmark utilities.

## Operators

| Operator | NVIDIA C++ | TileLang | MetaX |
| --- | --- | --- | --- |
| `copy` | runnable | runnable when TileLang is installed | runnable |
| `vector_add` | runnable | runnable when TileLang is installed | runnable |
| `reduce_sum` | runnable, row-wise fp32 | runnable when TileLang is installed | runnable, row-wise fp32 |
| `softmax` | runnable, row-wise fp32 | runnable when TileLang is installed | runnable, row-wise fp32 |

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

```bash
./scripts/build_metax.sh build
```

## Validate

```bash
python tests/run_ops.py --op copy --backend nvidia --mode all
CAMP_BUILD_DIR=build pytest tests/ -v --backend nvidia
pytest tests/ -v --backend tilelang
python tests/run_ops.py --op all --backend nvidia --mode bench
./scripts/build_metax.sh test
```

The TileLang backend requires the `tilelang` Python package. The MetaX backend is built as a separate variant, should use `CAMP_ENABLE_NVIDIA=OFF`, and stores backend sources under `ops/*/metax/*.maca`.

## Production Mapping

| Training concept | Production equivalent |
| --- | --- |
| directory convention `ops/<op>/nvidia/*.cu` | build system auto-discovery / operator registry |
| C header `include/operator_runtime/ops/<op>.h` | reviewed operator API contract |
| descriptor lifecycle | create, workspace, execute, destroy |
| `tests/cases/<op>.py` | correctness, layout, and API contract coverage |
| `PerformanceResult` | profiler report row with latency, bytes, flops, bandwidth |
| eager TileLang kernel | puzzle-stage kernel using `T.empty(...)` return values |
| lazy TileLang `out_idx` template | TileOPs-style kernel factory and output-position contract |
