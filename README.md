# Operator Runtime Training Camp

This repository is a training-oriented GPU operator runtime. It is intentionally
small, but its workflow mirrors production operator libraries:

1. Create a custom backend implementation under `ops/<op>/`, or place the
   operator under `ops/elementwise/<op>/` when it should reuse the elementwise framework.
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
| `relu` | runnable, elementwise fp16/fp32 with `negative_slope` | not implemented | not implemented |
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

The TileLang backend requires the `tilelang` Python package. The MetaX backend is built as a separate variant, should use `CAMP_ENABLE_NVIDIA=OFF`, and stores backend sources under `ops/*/metax/*.maca` or `ops/elementwise/*/metax/*.maca`. The C ABI exposes a unified backend-selection interface; each build artifact accepts only the backend compiled into it, and returns `not supported` for unavailable backends.

## Elementwise Framework

The training camp supports two operator paths:

- Custom operators live under `ops/<op>/<backend>/`. This path is useful for teaching hand-written kernels, fixed contiguous fast paths, custom workspace needs, or non-elementwise structures. `copy` and `vector_add` demonstrate this path.
- Reusable elementwise operators live under `ops/elementwise/<op>/<backend>/`. This path is useful for ordinary elementwise operators that share the shape/stride/broadcast execution model. Later exercises can ask students to reimplement `copy` / `add`-style operators with this path.

The shared NVIDIA launcher is in `ops/common/elementwise/nvidia/elementwise_nvidia.cuh`; shared descriptor helpers live in `include/operator_runtime/detail/elementwise.h`. Each elementwise operator usually only needs to provide its public C API, small dtype dispatch, and a device functor. On the Python side, use `ElementwiseOpSpec` to describe input count, scalar parameters, and broadcast semantics. `relu` is the teaching example: `negative_slope=0.0` behaves like standard ReLU, while non-zero values behave like leaky ReLU.

## Production Mapping

| Training concept | Production equivalent |
| --- | --- |
| directory convention `ops/<op>/nvidia/*.cu` and `ops/elementwise/<op>/nvidia/*.cu` | build system auto-discovery / operator registry |
| C header `include/operator_runtime/ops/<op>.h` | reviewed operator API contract |
| descriptor lifecycle | create, workspace, execute, destroy |
| `tests/cases/<op>.py` | correctness, layout, and API contract coverage |
| `PerformanceResult` | profiler report row with latency, bytes, flops, bandwidth |
| eager TileLang kernel | puzzle-stage kernel using `T.empty(...)` return values |
| lazy TileLang `out_idx` template | TileOPs-style kernel factory and output-position contract |
