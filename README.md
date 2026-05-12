# Operator Runtime Training Camp

This repository is a training-oriented nano GPU operator runtime. It is
intentionally small, but its workflow mirrors production operator libraries:

1. Choose one of two parallel operator paths: write a custom backend
   implementation under `ops/<op>/`, or reuse the elementwise framework under
   `ops/elementwise/<op>/`.
2. The build system auto-discovers sources by directory convention.
3. Implement a backend-specific descriptor lifecycle (create, workspace, execute, destroy).
4. Expose a Python API with out-of-place, out-variant, and prepared execution.
5. Validate correctness against PyTorch and benchmark steady-state execution.

The repository is a framework and exercise scaffold. The descriptor lifecycle,
Python API, tests, benchmark entry points, and example directories are wired up;
some phase-1 kernels are intentionally left as TODOs for students to complete.

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
| `copy` | phase-1 TODO kernel scaffold | phase-1 TODO kernel scaffold when TileLang is installed | scaffolded backend |
| `vector_add` | phase-1 TODO kernel scaffold | phase-1 TODO kernel scaffold when TileLang is installed | scaffolded backend |
| `reduce_sum` | phase-1 TODO row-wise fp32 kernel scaffold | phase-1 TODO kernel scaffold when TileLang is installed | scaffolded row-wise fp32 backend |
| `relu` | runnable elementwise fp16/fp32 example with `negative_slope` | not implemented | runnable elementwise fp16/fp32 example |
| `softmax` | phase-1 TODO row-wise fp32 kernel scaffold | phase-1 TODO kernel scaffold when TileLang is installed | scaffolded row-wise fp32 backend |

For scaffolded phase-1 operators, a successful build only proves that the
framework wiring is present. Full correctness tests are expected to fail until
the TODO kernels are implemented.

## Setup

```bash
pip install -r requirements.txt
```

## Build

```bash
./scripts/build_nvidia.sh build
```

```bash
./scripts/build_metax.sh build
```

`build_nvidia.sh` prefers `third_party/cutlass` when present, and otherwise
falls back to `CAMP_CUTLASS_ROOT` or `CUTLASS_ROOT`.
The default build directories are `build-nvidia` and `build-metax`. Use
`BUILD_DIR=...` to override them, and `CAMP_FORCE_RECONFIGURE=1` when you want a
script to clear an existing CMake cache before reconfiguring.

```bash
BUILD_DIR=build ./scripts/build_nvidia.sh build
CAMP_FORCE_RECONFIGURE=1 ./scripts/build_nvidia.sh configure
cmake --preset nvidia-release
cmake --build --preset nvidia-release
```

The default build produces one backend variant of `libcamp_ops.so` for the local
environment. NVIDIA and MetaX are intentionally built as separate variants; the
C/Python API keeps the same backend-selection interface, and a library returns
`not supported` when asked to use a backend that was not compiled into it.

Custom NVIDIA operators can be written as ordinary CUDA C++ kernels or as
CuTe/CUTLASS-style C++ implementations. CuTe/CUTLASS is not a separate backend;
it is an optional NVIDIA-backend implementation dependency. When
`CAMP_ENABLE_NVIDIA=ON`, the build auto-detects `cute/tensor.hpp` from
`CAMP_CUTLASS_ROOT`, `CAMP_CUTE_INCLUDE_DIRS`, `third_party/cutlass`, or the
`CUTLASS_ROOT` / `CUTLASS_HOME` / `CUTLASS_PATH` environment variables. If CuTe
is found, `CAMP_ENABLE_CUTE=1` is defined for `camp_ops`; if not found, ordinary
CUDA C++ operators still build. Set `CAMP_ENABLE_CUTE=ON` only when you want a
missing CuTe installation to be a configuration error:

```bash
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake .. -DCAMP_ENABLE_NVIDIA=ON
```

```bash
cmake .. \
  -DCAMP_ENABLE_NVIDIA=ON \
  -DCAMP_ENABLE_CUTE=ON \
  -DCAMP_CUTLASS_ROOT=/path/to/cutlass
```

## Validate

```bash
python tests/run_ops.py --op copy --backend nvidia --mode all
CAMP_BUILD_DIR=build-nvidia pytest tests/ -v --backend nvidia
pytest tests/ -v --backend tilelang
python tests/run_ops.py --op all --backend nvidia --mode bench
./scripts/build_metax.sh test
```

The TileLang backend requires the `tilelang` Python package. The MetaX backend is built as a separate variant, should use `CAMP_ENABLE_NVIDIA=OFF`, and stores backend sources under `ops/*/metax/*.maca` or `ops/elementwise/*/metax/*.maca`.

## Operator Paths

The training camp supports two operator paths:

- Custom operators live under `ops/<op>/<backend>/`. This path is useful for teaching kernel writing, fixed contiguous fast paths, special layout or workspace needs, CuTe/CUTLASS-style implementations, and non-elementwise structures. `copy` and `vector_add` demonstrate this path. It is a first-class path, not a temporary step before moving everything into the elementwise framework.
- Reusable elementwise operators live under `ops/elementwise/<op>/<backend>/`. This path is useful for ordinary unary, binary, multi-input, broadcast, or stride-aware elementwise operators that share the same execution model. Later exercises can ask students to reimplement `copy` / `add`-style operators with this path as a comparison exercise.

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
