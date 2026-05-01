# Operator Runtime Training Camp

This repository is a training-oriented GPU operator runtime. It is intentionally
small, but its workflow mirrors production operator libraries:

1. Define the operator contract in `ops/<op>/operator.yaml`.
2. Generate build and runtime registries from the manifest.
3. Implement a backend-specific descriptor lifecycle.
4. Expose a Python API with out-of-place, out-variant, and prepared execution.
5. Validate correctness against PyTorch and benchmark steady-state execution.

The project is rooted directly at `/workspace`; it does not create a nested
`/workspace/camp` project directory.

## Operators

| Operator | NVIDIA C++ | TileLang | MetaX |
| --- | --- | --- | --- |
| `copy` | runnable | runnable when TileLang is installed | stub |
| `vector_add` | runnable | runnable when TileLang is installed | stub |
| `reduce_sum` | runnable, row-wise fp32 | runnable when TileLang is installed | stub |
| `softmax` | runnable, row-wise fp32 | runnable when TileLang is installed | stub |

## Build

```bash
mkdir -p build
cd build
cmake .. -DCAMP_ENABLE_NVIDIA=ON -DCAMP_ENABLE_METAX=OFF
cmake --build . -j$(nproc)
```

## Validate

```bash
python tools/validate_operator_manifest.py --ops-root ops --tests-root tests
CAMP_BUILD_DIR=build pytest tests/ -v --backend nvidia
pytest tests/ -v --backend tilelang
python tests/bench_all.py --backend nvidia --profile tests/perf_profiles/local_gpu.yaml
```

The TileLang backend requires the `tilelang` Python package. 

## Production Mapping

| Training concept | Production equivalent |
| --- | --- |
| `operator.yaml` | reviewed operator spec / manifest |
| descriptor lifecycle | create, workspace, execute, destroy |
| generated registry | operation table / backend registry |
| `tests/cases/<op>.py` | correctness, layout, and API contract coverage |
| `PerformanceResult` | profiler report row with latency, bytes, flops, bandwidth |
