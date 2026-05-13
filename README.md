# Operator Runtime Training Camp

A nano GPU operator runtime for learning kernel development. The framework handles descriptor lifecycle, Python FFI, tests, and benchmarks — students focus on writing kernels in `ops/<op>/nvidia/kernel.cuh`.

## Quick Start

```bash
# activate environment
conda activate py312

# build (auto-fetches CUTLASS on first run)
bash scripts/build_nvidia.sh build

# run all tests
bash scripts/build_nvidia.sh test

# clean build
bash scripts/build_nvidia.sh clean
```

## Common Commands

### Build Script Modes

```bash
bash scripts/build_nvidia.sh env          # show current build environment variables
bash scripts/build_nvidia.sh configure    # cmake configure only
bash scripts/build_nvidia.sh build        # configure + build
bash scripts/build_nvidia.sh test         # run pytest + run_ops + examples
bash scripts/build_nvidia.sh all          # build + test
bash scripts/build_nvidia.sh clean        # remove build directory
```

### Single Operator Testing

```bash
# correctness test for one operator
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_copy.py -v

# run_ops supports --mode: test, bench, all
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode test
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode bench
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode all

# all operators
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op all --backend nvidia --mode all
```

### Force Rebuild

```bash
# clear cmake cache and rebuild
CAMP_FORCE_RECONFIGURE=1 bash scripts/build_nvidia.sh build

# target specific GPU architecture
CMAKE_CUDA_ARCHITECTURES=89 bash scripts/build_nvidia.sh build
```

## Project Structure

```
ops/<op>/nvidia/
  kernel.cuh          <-- implement your kernel here
  <op>_cuda.cu        <-- descriptor lifecycle (provided)
include/operator_runtime/
  ops/<op>.h          <-- public C API (provided)
python/operator_runtime/
  ops/<op>.py         <-- Python bindings (provided)
tests/
  op_tests/test_<op>.py   <-- correctness tests
  cases/<op>.py           <-- test cases
  bench/<op>.py           <-- benchmarks
```

## Operators

| Operator | Difficulty | Key Concept |
| --- | --- | --- |
| `copy` | easy | grid-stride loop, vectorized memory access |
| `vector_add` | easy | elementwise computation |
| `reduce_sum` | medium | shared memory, warp reduction |
| `softmax` | hard | multi-pass reduction + normalization |
| `relu` | reference | elementwise framework (already implemented) |

## Workflow

1. Read the kernel scaffold in `ops/<op>/nvidia/kernel.cuh`
2. Implement the TODO kernel
3. Build: `bash scripts/build_nvidia.sh build`
4. Test: `PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_<op>.py -v`
5. Benchmark: `PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op <op> --backend nvidia --mode bench`

## Adding a New Operator

See [docs/how-to-add-an-operator.md](docs/how-to-add-an-operator.md) for the full guide. Minimal steps:

1. Create `ops/<op>/nvidia/` with kernel and cuda source
2. Add Python bindings in `python/operator_runtime/ops/<op>.py`
3. Add test cases, correctness tests, and benchmarks under `tests/`
4. Re-run `bash scripts/build_nvidia.sh configure` (CMake re-discovers new `.cu` files)
5. Build and verify

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Target GPU arch (e.g. `89` for L40) |
| `CAMP_FORCE_RECONFIGURE` | `0` | Set `1` to clear CMake cache before configure |
| `CAMP_ENABLE_CUTE` | `AUTO` | CuTe/CUTLASS support: `AUTO`, `ON`, `OFF` |
| `BUILD_DIR` | `build-nvidia` | Build output directory |
| `CAMP_BUILD_DIR` | — | Tell Python where to find `libcamp_ops.so` |
| `CAMP_CUTLASS_ROOT` | — | Override CUTLASS path (skips auto-fetch) |
