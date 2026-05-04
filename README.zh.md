# Intro-ops训练营

这个仓库是一个面向训练的 GPU 算子运行时。它体积很小，但工作流尽量贴近真实的算子库开发流程：

1. 在 `ops/<op>/` 下创建后端实现目录。
2. 构建系统通过目录约定自动发现源码。
3. 实现后端专属的 descriptor 生命周期（create、workspace、execute、destroy）。
4. 提供 Python API，支持 out-of-place、out-variant 和 prepared 执行。
5. 通过 PyTorch 做正确性验证，并做稳定态性能测试。

## Python 目录结构

```text
python/
  operator_runtime/
    backend.py
    ops/
    _internal/
  operator_runtime_testing/
```

- `operator_runtime.ops` 放公开算子绑定。
- `operator_runtime._internal` 放私有 FFI 和运行时细节。
- `operator_runtime_testing` 放断言、benchmark 等仅测试使用的工具。

## 算子

| 算子 | NVIDIA C++ | TileLang | MetaX |
| --- | --- | --- | --- |
| `copy` | 可运行 | 安装 TileLang 后可运行 | stub |
| `vector_add` | 可运行 | 安装 TileLang 后可运行 | stub |
| `reduce_sum` | 可运行，row-wise fp32 | 安装 TileLang 后可运行 | stub |
| `softmax` | 可运行，row-wise fp32 | 安装 TileLang 后可运行 | stub |

## 安装

```bash
pip install -r requirements.txt
```

## 构建

```bash
mkdir -p build
cd build
cmake .. -DCAMP_ENABLE_NVIDIA=ON -DCAMP_ENABLE_METAX=OFF
cmake --build . -j$(nproc)
```

## 验证

```bash
python tests/run_ops.py --op copy --backend nvidia --mode all
CAMP_BUILD_DIR=build pytest tests/ -v --backend nvidia
pytest tests/ -v --backend tilelang
python tests/run_ops.py --op all --backend nvidia --mode bench
```

TileLang 后端需要安装 `tilelang` Python 包。

## 如何新增算子

1. 创建 `ops/<name>/nvidia/<name>_cuda.h`，提供 4 个 C API：create、workspace、execute、destroy。
2. 创建 `ops/<name>/nvidia/<name>_cuda.cu`，实现 CUDA 逻辑。
3. 创建 `python/operator_runtime/ops/<name>.py`，使用 `operator_runtime._internal.bind_*` 绑定。
4. 创建 `tests/cases/<name>.py`，提供 `correctness_cases()`、`api_error_cases()` 和 `benchmark_cases()`。
5. 创建 `tests/ops/test_<name>.py` 和 `tests/bench/<name>.py`。
6. 在构建目录里重新执行 `cmake ..`，因为 glob 会自动拾取新的 `.cu` 文件。
7. 在 `python/operator_runtime/__init__.py` 中导出公共 API。

这里没有 YAML、没有代码生成、也没有额外的注册步骤。

## 生产映射

| 训练概念 | 生产等价物 |
| --- | --- |
| `ops/<op>/nvidia/*.cu` 目录约定 | 构建系统自动发现 / 算子注册 |
| `ops/<op>/nvidia/<op>_cuda.h` 头文件 | 经过评审的算子 API 契约 |
| descriptor 生命周期 | create、workspace、execute、destroy |
| `tests/cases/<op>.py` | 正确性、布局和 API 契约覆盖 |
| `PerformanceResult` | 包含延迟、字节数、FLOPs、带宽的 profiler 报表行 |
| eager TileLang kernel | 使用 `T.empty(...)` 返回值的 puzzle-stage kernel |
| lazy TileLang `out_idx` template | TileOPs 风格的 kernel factory 和输出位置契约 |
