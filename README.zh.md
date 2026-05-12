# Intro-ops训练营

这个仓库是一个面向训练的 GPU 算子运行时。它体积很小，但工作流尽量贴近真实的算子库开发流程：

1. 在 `ops/<op>/` 下创建后端实现目录；如果算子可以复用 elementwise 框架，则放到 `ops/elementwise/<op>/`。
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
| `copy` | 可运行 | 安装 TileLang 后可运行 | 可运行 |
| `vector_add` | 可运行 | 安装 TileLang 后可运行 | 可运行 |
| `reduce_sum` | 可运行，row-wise fp32 | 安装 TileLang 后可运行 | 可运行，row-wise fp32 |
| `relu` | 可运行，elementwise fp16/fp32，支持 `negative_slope` | 未实现 | 未实现 |
| `softmax` | 可运行，row-wise fp32 | 安装 TileLang 后可运行 | 可运行，row-wise fp32 |

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

```bash
./scripts/build_metax.sh build
```

## 验证

```bash
python tests/run_ops.py --op copy --backend nvidia --mode all
CAMP_BUILD_DIR=build pytest tests/ -v --backend nvidia
pytest tests/ -v --backend tilelang
python tests/run_ops.py --op all --backend nvidia --mode bench
./scripts/build_metax.sh test
```

TileLang 后端需要安装 `tilelang` Python 包。MetaX 后端使用独立构建产物，构建时应关闭 NVIDIA 变体，并将后端源码放在 `ops/*/metax/*.maca` 或 `ops/elementwise/*/metax/*.maca`。

## Elementwise 框架

普通逐元素算子如果共享 shape、stride、broadcast 的执行模型，推荐放在 `ops/elementwise/<op>/<backend>/`。NVIDIA 公共 launcher 位于 `ops/common/elementwise/nvidia/elementwise_nvidia.cuh`；每个具体算子只需要提供公开 C API、descriptor 生命周期、dtype dispatch 和一个小的 device functor。`relu` 是教学示例：`negative_slope=0.0` 时等价于标准 ReLU，非 0 时等价于 leaky ReLU。

## 生产映射

| 训练概念 | 生产等价物 |
| --- | --- |
| `ops/<op>/nvidia/*.cu` 和 `ops/elementwise/<op>/nvidia/*.cu` 目录约定 | 构建系统自动发现 / 算子注册 |
| `include/operator_runtime/ops/<op>.h` 头文件 | 经过评审的算子 API 契约 |
| descriptor 生命周期 | create、workspace、execute、destroy |
| `tests/cases/<op>.py` | 正确性、布局和 API 契约覆盖 |
| `PerformanceResult` | 包含延迟、字节数、FLOPs、带宽的 profiler 报表行 |
| eager TileLang kernel | 使用 `T.empty(...)` 返回值的 puzzle-stage kernel |
| lazy TileLang `out_idx` template | TileOPs 风格的 kernel factory 和输出位置契约 |
