# Intro-ops训练营

这个仓库是一个面向训练的 nano 级 GPU 算子运行时。它体积很小，但工作流尽量贴近真实的算子库开发流程：

1. 在两条并行路径中选择一条：在 `ops/<op>/` 下创建自定义后端实现，或在 `ops/elementwise/<op>/` 下复用 elementwise 框架。
2. 构建系统通过目录约定自动发现源码。
3. 实现后端专属的 descriptor 生命周期（create、workspace、execute、destroy）。
4. 提供 Python API，支持 out-of-place、out-variant 和 prepared 执行。
5. 通过 PyTorch 做正确性验证，并做稳定态性能测试。

这个仓库当前是框架和练习骨架：descriptor 生命周期、Python API、测试、benchmark 入口和示例目录已经接好；部分第一阶段 kernel 逻辑刻意保留 TODO，留给学生补全。

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
| `copy` | 第一阶段 TODO kernel 骨架 | 安装 TileLang 后可用的第一阶段 TODO kernel 骨架 | 后端骨架已接好 |
| `vector_add` | 第一阶段 TODO kernel 骨架 | 安装 TileLang 后可用的第一阶段 TODO kernel 骨架 | 后端骨架已接好 |
| `reduce_sum` | 第一阶段 TODO row-wise fp32 kernel 骨架 | 安装 TileLang 后可用的第一阶段 TODO kernel 骨架 | row-wise fp32 后端骨架已接好 |
| `relu` | 可运行的 elementwise fp16/fp32 示例，支持 `negative_slope` | 未实现 | 可运行的 elementwise fp16/fp32 示例 |
| `softmax` | 第一阶段 TODO row-wise fp32 kernel 骨架 | 安装 TileLang 后可用的第一阶段 TODO kernel 骨架 | row-wise fp32 后端骨架已接好 |

对第一阶段骨架算子来说，编译通过只说明框架链路已接好；在补全 TODO kernel 之前，完整 correctness 测试预期不会全部通过。

## 安装

```bash
pip install -r requirements.txt
```

## 构建

```bash
./scripts/build_nvidia.sh build
```

```bash
./scripts/build_metax.sh build
```

`build_nvidia.sh` 会优先使用仓库内的 `third_party/cutlass`；如果不存在，则回退到 `CAMP_CUTLASS_ROOT` 或 `CUTLASS_ROOT`。
默认构建目录分别是 `build-nvidia` 和 `build-metax`。需要覆盖时传 `BUILD_DIR=...`；需要在重配前清理已有 CMake cache 时，传 `CAMP_FORCE_RECONFIGURE=1`。

```bash
BUILD_DIR=build ./scripts/build_nvidia.sh build
CAMP_FORCE_RECONFIGURE=1 ./scripts/build_nvidia.sh configure
cmake --preset nvidia-release
cmake --build --preset nvidia-release
```

默认构建会为本机环境生成一个 backend 版本的 `libcamp_ops.so`。NVIDIA 和 MetaX 刻意作为独立变体构建；C/Python API 保持统一的 backend 选择接口，如果请求当前库未编译进来的 backend，会返回 `not supported`。

自定义 NVIDIA 算子既可以写普通 CUDA C++ kernel，也可以写 CuTe/CUTLASS 风格 C++ 实现。CuTe/CUTLASS 不是单独 backend，而是 NVIDIA backend 内部的可选实现依赖。`CAMP_ENABLE_NVIDIA=ON` 时，构建会从 `CAMP_CUTLASS_ROOT`、`CAMP_CUTE_INCLUDE_DIRS`、`third_party/cutlass` 或 `CUTLASS_ROOT` / `CUTLASS_HOME` / `CUTLASS_PATH` 环境变量自动探测 `cute/tensor.hpp`。如果找到，会给 `camp_ops` 定义 `CAMP_ENABLE_CUTE=1`；如果没找到，普通 CUDA C++ 算子仍然正常构建。只有在希望找不到 CuTe 时直接配置失败，才显式传 `CAMP_ENABLE_CUTE=ON`：

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

## 验证

```bash
python tests/run_ops.py --op copy --backend nvidia --mode all
CAMP_BUILD_DIR=build-nvidia pytest tests/ -v --backend nvidia
pytest tests/ -v --backend tilelang
python tests/run_ops.py --op all --backend nvidia --mode bench
./scripts/build_metax.sh test
```

TileLang 后端需要安装 `tilelang` Python 包。MetaX 后端使用独立构建产物，构建时应关闭 NVIDIA 变体，并将后端源码放在 `ops/*/metax/*.maca` 或 `ops/elementwise/*/metax/*.maca`。

## 算子开发路径

训练营支持两条路径：

- 自定义算子路径：放在 `ops/<op>/<backend>/`，适合教学 kernel 编写、固定 contiguous fast path、特殊 layout、特殊 workspace、CuTe/CUTLASS 风格实现或非逐元素结构。`copy` 和 `vector_add` 是这条路径的演示。这是一条正式路径，不是所有算子迁移到 elementwise 前的临时阶段。
- elementwise 复用路径：放在 `ops/elementwise/<op>/<backend>/`，适合共享 shape、stride、broadcast 执行模型的普通 unary、binary、多输入逐元素算子。后续作业可以让学生用这条路径重新实现 `copy` / `add` 类算子作为对照训练。

NVIDIA 公共 launcher 位于 `ops/common/elementwise/nvidia/elementwise_nvidia.cuh`；公共 descriptor helper 位于 `include/operator_runtime/detail/elementwise.h`。每个 elementwise 算子通常只需要提供公开 C API、少量 dtype dispatch 和一个 device functor。Python 侧优先用 `ElementwiseOpSpec` 描述输入数、标量参数和 broadcast 语义。`relu` 是 elementwise 教学示例：`negative_slope=0.0` 时等价于标准 ReLU，非 0 时等价于 leaky ReLU。

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
