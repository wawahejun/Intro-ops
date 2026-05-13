# Intro-ops 训练营

面向 kernel 开发的 nano 级 GPU 算子运行时。框架已处理好 descriptor 生命周期、Python FFI、测试和 benchmark——学生只需在 `ops/<op>/nvidia/kernel.cuh` 中实现 kernel。

## 快速开始

```bash
# 激活环境
conda activate py312

# 构建（首次运行自动拉取 CUTLASS）
bash scripts/build_nvidia.sh build

# 运行全部测试
bash scripts/build_nvidia.sh test

# 清理构建
bash scripts/build_nvidia.sh clean
```

## 常用命令

### 构建脚本模式

```bash
bash scripts/build_nvidia.sh env          # 显示当前构建环境变量
bash scripts/build_nvidia.sh configure    # 仅 cmake 配置
bash scripts/build_nvidia.sh build        # 配置 + 编译
bash scripts/build_nvidia.sh test         # 运行 pytest + run_ops + examples
bash scripts/build_nvidia.sh all          # 编译 + 测试
bash scripts/build_nvidia.sh clean        # 删除构建目录
```

### 单算子测试

```bash
# 单算子正确性测试
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_copy.py -v

# run_ops 支持 --mode: test, bench, all
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode test
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode bench
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode all

# 全部算子
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op all --backend nvidia --mode all
```

### 强制重新构建

```bash
# 清除 cmake 缓存并重新构建
CAMP_FORCE_RECONFIGURE=1 bash scripts/build_nvidia.sh build

# 指定 GPU 架构
CMAKE_CUDA_ARCHITECTURES=89 bash scripts/build_nvidia.sh build
```

## 项目结构

```
ops/<op>/nvidia/
  kernel.cuh          <-- 在这里实现你的 kernel
  <op>_cuda.cu        <-- descriptor 生命周期（已提供）
include/operator_runtime/
  ops/<op>.h          <-- 公开 C API（已提供）
python/operator_runtime/
  ops/<op>.py         <-- Python 绑定（已提供）
tests/
  op_tests/test_<op>.py   <-- 正确性测试
  cases/<op>.py           <-- 测试用例
  bench/<op>.py           <-- 性能基准
```

## 算子列表

| 算子 | 难度 | 核心概念 |
| --- | --- | --- |
| `copy` | 入门 | grid-stride loop、向量化内存访问 |
| `vector_add` | 入门 | 逐元素计算 |
| `reduce_sum` | 中等 | shared memory、warp 归约 |
| `softmax` | 进阶 | 多趟归约 + 归一化 |
| `relu` | 参考实现 | elementwise 框架（已实现） |

## 开发流程

1. 阅读 `ops/<op>/nvidia/kernel.cuh` 中的 kernel 骨架
2. 实现 TODO kernel
3. 构建：`bash scripts/build_nvidia.sh build`
4. 测试：`PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_<op>.py -v`
5. Benchmark：`PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op <op> --backend nvidia --mode bench`

## 新增算子

完整指南见 [docs/how-to-add-an-operator.md](docs/how-to-add-an-operator.md)。最小步骤：

1. 创建 `ops/<op>/nvidia/`，包含 kernel 和 cuda 源文件
2. 在 `python/operator_runtime/ops/<op>.py` 添加 Python 绑定
3. 在 `tests/` 下添加测试用例、正确性测试和 benchmark
4. 重新运行 `bash scripts/build_nvidia.sh configure`（CMake 重新发现新 `.cu` 文件）
5. 构建并验证

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | 目标 GPU 架构（如 L40 用 `89`） |
| `CAMP_FORCE_RECONFIGURE` | `0` | 设为 `1` 在配置前清除 CMake 缓存 |
| `CAMP_ENABLE_CUTE` | `AUTO` | CuTe/CUTLASS 支持：`AUTO`、`ON`、`OFF` |
| `BUILD_DIR` | `build-nvidia` | 构建输出目录 |
| `CAMP_BUILD_DIR` | — | 告诉 Python 在哪里找 `libcamp_ops.so` |
| `CAMP_CUTLASS_ROOT` | — | 覆盖 CUTLASS 路径（跳过自动拉取） |
