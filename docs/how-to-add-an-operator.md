# 如何开发一个新算子

## 目标

在当前项目里，新增一个算子的最小闭环包括四部分。训练营同时保留两条并行路径：自定义算子路径用于教学 kernel 编写、特殊 layout、特殊 workspace 和非普通逐元素结构，elementwise 复用路径用于普通 unary / binary / broadcast / stride 逐元素算子。

1. 在 `ops/<算子名>/` 下补自定义后端实现，或在 `ops/elementwise/<算子名>/` 下补复用 elementwise 框架的实现。
2. 在 `python/operator_runtime/ops/` 下补 Python API。
3. 在 `tests/` 下补正确性测试和 benchmark 入口。
4. 重新构建并验证。

当前仓库是训练营框架和练习骨架。`copy`、`vector_add`、`reduce_sum`、`softmax` 的第一阶段 kernel 文件仍保留 TODO，用来让学生补齐计算逻辑；`relu` 是已经接通的 elementwise 示例。


## Step 1：明确算子接口

开始前先确认：

1. 这个算子有几个输入、几个输出。
2. 输出 shape 是否和输入一致。
3. 是否要求输入输出 dtype 一致。
4. 是否只支持 contiguous tensor。
5. 是否需要额外参数，例如 `dim`、`scalar`。
6. 是否需要 workspace。

这一步的目的，是确定后面 C API 和 Python 绑定该按哪种模式实现。不要把“逐元素”简单等同于“必须用 elementwise 框架”：训练营会先用 `copy`、`vector_add` 展示自定义算子写法，也会保留 elementwise 复用路径作为普通逐元素算子的工程化选择。两条路径是并行选项，不是“先自定义、最后都迁移到 elementwise”的过渡关系。

## Step 2：创建算子目录

先选择算子目录：

- 自定义算子：放在 `ops/<算子名>/`。适合教学底层 kernel、固定 contiguous fast path、特殊 layout、特殊 workspace、reduce/softmax 等非普通 elementwise 算子，也允许写 CuTe/CUTLASS 风格 C++ 实现。
- elementwise 复用算子：放在 `ops/elementwise/<算子名>/`。适合共享 shape、stride、broadcast 执行模型的 unary / binary / 多输入逐元素算子。

`copy` 和 `vector_add` 是自定义算子教学示例。`relu` 是 elementwise 框架示例，目录为 `ops/elementwise/relu/nvidia/`，公共 NVIDIA launcher 位于 `ops/common/elementwise/nvidia/elementwise_nvidia.cuh`。

先在选定目录下建立对应后端目录。

当前建议至少补齐：

- `ops/<算子名>/nvidia/` 或 `ops/elementwise/<算子名>/nvidia/`
- `python/operator_runtime/ops/<算子名>.py`
- `tests/cases/<算子名>.py`
- `tests/op_tests/test_<算子名>.py`
- `tests/bench/<算子名>.py`

如果后续要支持 TileLang 或 MetaX，再分别补 `tilelang/` 或 `metax/`。
但对当前流程来说，NVIDIA 版本是最小必需项。

## Step 2.5：选择实现风格

自定义 NVIDIA 算子可以采用两种 C++ 实现风格：

1. 原生 CUDA C++：直接在 `.cu` / `.cuh` 里写 `__global__` kernel。这是第一阶段默认训练方式，`copy`、`vector_add`、`reduce_sum`、`softmax` 都按这个方式组织。
2. CuTe/CUTLASS 风格 C++：仍然放在 `ops/<算子名>/nvidia/`，仍然导出同一组 descriptor 生命周期符号，但内部可以使用 CuTe tensor、layout、copy atom、TiledMMA 等抽象。

CuTe/CUTLASS 不是单独 backend，而是 NVIDIA backend 内部的可选实现依赖。`CAMP_ENABLE_NVIDIA=ON` 时，构建会默认自动探测 `cute/tensor.hpp`，来源包括 `CAMP_CUTLASS_ROOT`、`CAMP_CUTE_INCLUDE_DIRS`、`third_party/cutlass` 或 `CUTLASS_ROOT` / `CUTLASS_HOME` / `CUTLASS_PATH` 环境变量。找到后目标会定义 `CAMP_ENABLE_CUTE=1` 并把 include 目录加到 `camp_ops`；找不到时继续构建普通 CUDA C++ 算子。

推荐约定是把 CUTLASS 源码 clone 到仓库根目录下的 `third_party/cutlass`，这样 `CAMP_ENABLE_NVIDIA=ON` 时就能自动探测，不需要每次手工传路径。

如果你希望“找不到 CuTe 就直接失败”，可以显式打开强制模式：

```bash
cmake .. \
  -DCAMP_ENABLE_NVIDIA=ON \
  -DCAMP_ENABLE_CUTE=ON \
  -DCAMP_CUTLASS_ROOT=/path/to/cutlass
```

也可以用 `"-DCAMP_CUTE_INCLUDE_DIRS=/path/a;/path/b"` 显式指定 include 目录。这个选项只提供框架扩展点，不要求第一阶段学生必须使用 CuTe。

## Step 3：实现 NVIDIA 后端

`ops/<算子名>/nvidia/` 或 `ops/elementwise/<算子名>/nvidia/` 这一层负责 C++/CUDA 实现。

自定义算子通常需要这几部分：

1. kernel 文件
2. C API 头文件
3. C API 实现文件

这里最关键的是保持现有命名约定一致，因为 Python 侧会按固定符号名去找函数。

如果是 elementwise 复用算子，不需要每个算子重复写 shape/stride kernel。推荐复用 `include/operator_runtime/detail/elementwise.h` 里的 descriptor helper 和 `ops/common/elementwise/nvidia/elementwise_nvidia.cuh`：

1. descriptor 继承 `oprt::ElementwiseDescriptorBase`。
2. create 阶段调用 `oprt::init_elementwise_descriptor(desc, out, {inputs...})`。
3. workspace 复用 `oprt::get_elementwise_workspace_size`。
4. execute 阶段只做 dtype dispatch，并调用 `oprt::elementwise::nvidia::launch<T, N>(...)`。
5. 算子自身只提供 device functor，例如 `relu` 的 `value > 0 ? value : value * negative_slope`。

elementwise 路径推荐覆盖：

1. 输出 shape 由输入 broadcast 得出，或由调用方提供的 out 固定。
2. 每个输出元素可以独立计算。
3. 输入之间只需要普通 shape/stride/broadcast 索引。
4. 不需要跨元素同步、规约、排序、扫描、复杂临时 workspace 或特殊数据布局。

不满足这些条件时，优先走自定义算子路径。

一个 NVIDIA 算子需要完整提供四个生命周期接口：

1. create
2. workspace
3. execute
4. destroy

整体流程可以理解为：

1. `create`：检查输入是否合法，并保存运行所需信息。
2. `workspace`：返回执行需要的临时内存大小。
3. `execute`：按 dtype 和参数启动 kernel。
4. `destroy`：释放 descriptor。

## Step 4：补 Python 绑定

Python 入口放在 `python/operator_runtime/ops/<算子名>.py`。

当前项目对外通常暴露三类接口：

1. `prepare_<算子名>`
2. `<算子名>_`
3. `<算子名>`

职责分工一般是：

1. `prepare_<算子名>`：完成参数检查、绑定底层函数、创建 descriptor 和 workspace。
2. `<算子名>_`：接收调用方提供的输出 tensor，执行一次。
3. `<算子名>`：自动分配输出 tensor，再调用 `<算子名>_`。

如果你的算子是自定义算子，并且签名和现有 unary、binary、reduce-like 模式一致，就沿用现有 helper。
如果你的算子是 elementwise 复用算子，优先在 Python 里定义 `ElementwiseOpSpec`，再调用 `prepare_elementwise_op`。这个 spec 描述算子名、输入数、标量参数和 broadcast 语义，避免为每个 elementwise 算子重复写 FFI 绑定和基础校验。

示例：

```python
_MY_OP_SPEC = ElementwiseOpSpec(
    name="my_op",
    input_count=1,
    scalar_argtypes=(ctypes.c_float,),
)
```

## Step 5：导出到公共 API

新增 Python 文件后，还要把公开接口补到：

1. `python/operator_runtime/ops/__init__.py`
2. `python/operator_runtime/__init__.py`

至少需要把这三个名字暴露出去：

1. `<算子名>`
2. `<算子名>_`
3. `prepare_<算子名>`

这样测试和用户代码才能直接导入。

## Step 6：补测试用例数据

`tests/cases/<算子名>.py` 只负责组织测试数据。

当前仓库一般分成三类：

1. `correctness_cases()`
2. `api_error_cases()`
3. `benchmark_cases()`

建议在这里把不同 shape、dtype、容差、错误场景、性能场景分开整理，避免把 case 直接写死在测试函数里。

## Step 7：补正确性测试

`tests/op_tests/test_<算子名>.py` 主要负责三件事：

1. 正确性对比
2. API contract 检查
3. prepared 执行复用检查

正确性测试通常是拿 PyTorch 结果做对照。
API contract 测试主要覆盖 shape 不匹配、dtype 不匹配、非 contiguous 等场景。
如果底层走 descriptor lifecycle，建议补一个 prepared 多次执行的测试，确认 descriptor 可以复用。

## Step 8：补 benchmark

`tests/bench/<算子名>.py` 负责性能入口。

当前流程里，一般会：

1. 从 `benchmark_cases()` 取输入规模。
2. 构造 CUDA tensor。
3. 测量自定义算子耗时。
4. 测量对应 PyTorch 实现耗时。
5. 汇总成性能结果。

这一步的目标不是做复杂分析，而是保证新算子已经接入仓库现有 benchmark 流程。

## Step 9：重新配置和编译

因为 `ops/CMakeLists.txt` 是通过 glob 自动发现 `ops/*/nvidia/*.cu` 和 `ops/elementwise/*/nvidia/*.cu`，所以新增 `.cu` 之后要重新配置。

顺序是：

1. 重新执行 `./scripts/build_nvidia.sh configure`，或手工进入 `build-nvidia/` 后重新执行 `cmake ..`。
2. 再执行 `./scripts/build_nvidia.sh build`，或手工执行编译。

如果你只改了 Python，不新增 `.cu`，通常不需要重新配置。
但只要新加了 NVIDIA 源文件，就必须重新跑一次 CMake。

当前构建策略是统一 API、单 backend 产物：NVIDIA 和 MetaX 不要求编进同一个 `libcamp_ops.so`。每个环境按本机硬件构建一个 backend 版本；Python / C ABI 保持统一 backend 参数；当前库只接受已编译进来的 backend，未启用 backend 返回 `not supported`。

## Step 10：验证

建议按下面顺序验证：

1. 先跑单算子的 pytest。
2. 再跑更大范围的算子测试。
3. 最后跑 benchmark 或统一入口。

优先确认：

1. 输出结果和 PyTorch 一致。
2. 错误输入能稳定报错。
3. descriptor/workspace 流程正常。
4. 没有破坏已有算子。

## 推荐开发顺序

如果你要新增的是一个普通算子，建议按这个顺序推进：

1. 先参考最接近的现有算子选模板。
2. 先打通 NVIDIA 后端最小可运行版本。
3. 再补 Python API。
4. 再补 cases、pytest、benchmark。
5. 最后补其他后端或做性能优化。

## 现有模板怎么选

- `copy`：适合最简单的自定义单输入单输出流程，当前第一阶段 kernel 逻辑保留 TODO。
- `vector_add`：适合标准 contiguous 双输入自定义算子，当前第一阶段 kernel 逻辑保留 TODO。
- `relu`：适合复用 elementwise 框架的单输入逐元素算子，也展示了额外标量参数 `negative_slope` 的 C/Python 绑定方式。
- `reduce_sum`：适合带 reduce 维度的算子，当前第一阶段 kernel 逻辑保留 TODO。
- `softmax`：适合带更明确 shape 约束和归一化逻辑的算子，当前第一阶段 kernel 逻辑保留 TODO。

如果新算子本质上是普通 elementwise，优先参考 `relu`、`ElementwiseOpSpec` 和 `ops/common/elementwise/nvidia/elementwise_nvidia.cuh`；如果课程目标是练习手写 contiguous kernel，再参考 `copy` 或 `vector_add` 的自定义路径。

## 最终检查清单

提交前至少确认以下内容都已完成：

1. `ops/<算子名>/nvidia/` 或 `ops/elementwise/<算子名>/nvidia/` 已补齐实现。
2. `python/operator_runtime/ops/<算子名>.py` 已补齐。
3. 两个 `__init__.py` 已导出新接口。
4. `tests/cases/<算子名>.py` 已补数据。
5. `tests/op_tests/test_<算子名>.py` 已补测试。
6. `tests/bench/<算子名>.py` 已补 benchmark。
7. 新增 `.cu` 后已经重新执行过 `./scripts/build_nvidia.sh configure` 或等价的 `cmake ..`。
8. 至少完成一次单算子验证。
