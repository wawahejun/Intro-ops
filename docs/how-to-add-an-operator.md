# 如何开发一个新算子

## 目标

在当前项目里，新增一个可运行算子的最小闭环包括四部分：

1. 在 `ops/<算子名>/` 下补后端实现。
2. 在 `python/operator_runtime/ops/` 下补 Python API。
3. 在 `tests/` 下补正确性测试和 benchmark 入口。
4. 重新构建并验证。


## Step 1：明确算子接口

开始前先确认：

1. 这个算子有几个输入、几个输出。
2. 输出 shape 是否和输入一致。
3. 是否要求输入输出 dtype 一致。
4. 是否只支持 contiguous tensor。
5. 是否需要额外参数，例如 `dim`、`scalar`。
6. 是否需要 workspace。

这一步的目的，是确定后面 C API 和 Python 绑定该按哪种模式实现。

## Step 2：创建算子目录

先在 `ops/<算子名>/` 下建立对应目录。

当前建议至少补齐：

- `ops/<算子名>/nvidia/`
- `python/operator_runtime/ops/<算子名>.py`
- `tests/cases/<算子名>.py`
- `tests/ops/test_<算子名>.py`
- `tests/bench/<算子名>.py`

如果后续要支持 TileLang 或 MetaX，再分别补 `tilelang/` 或 `metax/`。
但对当前流程来说，NVIDIA 版本是最小必需项。

## Step 3：实现 NVIDIA 后端

`ops/<算子名>/nvidia/` 这一层负责 C++/CUDA 实现。

通常需要这几部分：

1. kernel 文件
2. C API 头文件
3. C API 实现文件

这里最关键的是保持现有命名约定一致，因为 Python 侧会按固定符号名去找函数。

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

如果你的算子签名和现有 unary、binary、reduce-like 模式一致，就沿用现有 helper。
如果签名比较特殊，例如带额外标量参数，就单独写一层绑定。

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

`tests/ops/test_<算子名>.py` 主要负责三件事：

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

因为 `ops/CMakeLists.txt` 是通过 glob 自动发现 `ops/*/nvidia/*.cu`，所以新增 `.cu` 之后要重新配置。

顺序是：

1. 进入 `build/` 目录。
2. 重新执行 `cmake ..`。
3. 再执行编译。

如果你只改了 Python，不新增 `.cu`，通常不需要重新配置。
但只要新加了 NVIDIA 源文件，就必须重新跑一次 CMake。

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

- `copy`：适合最简单的单输入单输出流程。
- `vector_add`：适合标准 elementwise 双输入算子。
- `reduce_sum`：适合带 reduce 维度的算子。
- `softmax`：适合带更明确 shape 约束和归一化逻辑的算子。

如果新算子本质上是普通 elementwise，优先参考 `vector_add`。

## 最终检查清单

提交前至少确认以下内容都已完成：

1. `ops/<算子名>/nvidia/` 已补齐实现。
2. `python/operator_runtime/ops/<算子名>.py` 已补齐。
3. 两个 `__init__.py` 已导出新接口。
4. `tests/cases/<算子名>.py` 已补数据。
5. `tests/ops/test_<算子名>.py` 已补测试。
6. `tests/bench/<算子名>.py` 已补 benchmark。
7. 新增 `.cu` 后已经重新执行过 `cmake ..`。
8. 至少完成一次单算子验证。
