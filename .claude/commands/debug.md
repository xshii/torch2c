# 调试指南

当测试失败或编译结果不正确时，按以下方法定位问题。

## VSCode Tasks 速查

在 VSCode 中 `Cmd+Shift+P` → `Tasks: Run Task`，可用以下 task 快速调试：

| Task | 作用 | 什么时候用 |
|------|------|------------|
| `test:all` | 全量 pytest（448 用例） | 改完代码后回归验证 |
| `test:current` | pytest 当前打开的测试文件 | 开发时快速验证单个文件 |
| `test:module` | pytest 指定模块路径 | 缩小到某个模块排查 |
| `run:current` | 运行当前 Python 文件 | 跑 demo 脚本或任意 .py |
| `run:current+viz` | 运行 + 自动打开 pipeline.html | 编译后立即看可视化 |
| `compile:debug` | 编译 DemoEncoder（debug_dump 模式） | 全链路调试，产出 debug 快照 + 可视化 |
| `demo:e2e` | 完整编译链 demo | 快速验证端到端 |
| `demo:st` | ST1-ST6 端到端场景测试 | 验证 bulk/spill/tiling 策略 |
| `demo:st6` | ST6 MHA tiled 场景 | MHA 相关问题排查 |
| `demo:module` | 选择模块运行 demo（下拉菜单） | 单步查看某个 pass 的中间结果 |
| `demo:viz` | 生成内存 lifetime 可视化 | 排查内存分配问题 |
| `install:dev` | 安装开发依赖 | 环境初始化 |

## 日志系统

### Python 日志（4 层）

通过环境变量 `NPU_LOG_LEVEL` 控制级别（默认 INFO）：

```bash
# 看详细调试信息
NPU_LOG_LEVEL=DEBUG .venv/bin/pytest torch2c/xxx/tests/ -v -s

# 只看警告和错误
NPU_LOG_LEVEL=WARNING .venv/bin/pytest --tb=short -q
```

日志格式：`[时间] [级别] [模块名] 消息`

| 级别 | 内容 | 用途 |
|------|------|------|
| DEBUG | 每个 tensor 的 shape 变化、每步变换细节 | 精确定位某个 tensor 的问题 |
| INFO | pass 开始/完成、变换计数、编译阶段 | 正常监控编译流程 |
| WARNING | 非致命异常（跳过的 tensor、默认值回退） | 发现潜在问题 |
| ERROR | 校验失败、编译错误 | 定位错误根因 |

### opt_log — 优化决策日志

每个 pass 在节点上记录为什么做了某个变换，存储在 `node.params["_opt_log"]` 中：

```python
from torch2c.common.opt_log import log_opt

# 记录优化决策
log_opt(node, "format_planner", "格式变更", "nd → nz: cube src1 需要 Fractal_NZ")
```

查看方式：
- **可视化**：pipeline.html 中悬浮节点 → tooltip 显示 opt_log
- **JSON 快照**：`output/debug/debug/*.json` 中每个节点的 `_opt_log` 字段
- **代码中**：`node.params.get("_opt_log", [])`

### Pass 耗时统计

每个 pass 的执行耗时自动记录在 `configs["_pass_timing"]` 中：

```python
# pipeline.py 自动收集，示例输出：
{
    "graph_capture": {"enabled": True, "duration_ms": 12.3},
    "op_mapping": {"enabled": True, "duration_ms": 0.8},
    "block_pad": {"enabled": True, "duration_ms": 1.2},
    ...
}
```

### debug_dump 模式 — 全链路快照

```python
compile(model, dummy_input,
        config_dir=str(INTEGRATION_CONFIG_DIR),
        output_dir="output/debug",
        debug_dump=True)
```

产出目录结构：

```
output/debug/
├── debug/
│   ├── 00_graph_capture_after.json    # 每个 pass 后的 Graph IR 快照
│   ├── 01_op_mapping_after.json
│   ├── 02_op_decomposition_after.json
│   ├── ...
│   └── 08_codegen_after.json
├── viz/
│   ├── pipeline.html                  # 流水线可视化（可点击展开甬道图）
│   ├── schedule.html                  # 调度可视化
│   └── lifetime.html                  # 内存生命周期可视化
├── src/                               # 生成的 C 工程
│   ├── model_graph.c
│   ├── model_graph.h
│   ├── model_memory.h
│   └── model_weights.h
├── golden/                            # Python 计算的 golden 数据
└── npu_cpu_mock/                      # C mock 源码（自包含）
```

### C Mock 调试日志

C 工程编译时可设置 `NPU_DEBUG_LEVEL` 控制输出：

```bash
# 编译 C 工程（level=2 打印所有算子的输入/输出值）
cc -std=c99 -Wall -O2 -o npu_model_run main.c src/model_graph.c \
   utils/*.c npu_cpu_mock/src/*.c \
   -I. -Isrc -Inpu_cpu_mock/include \
   -DNPU_DEBUG_LEVEL=2 -lm

./npu_model_run
```

| Level | 输出内容 |
|-------|----------|
| 0 | 无调试输出（默认） |
| 1 | 每个算子的 BEGIN/END 标记 |
| 2 | 算子输入/输出 tensor 的前几个元素值 |

### debug.yaml 维测配置

可选的 `torch2c/integration/config/debug.yaml`：

```yaml
torch_trace:
  enabled: false       # 开启 PyTorch 算子 trace
  leaf_only: false     # 只 trace 叶子算子
memory_layout:
  enabled: false       # 打印内存布局详情
c_mock_trace:
  compile_level: 0     # C 编译时的 NPU_DEBUG_LEVEL
  runtime_level: 0     # 运行时 debug 级别
```

## 快速分类

| 症状 | 可能原因 | 排查方向 |
|------|----------|----------|
| `MappingError` | mapping 表缺算子 | `direct_mappings.yaml` |
| `DecompositionError` | 裂解规则缺失 | `decompositions.yaml` |
| `ValidationError` | format/dtype 不匹配 | `format_annotator` + `c_api_signatures.yaml` |
| `CodegenError` | 签名参数不匹配 | `c_api_signatures.yaml` + `c_emitter.py` |
| `MemoryPlanError` | L1 溢出 | `hardware_config.yaml` memory.l1 |
| C 编译失败 | mock 实现有 bug | `npu_cpu_mock/src/` |
| Golden FAIL (cosine < 0.95) | FP16 精度 / mock 逻辑 | 逐 pass 对比中间结果 |
| Config 一致性失败 | 某个表漏了算子 | 按报错补全对应 YAML |

## 逐 pass 缩小范围

### 方式 1: VSCode demo:module

`Cmd+Shift+P` → `Tasks: Run Task` → `demo:module` → 选择模块

可选模块：graph_capture / op_mapping / op_decomposition / op_absorption / format_annotator / validator / memory_planner / scheduler / codegen

### 方式 2: 命令行

```bash
# 逐模块 demo
python torch2c/a_capture/graph_capture/demo/run_demo.py
python torch2c/b_lowering/op_mapping/demo/run_demo.py
python torch2c/b_lowering/op_decomposition/demo/run_demo.py
python torch2c/c_backend/format_annotator/demo/run_demo.py
python torch2c/c_backend/validator/demo/run_demo.py
python torch2c/d_emission/memory_planner/demo/run_demo.py
python torch2c/d_emission/scheduler/demo/run_demo.py
python torch2c/d_emission/codegen/demo/run_demo.py
```

### 方式 3: 只跑某个模块测试

```bash
.venv/bin/pytest torch2c/{module}/tests/ -v -s
```

## 常见坑

### tensor 字段访问

```python
# 正确
t = graph.tensors[tensor_id]
node = graph.nodes[node_id]
```

### C mock 注意事项

```c
// 正确：.ptr
_Float16 *data = (_Float16 *)tensor.ptr;

// 错误：.addr 不存在
// _Float16 *data = (_Float16 *)tensor.addr;

// 正确：辅助函数
void* p = npu_t_ptr(tensor);
float val = npu_read_compute(ptr, index, src_dtype, compute_dtype);
npu_write_store(ptr, index, val, compute_dtype, dst_dtype);
```

### calc_padded_size 使用

```python
from torch2c.common.sizing import calc_padded_size, get_dim_align

# 正确：用 get_dim_align
size = calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))

# 错误：硬编码
# size = calc_padded_size(t.shape, t.dtype, t.format, (16, 16))
```

## 测试技巧

```bash
# 跑单个测试方法
.venv/bin/pytest torch2c/xxx/tests/test_xxx.py::TestClass::test_method -v -s

# 跑到第一个失败就停
.venv/bin/pytest -x --tb=long

# 显示 print 输出
.venv/bin/pytest -s

# 跳过 C golden 比对（慢）
.venv/bin/pytest --deselect torch2c/integration/tests/test_pipeline.py::TestCGoldenComparison

# 看详细日志
NPU_LOG_LEVEL=DEBUG .venv/bin/pytest torch2c/xxx/tests/ -v -s
```
