# Quickstart

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 运行 Demo

### E2E 编译 Demo（推荐）

从 PyTorch 模型到 C 工程的完整编译 + golden 验证：

```bash
# 各模块单步 demo（graph_capture → codegen 逐步查看中间结果）
.venv/bin/python torch2c/graph_capture/demo/run_demo.py
.venv/bin/python torch2c/op_mapping/demo/run_demo.py
.venv/bin/python torch2c/op_decomposition/demo/run_demo.py
.venv/bin/python torch2c/op_absorption/demo/run_demo.py
.venv/bin/python torch2c/format_annotator/demo/run_demo.py
.venv/bin/python torch2c/validator/demo/run_demo.py
.venv/bin/python torch2c/memory_planner/demo/run_demo.py
.venv/bin/python torch2c/scheduler/demo/run_demo.py
.venv/bin/python torch2c/codegen/demo/run_demo.py
```

### 可视化 Demo

内存 lifetime 图 + pipeline schedule 甘特图：

```bash
.venv/bin/python torch2c/memory_planner/demo/viz_lifetime.py
.venv/bin/python torch2c/memory_planner/demo/viz_graph.py
```

生成的 HTML 在各 demo 的 `demo_output/viz/` 目录下，用浏览器打开即可。

### ST 场景测试

6 个系统级端到端测试场景（ST1-ST6），覆盖 bulk / spill / tiling 策略：

```bash
.venv/bin/pytest torch2c/integration/tests/demo_st/test_st_scenarios.py -v
```

场景定义见 `test_scenarios.yaml`。

## 运行测试

```bash
# 全部测试
.venv/bin/pytest --tb=short -q

# 单模块测试
.venv/bin/pytest torch2c/memory_planner/tests/ -v

# ST 测试
.venv/bin/pytest torch2c/integration/tests/ -v
```

## VSCode 快捷方式

在 VSCode 中按 `Cmd+Shift+P` → `Tasks: Run Task`：

| Task | 说明 |
|------|------|
| `test:all` | 运行全部测试 |
| `test:module` | 运行指定模块测试 |
| `demo:e2e` | 运行 codegen E2E demo（完整编译链） |
| `demo:st` | 运行 ST 场景测试（6 个场景） |
| `demo:viz` | 生成可视化（lifetime + schedule） |

## 项目结构

```
torch2c/
├── common/              # 公共类型、配置、错误
├── graph_capture/       # ① PyTorch → Graph
├── op_mapping/          # ② ATen → NPU 算子映射
├── op_decomposition/    # ③ 复合算子裂解
├── op_absorption/       # ④ 算子融合（bias absorption）
├── format_annotator/    # ⑤ NZ/ND 格式标注
├── validator/           # ⑥ 图验证
├── memory_planner/      # ⑦⑧ L1/HBM 内存分配 + DMA 计划
├── scheduler/           # ⑨ 执行调度
├── codegen/             # ⑩ C 代码生成
├── integration/         # 编译管线 + E2E 测试
│   ├── demo/            #   Encoder 模型 demo
│   └── tests/demo_st/   #   ST 场景测试
├── viz/                 # 可视化（ECharts HTML）
└── main.py              # 入口（预留）
```
