# Quickstart

## 安装

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> **VSCode**: Task `install:dev`

## 编译 Demo 模型

### 最简模型（Y = AX + B）

```bash
python scripts/demo_axb.py
# 输出: output/demo_axb/
```

### Embedding 模型

```bash
python scripts/demo_embedding.py
# 输出: output/demo_embedding/
```

### 单层 Attention（2 头 + LayerNorm）

```bash
python scripts/demo_single_attn.py
# 输出: output/single_attn/
```

### 2 层 Encoder Transformer（完整 demo）

```bash
python scripts/compile_and_viz.py
# 输出: output/compile_viz/  (含 debug 快照 + pipeline 可视化)
```

> **VSCode**: Task `compile:debug` 或打开任意脚本 → Task `run:current`

## 可视化

### 本地浏览器

```bash
open output/compile_viz/viz/pipeline.html
```

> **VSCode**: 打开编译脚本 → Task `run:current+viz`（编译完自动打开 HTML）

### 远端访问（Flask）

```bash
python scripts/viz_server.py --compile --port 8080
# 浏览器: http://<ip>:8080/
```

### pipeline.html 操作

| 操作 | 效果 |
|------|------|
| 点击 pass 节点 | 展开 4 列甬道图 (Cube/Vector/IDMA/DMA) + pass 说明 |
| ← → 方向键 | 切换 pass |
| ↑ ↓ 方向键 | 切换图中节点，高亮上下游 |
| 悬浮节点 | tooltip: tensor ID/shape + 优化原因 |
| 点击图中节点 | 高亮直接依赖（蓝=上游，橙=下游，流动动画） |
| 再次点击 | 取消高亮 |
| Esc | 关闭甬道图 |

## 各模块单步 Demo

按编译流水线顺序逐步查看中间结果：

```bash
python torch2c/a_capture/graph_capture/demo/run_demo.py
python torch2c/b_lowering/op_mapping/demo/run_demo.py
python torch2c/b_lowering/op_decomposition/demo/run_demo.py
python torch2c/optpass/bc_op_absorption/demo/run_demo.py
python torch2c/c_backend/format_annotator/demo/run_demo.py
python torch2c/c_backend/validator/demo/run_demo.py
python torch2c/d_emission/memory_planner/demo/run_demo.py
python torch2c/d_emission/scheduler/demo/run_demo.py
python torch2c/d_emission/codegen/demo/run_demo.py
```

> **VSCode**: Task `demo:module`（下拉选择模块）

## 测试

| 范围 | 命令 | VSCode Task |
|------|------|-------------|
| 全量 UT（441 用例） | `pytest` | `test:all` |
| 当前打开的文件 | `pytest <file> -v` | `test:current` |
| 指定模块 | `pytest <path> -v` | `test:module` |
| 端到端 ST | `pytest torch2c/integration/tests/demo_st/ -v` | `demo:st` |
| ST6 MHA tiling | — | `demo:st6` |

## 验证 C 工程

```bash
cd output/demo_axb/
cmake -B build && cmake --build build
cd build && ctest -V
```

## VSCode Tasks 速查

`Cmd+Shift+P` → `Tasks: Run Task`：

| Task | 说明 |
|------|------|
| `install:dev` | 安装开发依赖 |
| `test:all` | 全量测试 |
| `test:current` | 测试当前文件 |
| `test:module` | 测试指定模块 |
| `run:current` | 运行当前 Python 文件 |
| `run:current+viz` | 运行 + 打开 pipeline.html |
| `compile:debug` | 编译 DemoEncoder + 可视化 |
| `demo:e2e` | 完整编译链 demo |
| `demo:st` | ST 场景测试 |
| `demo:module` | 选择模块运行 demo |
| `demo:viz` | 内存 lifetime 可视化 |

## 项目结构

```
torch2c/
├── a_capture/graph_capture/     ① Frontend: PyTorch → Graph IR
├── b_lowering/                  Op Lowering
│   ├── op_mapping/              ② ATen → NPU 命名映射
│   └── op_decomposition/        ③ 裂解 + 广播 + is_mapped
├── c_backend/                   Target Annotation
│   ├── format_annotator/        ⑤ format/dtype 标注
│   ├── reformat_inserter/       ⑤b format 转换
│   └── validator/               ⑥ 合法性校验
├── d_emission/                  Scheduling + Codegen
│   ├── scheduler/               ⑦ 拓扑排序
│   ├── memory_planner/          ⑧ 内存编排 + DMA
│   └── codegen/                 ⑨ C 代码生成
├── optpass/                     可选优化 Pass（bc_/c_/cd_/d_ 前缀标记插入位置）
├── common/                      Graph IR / 日志 / 配置 / opt_log
├── integration/                 管线 + 配置 + 测试
├── viz/                         可视化（pipeline + 甬道图）
├── scripts/                     编译 + 可视化脚本
└── npu_cpu_mock/                NPU C API 的 CPU 模拟
```
