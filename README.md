# torch2c — PyTorch to NPU C Code Offline Compiler

NPU离线编译栈：将PyTorch模型编译为可在自研NPU上运行的完整C工程。

## 项目概述

本项目是一个**纯离线编译器/代码生成器**，Python端不与C侧有任何运行时交互。

**编译流水线：**

```
PyTorch 模型
    │
    ▼
① graph_capture    : torch.export → Graph IR
    ▼
② op_mapping       : ATen op → NPU op（1对1直接映射）
    ▼
③ op_decomposition : ATen op → NPU ops（1对N裂解）
    ▼
④ op_absorption    : 独立算子吸收为相邻算子的参数
    ▼
⑤ format_annotator : 标注每个tensor的format/dtype/compute_dtype
    ▼
⑥ validator        : 校验所有算子在C接口中有对应
    ▼
⑦ memory_planner   : HBM全局规划 + L1局部排列 + DMA计划
    ▼
⑧ scheduler        : 计算单元分配 + 依赖关系生成
    ▼
⑨ codegen          : 生成完整C工程
    ▼
输出：完整C工程目录
```

**最终产物**是一整套可编译的C源代码和数据文件。

## 一阶段 MVP 目标

给定一个固定shape的**2层Encoder Transformer**（含attention mask，小shape），完成端到端编译：
- 图捕获 → 算子映射/裂解/吸收 → 内存编排 → 生成完整C工程
- C工程在NPU工具链上编译通过
- 执行后与PyTorch golden数据精度比对通过

**Demo模型参数：** batch=1, seq_len=32, hidden_size=64, num_heads=4, ffn_dim=256, mask_shape=[1,1,32,32]

### 一阶段不做

| 特性 | 说明 |
|------|------|
| 动态shape | 所有shape固定 |
| 训练 | 仅推理 |
| 算子融合 | 所有算子独立执行 |
| 多核调度 | 单核 |
| L1 tiling | 假设L1够大 |
| Double buffer | DMA和计算不重叠 |
| Auto-tuning | 不搜索最优配置 |

## 硬件架构

目标NPU架构类似华为昇腾达芬奇体系（多计算单元 + 片上SRAM + HBM）。

**计算单元：**

| 计算单元 | 职责 | 典型算子 |
|----------|------|----------|
| Cube | 矩阵乘加 | matmul, linear |
| Vector | 向量运算、激活、归一化 | relu, gelu, layernorm子算子 |
| Scalar | 流程控制、地址计算 | reshape（地址重算） |
| DMA | 数据搬运 | load, store, 随路format转换 |

**存储层次：**

```
HBM / DDR（片外，大容量，高延迟）
    ↓ DMA（随路format/dtype转换）
L2 Buffer（片上共享）
    ↓ DMA
L1 Buffer（核内，一阶段假设足够大）
    ↓
L0A / L0B / L0C（计算单元缓冲，C API内部管理）
Unified Buffer（Vector工作空间）
```

## 项目结构

```
torch2c/
├── npu_compiler/                Python包
│   ├── common/                  基础设施（Graph IR、日志、配置、异常）
│   │   ├── graph_ir.py
│   │   ├── logger.py
│   │   ├── config_loader.py
│   │   ├── errors.py
│   │   └── tests/
│   │
│   ├── graph_capture/           Pass①：torch.export → Graph IR
│   │   ├── graph_capture.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── op_mapping/              Pass②：ATen op → NPU op（1对1映射）
│   │   ├── op_mapping.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── op_decomposition/        Pass③：ATen op → NPU ops（1对N裂解）
│   │   ├── op_decomposition.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── op_absorption/           Pass④：独立算子吸收为相邻算子参数
│   │   ├── op_absorption.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── format_annotator/        Pass⑤：标注format/dtype/compute_dtype
│   │   ├── format_annotator.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── validator/               Pass⑥：合法性校验
│   │   ├── validator.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── memory_planner/          Pass⑦：内存编排
│   │   ├── memory_planner.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── scheduler/               Pass⑧：调度与依赖生成
│   │   ├── scheduler.py
│   │   ├── config/  demo/  tests/
│   │
│   ├── codegen/                 Pass⑨：C代码生成
│   │   ├── c_emitter.py  weight_exporter.py  golden_exporter.py
│   │   ├── utils_emitter.py  mock_emitter.py  cmake_emitter.py
│   │   ├── templates/  config/  demo/  tests/
│   │
│   ├── integration/             管线串联与端到端测试
│   │   ├── pipeline.py
│   │   ├── config/  demo/  tests/
│   │
│   └── main.py                  入口
│
├── docs/
│   └── ordr.md                  需求文档
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 模块依赖关系

```
common（所有模块的唯一依赖）
  │
  ├── graph_capture     ─┐
  ├── op_mapping        ─┤
  ├── op_decomposition  ─┤
  ├── op_absorption     ─┼── 全部只依赖common，互相无依赖，可完全并行开发
  ├── format_annotator  ─┤
  ├── validator         ─┤
  ├── memory_planner    ─┤
  ├── scheduler         ─┤
  └── codegen           ─┘
          │
          ▼
      integration（串联所有模块，最后开发）
```

## 快速开始

### 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.4+（需支持 torch.export） |
| PyYAML | 6.0+ |
| NumPy | 1.24+ |

### 创建虚拟环境

```bash
# 创建 venv
python3.10 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# 验证 Python 版本
python --version  # 应输出 Python 3.10.x
```

### 安装依赖

```bash
# 激活虚拟环境后
pip install -e ".[dev]"
```

或手动安装：

```bash
pip install -r requirements.txt
pip install pytest pytest-cov  # 开发依赖
```

### 运行测试

```bash
# 全量UT
pytest

# 单模块UT
pytest npu_compiler/common/tests/
pytest npu_compiler/op_mapping/tests/
```

### 运行端到端Demo

```bash
python -m npu_compiler.main
```

或分步运行：

```bash
# 1. 图捕获
python -m npu_compiler.graph_capture.demo.run_demo

# 2. 端到端流水线
python -m npu_compiler.integration.demo.run_full_demo
```

### 验证生成的C工程

```bash
# Mock模式编译验证
cd output/
cmake .
make
ctest

# 语法检查
gcc -fsyntax-only -include npu_mock.h src/model_graph.c
```

## 关键设计决策

| 决策项 | 结论 |
|--------|------|
| torch.export分解 | 禁止自动分解，保留layer_norm/softmax高级op |
| ATen算子命名 | 使用全称（如 `aten.mm.default`） |
| Format冲突 | DMA随路转换，不插入显式转换节点 |
| tensor.format语义 | HBM存储格式，DMA load时按消费者需求转换 |
| 精度标注 | 输入format/dtype、计算dtype、输出format/dtype 均可不同 |
| Scalar值 | 常量tensor（is_weight=True, shape=[1]） |
| 多输出算子 | 全部保留，无消费者的输出由memory_planner回收 |
| Reshape | 正常DMA搬运，不做零拷贝 |
| npu_transpose | 4D接口（ndim + dims参数） |
| Attention mask | 模型外部输入（is_model_input=True, shape=[1,1,32,32]） |
| 配置管理 | integration/config与模块局部config完全相同 |

## 代码原则

- **简化代码**：每个模块核心代码不超过300行，每个函数不超过50行
- **高复用**：所有Pass共用同一套Graph IR数据结构
- **高可靠**：统一日志系统（Python logging），统一异常定义
- **模块自包含**：每个模块独立文件夹，包含代码、配置、UT和局部demo

## 精度验证标准

| 指标 | 阈值 |
|------|------|
| 最大绝对误差 | < 1e-3（FP16） |
| 余弦相似度 | > 0.999 |
| 不匹配元素比例 | < 0.1% |

## 多Agent并行开发

| Agent | 负责模块 | 前置 |
|-------|---------|------|
| Agent 0 | common | 无 |
| Agent 1 | graph_capture + op_mapping + op_decomposition | common |
| Agent 2 | op_absorption + format_annotator + validator | common |
| Agent 3 | memory_planner + scheduler | common |
| Agent 4 | codegen | common |
| Agent 0 | integration（串联+端到端测试） | 全部模块 |

详见 [docs/ordr.md](docs/ordr.md) 获取完整需求文档（含第16节补充决策记录）。
