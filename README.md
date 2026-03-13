# torch2c — PyTorch to NPU C Code Offline Compiler

NPU离线编译栈：将PyTorch模型编译为可在自研NPU上运行的完整C工程。

## 项目概述

本项目是一个**纯离线编译器/代码生成器**，Python端不与C侧有任何运行时交互。

**编译流水线：**

```
PyTorch 模型
    │
    ▼
① graph_capture      : torch.export → Graph IR
    ▼
② op_mapping         : ATen op → NPU op（1对1直接映射）
    ▼
③ op_decomposition   : ATen op → NPU ops（1对N裂解）
    ▼
④ op_absorption      : 独立算子吸收为相邻算子的参数
    ▼
⑤ format_annotator   : 标注每个tensor的format/dtype/compute_dtype
    ▼
⑤a format_planner    : 基于硬件能力的图级最优format分配
    ▼
⑤b reformat_inserter : 插入format转换节点（format不匹配时）
    ▼
⑤c storage_assigner  : 分配tensor存储类型（hbm/local/pipe）
    ▼
⑥ validator          : 校验所有算子在C接口中有对应
    ▼
⑦ memory_planner     : HBM全局规划 + L1局部排列 + DMA计划
    ▼
⑧ scheduler          : 拓扑排序 + 依赖关系生成
    ▼
⑨ codegen            : 生成完整C工程
    ▼
输出：完整C工程目录
```

**最终产物**是一整套可编译的C源代码和数据文件。

## 一阶段 MVP 目标

给定一个固定shape的**2层Encoder Transformer**（含attention mask，小shape），完成端到端编译：
- 图捕获 → 算子映射/裂解/吸收 → 内存编排 → 生成完整C工程
- C工程在NPU工具链上编译通过
- 执行后与PyTorch golden数据精度比对通过

**Demo模型参数：** batch=1, seq_len=32, hidden_size=64, num_heads=3, ffn_dim=256, mask_shape=[1,1,32,32]

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
├── torch2c/                     Python包
│   ├── common/                  基础设施（Graph IR、日志、配置、异常）
│   ├── graph_capture/           Pass①：torch.export → Graph IR
│   ├── op_mapping/              Pass②：ATen op → NPU op（1对1映射）
│   ├── op_decomposition/        Pass③：ATen op → NPU ops（1对N裂解）
│   ├── op_absorption/           Pass④：独立算子吸收为相邻算子参数
│   ├── format_annotator/        Pass⑤：标注format/dtype/compute_dtype
│   ├── format_planner/          Pass⑤a：图级最优format分配
│   ├── reformat_inserter/       Pass⑤b：插入format转换节点
│   ├── storage_assigner/        Pass⑤c：分配tensor存储类型
│   ├── validator/               Pass⑥：合法性校验
│   ├── memory_planner/          Pass⑦：内存编排（HBM+L1+DMA）
│   ├── scheduler/               Pass⑧：拓扑排序与依赖生成
│   ├── codegen/                 Pass⑨：C代码生成
│   ├── viz/                     可视化工具（依赖图、生命周期图）
│   ├── integration/             管线串联与端到端测试
│   │   ├── pipeline.py          编译入口
│   │   ├── config/              全部YAML配置（single source of truth）
│   │   └── demo/                端到端demo + 系统测试
│   └── main.py                  入口
│
├── npu_cpu_mock/                NPU C API 的 CPU 模拟实现
│   ├── include/                 npu_api.h + npu_fp16.h
│   ├── src/                     全部算子实现（C99）
│   └── tests/                   C单元测试（CMake + ctest）
│
├── docs/
│   ├── ordr.md                  需求文档（权威来源）
│   ├── architecture.md          架构设计文档
│   ├── dev-guide.md             开发指南（扩展算子/定位问题/测试）
│   └── roadmap.md               开发路线图
│
├── pyproject.toml
├── requirements.txt
└── CHANGELOG.md
```

每个 Pass 模块内部结构统一：`<module>.py` + `config/` + `demo/` + `tests/` + `README.md`

## 模块依赖关系

```
common（所有模块的唯一依赖）
  │
  ├── graph_capture      ─┐
  ├── op_mapping         ─┤
  ├── op_decomposition   ─┤
  ├── op_absorption      ─┤
  ├── format_annotator   ─┼── 全部只依赖common，互相无依赖
  ├── format_planner     ─┤
  ├── reformat_inserter  ─┤
  ├── storage_assigner   ─┤
  ├── validator          ─┤
  ├── memory_planner     ─┤
  ├── scheduler          ─┤
  └── codegen            ─┘
          │
          ▼
      integration（串联所有模块）
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
pytest torch2c/common/tests/
pytest torch2c/op_mapping/tests/
```

### 编写模型并生成 C 代码

**1. 定义模型** — 使用 `@npu` 标注每个子模块的精度和格式：

```python
import torch
import torch.nn as nn
from torch2c.common import NpuSpec, npu

# 定义精度 spec
FP16    = NpuSpec("fp16", "nd")
FP16_NZ = NpuSpec("fp16", "nz")

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # npu() 标注：指定 input/output 的 dtype+format, 以及权重格式
        self.linear = npu(nn.Linear(64, 64),
                          input=FP16, output=FP16, weight=FP16_NZ,
                          compute_dtype="fp32")
        self.act = npu(nn.GELU(),
                       input=FP16, output=FP16, compute_dtype="fp16")
        self.norm = npu(nn.LayerNorm(64),
                        input=FP16, output=FP16, weight=FP16,
                        compute_dtype="fp32")

    def forward(self, x):
        return self.norm(self.act(self.linear(x)))
```

**2. 编译到 C 工程** — 一行调用 `compile()`：

```python
from torch2c.common import INTEGRATION_CONFIG_DIR
from torch2c.integration.pipeline import compile

model = MyModel().eval()
dummy_input = torch.randn(1, 32, 64)

output_dir = compile(
    model=model,
    dummy_input=dummy_input,
    config_dir=str(INTEGRATION_CONFIG_DIR),
    output_dir="output/MyModel",
)
# output_dir 下生成完整可编译的 C 工程
```

**3. 使用内置 demo 模型**（多层 Encoder Transformer）：

```python
from torch2c.integration.demo.encoder_model import EncoderModel

model = EncoderModel(
    d_model=192,     # 模型维度
    dim_ff=384,      # FFN 中间维度
    num_layers=2,    # Encoder 层数
    num_heads=3,     # 注意力头数
    precision="fp16" # "mixed" 或 "fp16"
).eval()

dummy = torch.randn(1, 32, 192)
mask = torch.zeros(1, 32, 32)       # 可选 attention mask

output_dir = compile(
    model=model,
    dummy_input=dummy,
    config_dir=str(INTEGRATION_CONFIG_DIR),
    mask=mask,
)
```

**compile() 常用参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `target_dtype` | 全局目标精度 | `None`（由 npu() 标注决定） |
| `target_format` | 全局目标格式 | `None` |
| `pass_toggles` | 开关可选 Pass | `None`（全开） |
| `tile_override` | 手动指定 tiling 参数 | `None` |
| `debug_dump` | 输出每个 Pass 后的中间图 | `False` |

### 验证生成的 C 工程

```bash
cd output/MyModel/
cmake .
make
./model_run
# 自动加载 golden 数据并比对精度
```

### 运行端到端 ST 测试

```bash
# 全量系统测试
pytest torch2c/integration/tests/demo_st/ -v

# 单模块 UT
pytest torch2c/format_planner/tests/ -v
pytest torch2c/memory_planner/tests/ -v
```

## 关键设计决策

| 决策项 | 结论 |
|--------|------|
| torch.export分解 | 禁止自动分解，保留layer_norm/softmax高级op |
| ATen算子命名 | 使用全称（如 `aten.mm.default`） |
| Format冲突 | format_planner 自动分配最优格式，不匹配时插入 dma_reformat |
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

## 文档

| 文档 | 说明 |
|------|------|
| [docs/ordr.md](docs/ordr.md) | 一阶段需求文档（权威来源，含补充决策记录） |
| [docs/architecture.md](docs/architecture.md) | 架构设计：管线、Graph IR、模块接口、配置系统 |
| [docs/dev-guide.md](docs/dev-guide.md) | 开发指南：扩展算子、问题定位、测试策略 |
| [docs/roadmap.md](docs/roadmap.md) | 开发路线图：改进方向与技术债清单 |
