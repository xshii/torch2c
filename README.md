# torch2c — PyTorch to NPU C Code Offline Compiler

NPU 离线编译栈：将 PyTorch 模型编译为可在自研 NPU 上运行的完整 C 工程。

## 编译流水线

```
PyTorch 模型
    │
    ▼
 a_capture ─── ① graph_capture        : torch.export → Graph IR
    │
 b_lowering ── ② op_mapping           : ATen op → NPU op（命名映射）
    │          ③ op_decomposition     : 裂解复合算子 + 广播展开 + 确认 is_mapped
    │          ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    │          [④ op_absorption]       : bias/scalar 吸收（可选）
    │          [④b mha_merge]          : MHA 投影 split/merge 优化（可选）
    │
 c_backend ─── ⑤ format_annotator     : 标注 tensor format/dtype
    │          [⑤a format_planner]     : 全局最优 format 分配（可选）
    │           ⑤b reformat_inserter   : 插入 format 转换节点
    │          [⑤c storage_assigner]   : 分配存储类型 hbm/local/pipe（可选）
    │          [⑤d block_pad]          : shape 对齐到硬件块（可选）
    │           ⑥ validator            : 合法性校验
    │          ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    │          [⑥b roofline]           : 算力/访存瓶颈分析（可选）
    │          [⑥c fusion]             : 算子融合规划（可选）
    │
 d_emission ── ⑦ scheduler            : 拓扑排序 + 依赖生成
    │          [⑦b global_tiler]       : L1 tiling 自动决策（可选）
    │           ⑧ memory_planner       : HBM/L1 地址分配 + DMA 计划
    │           ⑨ codegen              : 生成完整 C 工程
    ▼
输出：完整 C 工程目录
```

`[方括号]` = 可选 pass，可通过 `pass_toggles` 关闭。

## 项目结构

```
torch2c/
├── a_capture/                   Frontend
│   └── graph_capture/           ① torch.export → Graph IR
├── b_lowering/                  Op Lowering
│   ├── op_mapping/              ② ATen → NPU 命名映射
│   └── op_decomposition/        ③ 裂解 + 广播 + is_mapped 确认
├── c_backend/                   Target Annotation
│   ├── format_annotator/        ⑤ format/dtype 标注
│   ├── reformat_inserter/       ⑤b format 转换节点
│   └── validator/               ⑥ 合法性校验
├── d_emission/                  Scheduling + Codegen
│   ├── scheduler/               ⑦ 拓扑排序
│   ├── memory_planner/          ⑧ 内存编排
│   └── codegen/                 ⑨ C 代码生成
├── optpass/                     可选优化 Pass
│   ├── bc_op_absorption/        ④ 参数吸收
│   ├── bc_mha_merge/            ④b MHA 优化
│   ├── c_format_planner/        ⑤a 全局格式规划
│   ├── c_storage_assigner/      ⑤c 存储分配
│   ├── c_block_pad/             ⑤d 块对齐
│   ├── cd_roofline/             ⑥b 算力分析
│   ├── cd_fusion/               ⑥c 算子融合
│   └── d_global_tiler/          ⑦b 全局分块
├── common/                      基础设施（Graph IR、日志、配置、opt_log）
├── integration/                 管线串联 + 配置 + 端到端测试
├── viz/                         可视化（pipeline 流程图 + 甬道图）
│
├── npu_cpu_mock/                NPU C API 的 CPU 模拟实现
├── scripts/                     编译 + 可视化脚本
└── docs/                        文档
```

## 快速开始

### 环境

| 依赖 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.4+（需支持 torch.export） |
| PyYAML | 6.0+ |
| NumPy | 1.24+ |
| Flask | （可选，远端可视化） |

### 安装

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 编译模型

```python
import torch
from torch2c.integration.pipeline import compile
from torch2c.common.paths import INTEGRATION_CONFIG_DIR

model = YourModel().eval()
x = torch.randn(1, 32, 64)

compile(model, x,
        config_dir=str(INTEGRATION_CONFIG_DIR),
        output_dir="output/YourModel",
        debug_dump=True)  # 生成 debug 快照 + 可视化
```

### Demo 脚本

```bash
# Y = AX + B（最简单的线性层）
python scripts/demo_axb.py

# Embedding → Linear → ReLU
python scripts/demo_embedding.py

# 单层 Attention（2 头，含 LayerNorm）
python scripts/demo_single_attn.py

# 2 层 Encoder Transformer
python scripts/compile_and_viz.py
```

### 可视化

```bash
# 本地打开
open output/compile_viz/viz/pipeline.html

# 远端访问（Flask 服务器）
pip install flask
python scripts/viz_server.py --compile --port 8080
# 浏览器访问 http://<ip>:8080/
```

### 测试

```bash
# 全量 UT（441 个用例）
pytest

# 单模块
pytest torch2c/b_lowering/op_mapping/tests/ -v
pytest torch2c/optpass/bc_mha_merge/tests/ -v

# 端到端 ST
pytest torch2c/integration/tests/demo_st/ -v
```

### 验证 C 工程

```bash
cd output/YourModel/
cmake -B build && cmake --build build
cd build && ctest -V
```

## 硬件架构

目标 NPU 架构类似华为昇腾达芬奇体系。

| 计算单元 | 职责 | 典型算子 |
|----------|------|----------|
| Cube | 矩阵乘加（16×16×16 MAC） | matmul, linear |
| Vector | 逐元素/激活/归一化（SIMD） | relu, gelu, softmax, layernorm |
| IDMA | 片上数据搬运 + transpose | reshape, transpose, broadcast, embedding |
| DMA | HBM ↔ L1 搬运（随路格式转换） | load, store, ND↔NZ/ZZ 转换 |

**Tensor 格式**（详见 [docs/tensor_formats.md](docs/tensor_formats.md)）：

| 格式 | 用途 |
|------|------|
| ND | 标准行优先（默认） |
| NZ (Fractal_NZ) | Cube src1（权重），列优先分形块 |
| ZZ (Fractal_Z) | Cube src0（激活），行优先分形块 |
| NN | 列优先块 + 列优先元素 |

## 关键设计

| 决策 | 结论 |
|------|------|
| op_mapping | 纯命名翻译（ATen→NPU），不设 is_mapped |
| op_decomposition | 按 npu_op 查裂解表，裂解后才设 is_mapped=True |
| Format 语义 | tensor.format = HBM 存储格式，DMA load 时随路转换 |
| opt_log | 每个 pass 在节点上记录优化决策原因 |
| Graph.renumber() | 每个 pass 后重编号节点，保持 ID 与执行序一致 |
| Pass 耗时 | _run_pass_list 自动记录 duration_ms |

## 文档

| 文档 | 说明 |
|------|------|
| [docs/ordr.md](docs/ordr.md) | 需求文档（权威来源） |
| [docs/tensor_formats.md](docs/tensor_formats.md) | ND/NZ/ZZ/NN 格式详解 + 设计分析 |
| [docs/architecture.md](docs/architecture.md) | 架构设计 |
| [docs/dev-guide.md](docs/dev-guide.md) | 开发指南 |
| [docs/roadmap.md](docs/roadmap.md) | 路线图 + 技术债 |
