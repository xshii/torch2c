# CLAUDE.md — torch2c 项目规则

## 项目定位

torch2c 是一个 NPU 离线编译器：将 PyTorch 模型编译为可在自研 NPU 上运行的完整 C 工程。
目标硬件类似华为昇腾达芬奇架构，有 Cube（矩阵乘）、Vector（SIMD）、IDMA（片上搬运）、DMA（HBM搬运）四个计算/搬运单元。

## 编译流水线（17 个 pass）

```
阶段 A — Frontend
  ① graph_capture        : torch.export → Graph IR

阶段 B — Lowering
  ② op_mapping           : ATen op → NPU op（纯命名映射，不设 is_mapped）
  ③ op_decomposition     : 裂解复合算子 + 广播展开 + 确认 is_mapped

可选 Pass (bc_前缀 = B→C 之间)
  [④  op_absorption]     : bias/scalar 吸收
  [④b mha_merge]         : MHA 投影 split/merge 优化

阶段 C — Backend Annotation
  ⑤  format_annotator    : 标注 tensor format/dtype
  [⑤a format_planner]    : 全局最优 format 分配（可选）
  ⑤b reformat_inserter   : 插入 format 转换节点

可选 Pass (c_前缀 = C 阶段内)
  [⑤c storage_assigner]  : hbm/local/pipe 存储分配
  [⑤d block_pad]         : shape 对齐到硬件块尺寸

  ⑥  validator           : 合法性校验

可选 Pass (cd_前缀 = C→D 之间)
  [⑥b roofline]          : 算力/访存瓶颈分析
  [⑥c fusion]            : 算子融合规划

阶段 D — Emission
  ⑦  scheduler           : 拓扑排序 + 依赖生成

可选 Pass (d_前缀 = D 阶段内)
  [⑦b global_tiler]      : L1 tiling 自动决策

  ⑧  memory_planner      : HBM/L1 地址分配 + DMA 计划
  ⑨  codegen             : 生成完整 C 工程
```

方括号 = 可选 pass，通过 `pass_toggles` 关闭。

## 核心 IR 模型

```python
Graph:
  nodes: dict[str, Node]           # 所有计算节点
  tensors: dict[str, Tensor]       # 所有张量
  execution_order: list[str]       # 执行顺序（node id）
  dma_plans: dict[str, DmaPlan]    # DMA 计划（memory_planner 产出）

Node:
  id, op_type, npu_op              # 标识和算子名
  compute_unit: str                # cube / vector / idma / dma
  inputs: list[str]                # 输入 tensor id
  outputs: list[str]               # 输出 tensor id
  params: dict                     # 算子参数 + 优化元数据
  is_mapped: bool                  # decomposition 后才为 True
  format_annotation: dict | None   # format_annotator 填入

Tensor:
  id, shape, dtype, format         # format = HBM 存储格式 (nd/nz/zz/nn)
  original_shape: list | None      # block_pad 前的原始 shape
  hbm_offset, hbm_size             # memory_planner 分配
  l1_offset                        # L1 地址
  storage: str                     # hbm / local / pipe
  producer_node_id: str | None
  consumer_node_ids: list[str]
  is_weight, is_model_input, is_model_output: bool
```

## Tensor 格式系统

| 格式 | 块结构 | 用途 |
|------|--------|------|
| ND | 标准行优先（无分块） | 默认，Vector 输入/输出 |
| NZ (Fractal_NZ) | [⌈N/c0⌉, ⌈M/cube⌉, c0, cube] 列优先块 | Cube src1（权重） |
| ZZ (Fractal_Z) | [⌈M/cube⌉, ⌈N/c0⌉, cube, c0] 行优先块 | Cube src0（激活） |
| NN | [⌈N/c0⌉, ⌈M/cube⌉, c0, cube] 列优先块 | 列优先变体 |

format×dtype 对齐表（block_pad 使用）:

| 格式 | fp16 [dim[-2], dim[-1]] | int8 |
|------|-------------------------|------|
| ND | [1, 16] | [1, 32] |
| NZ | [16, 16] | [32, 16] |
| ZZ | [16, 16] | [16, 32] |
| NN | [16, 16] | [32, 16] |

核心语义：`tensor.format` = HBM 存储格式，DMA load 时按消费者需求随路转换到 L1。

## 配置系统

所有配置在 `torch2c/integration/config/` 下：

| 文件 | 内容 |
|------|------|
| `direct_mappings.yaml` | ATen op → NPU op 映射表 |
| `decompositions.yaml` | 复合算子裂解规则（key = npu_op） |
| `c_api_signatures.yaml` | NPU C API 函数签名 |
| `hardware_config.yaml` | 硬件参数（内存、分形块、format能力、block_pad对齐表） |
| `optimization_config.yaml` | 可选 pass 开关 |
| `tiling_config.yaml` | tiling 参数 |
| `cost_model_config.yaml` | 代价模型 |
| `naming_rules.yaml` | tensor 命名规则 |

**一致性约束**：一个算子必须同时出现在 mapping、signatures、tiling、naming、cost_model 中。
`test_config_consistency.py` 会自动检查。

## 工作流规则

- 方案决策需要用户同意，其余（写代码、跑测试、文件操作等）自行执行
- 每个模块核心代码 < 300 行，每个函数 < 50 行
- 使用 common 的 logger、config_loader、errors
- 修改后必须 `pytest` 全量通过（448 用例）
- 使用 `opt_log` 记录优化决策原因
- 每个 pass 后自动 `graph.renumber()` 重编号

## 技术约束

- Python 3.10, PyTorch 2.4+, C99
- 测试：pytest (Python), ctest (C)
- 精度：max_abs_diff < 2.0 (FP16 端到端), cosine > 0.95
- 全量测试：`.venv/bin/pytest --tb=short -q`
- 单模块测试：`.venv/bin/pytest torch2c/<module>/tests/ -v`

## 目录结构

```
torch2c/
├── a_capture/graph_capture/     ① Frontend
├── b_lowering/                  Lowering
│   ├── op_mapping/              ② 命名映射
│   └── op_decomposition/        ③ 裂解 + 广播
├── c_backend/                   Backend Annotation
│   ├── format_annotator/        ⑤ format/dtype 标注
│   ├── reformat_inserter/       ⑤b format 转换
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
├── common/                      基础设施（Graph IR、日志、配置、sizing、opt_log）
├── integration/                 管线串联 + 配置 + 端到端测试
├── viz/                         可视化
├── npu_cpu_mock/                NPU C API 的 CPU 模拟
└── scripts/                     编译 + 可视化脚本
```

每个模块目录结构：`code.py` + `__init__.py` + `tests/` + `demo/`（可选）+ `config/`（可选）

## 关键文件速查

| 需求 | 文件 |
|------|------|
| 改 IR 数据结构 | `torch2c/common/graph_ir.py` |
| 改内存大小计算 | `torch2c/common/sizing.py` (calc_padded_size + get_dim_align) |
| 改 pass 管线顺序 | `torch2c/integration/pipeline.py` (_OPTIMIZATION_PASSES) |
| 改 pass 开关 | `torch2c/common/pass_config.py` (OptionalPass enum) |
| 加新算子映射 | `torch2c/integration/config/direct_mappings.yaml` |
| 加算子裂解 | `torch2c/integration/config/decompositions.yaml` |
| 加 C 函数签名 | `torch2c/integration/config/c_api_signatures.yaml` |
| 加 C mock 实现 | `npu_cpu_mock/src/` + `npu_cpu_mock/include/npu_api.h` |
| 改硬件参数 | `torch2c/integration/config/hardware_config.yaml` |
| 改 codegen 模板 | `torch2c/d_emission/codegen/` |
| 改可视化 | `torch2c/viz/` |

## 文档

| 文档 | 说明 |
|------|------|
| `docs/ordr.md` | 需求文档（权威来源） |
| `docs/tensor_formats.md` | ND/NZ/ZZ/NN 格式详解 |
| `docs/architecture.md` | 架构设计 |
| `docs/roadmap.md` | 路线图 + 技术债 |
