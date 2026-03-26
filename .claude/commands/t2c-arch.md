# /t2c-arch — 架构大脑

> 合并自: arch + debug(参考) + setup
> 用法: `/t2c-arch [topic]`
> 主题: `arch` | `flow` | `ir` | `hardware` | `format` | `config` | `setup` | `tools`
> 示例: `/t2c-arch arch` | `/t2c-arch format` | `/t2c-arch setup`

你是 torch2c NPU 编译器的架构顾问。根据 topic 提供深度技术参考。
$ARGUMENTS 包含 topic。若为空，给出总览。

---

## 编译流程总览

```
PyTorch nn.Module
  │ torch.export（禁止自动分解，保留高级 op）
  ▼
① graph_capture → Graph IR (ATen ops)
  │
② op_mapping → NPU ops（命名映射，is_mapped=False）
③ op_decomposition → 原子 NPU ops（裂解 + 广播，is_mapped=True）
  │
  │ ── 可选 pass ──
  │ [④  op_absorption]      bias/scalar 吸收
  │ [④b mha_merge]          MHA 投影优化
  │
⑤ format_annotator → 标注 format/dtype
  │ [⑤a format_planner]     全局格式优化
  │ ⑤b reformat_inserter    插入格式转换节点
  │ [⑤c storage_assigner]   hbm/local/pipe 分配
  │ [⑤d block_pad]          shape 对齐到硬件块
  │
⑥ validator → 合法性校验
  │ [⑥b roofline]           算力/访存分析
  │ [⑥c fusion]             算子融合规划
  │
⑦ scheduler → 拓扑排序 + 依赖
  │ [⑦b global_tiler]       L1 tiling
  │
⑧ memory_planner → HBM/L1 地址 + DMA 计划
⑨ codegen → 完整 C 工程
```

方括号 = 可选 pass，通过 `pass_toggles` 关闭。

---

## 硬件模型

```
+======================================+
|               NPU Core               |
|  +---------+  +---------+            |
|  |  Cube   |  | Vector  |            |
|  | 16×16×16|  |  SIMD   |            |
|  |  MAC    |  |  128w   |            |
|  +----+----+  +----+----+            |
|       |             |                 |
|  +----+-------------+----+           |
|  |         L1 SRAM        | 16 MB    |
|  +------------+-----------+           |
|          +----+----+                  |
|          |  IDMA   | 片上搬运         |
|          |  DMA    | HBM↔L1          |
|          +----+----+                  |
+===============|=======================+
           +----+----+
           |   HBM   | 4 GB
           +---------+
```

| 单元 | 职责 | 典型算子 |
|------|------|----------|
| Cube | 矩阵乘加 (16×16×16 MAC) | cube_matmul, cube_matmul_bias |
| Vector | 逐元素/SIMD (128 宽) | vector_add, vector_gelu, vector_softmax |
| IDMA | 片上搬运 + transpose | idma_reshape, idma_transpose |
| DMA | HBM ↔ L1（随路格式转换） | dma_move, dma_reformat |

---

## Graph IR 生命周期

```python
Graph:
  nodes: dict[str, Node]           # 计算节点
  tensors: dict[str, Tensor]       # 张量
  execution_order: list[str]       # 执行顺序
  dma_plans: dict[str, DmaPlan]    # DMA 计划（memory_planner 产出）
  metadata: dict                   # 全局元数据（roofline_summary, fusion_groups 等）

Node:
  id, op_type, npu_op, compute_unit
  inputs/outputs: list[str]        # tensor id
  params: dict                     # 算子参数 + 优化元数据（_opt_log, _weight_slices 等）
  is_mapped: bool                  # decomposition 后才为 True
  format_annotation: dict | None

Tensor:
  id, shape, dtype, format         # format = HBM 存储格式
  original_shape: list | None      # block_pad 前的原始 shape
  hbm_offset, hbm_size, l1_offset  # memory_planner 分配
  storage: str                     # hbm / local / pipe
  producer_node_id, consumer_node_ids
  is_weight, is_model_input, is_model_output
```

### IR 各阶段状态变化

| 阶段 | 变化 |
|------|------|
| graph_capture | op_type="aten.xxx", format="nd", is_mapped=False |
| op_mapping | npu_op="vector_xxx", compute_unit 设置 |
| op_decomposition | 复合→原子, is_mapped=True |
| format_annotator | tensor.format 可能变 nz/zz, format_annotation 填入 |
| block_pad | shape 对齐到分形块, original_shape 保存 |
| scheduler | execution_order 排序, dependencies 填入 |
| memory_planner | hbm_offset/hbm_size/l1_offset 分配, dma_plans 生成 |

---

## Tensor 格式系统

**核心语义**: `tensor.format` = HBM 存储格式。DMA load 时硬件自动转换为消费者需要的 L1 格式。

| 格式 | 内存布局 | 用途 |
|------|----------|------|
| ND | 标准行优先 | 默认，Vector 输入/输出 |
| NZ (Fractal_NZ) | [⌈N/c0⌉, ⌈M/cube⌉, c0, cube] | Cube src1（权重） |
| ZZ (Fractal_Z) | [⌈M/cube⌉, ⌈N/c0⌉, cube, c0] | Cube src0（激活） |
| NN | [⌈N/c0⌉, ⌈M/cube⌉, c0, cube] | 列优先变体 |

cube_size=16（固定），c0 随 dtype: fp16=16, int8=32

### format × dtype 对齐表

| 格式 | fp16 [dim[-2], dim[-1]] | int8 |
|------|-------------------------|------|
| ND | [1, 16] | [1, 32] |
| NZ | [16, 16] | [32, 16] |
| ZZ | [16, 16] | [16, 32] |
| NN | [16, 16] | [32, 16] |

### format_capabilities

```yaml
cube:    src0=[nd,zz], src1=nz, dst=[nd,nz,zz]
vector:  src=[nd], dst=[nd]
idma:    src=[nd,nz,zz,nn], dst=[nd,nz,zz,nn]
```

### 计算 padded 字节数

```python
from torch2c.common.sizing import calc_padded_size, get_dim_align
size = calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
```

---

## 配置系统

所有配置在 `torch2c/integration/config/`：

| 文件 | 内容 |
|------|------|
| `direct_mappings.yaml` | ATen op → NPU op |
| `decompositions.yaml` | 裂解规则 |
| `c_api_signatures.yaml` | C API 签名 |
| `hardware_config.yaml` | 硬件参数（内存/分形/format/对齐） |
| `optimization_config.yaml` | 可选 pass 开关 |
| `tiling_config.yaml` | tiling 参数 |
| `cost_model_config.yaml` | 代价模型（3 层: Python fn > YAML per-op > YAML unit default） |
| `naming_rules.yaml` | tensor 命名 |

**一致性约束**: 一个算子必须同时出现在 mapping/signatures/tiling/naming/cost_model 中。
`test_config_consistency.py` 自动检查。

配置流转:
```
所有 YAML → pipeline._build_pass_configs() → pass_configs: dict → 各 pass.run(graph, config)
```

---

## 内存策略

memory_planner 按优先级尝试:

1. **bulk** — 所有 tensor 同时放 L1（最快，无 DMA 开销）
2. **perop** — 按算子 liveness 复用 L1（标准策略）
3. **spill** — 选择性换出到 HBM（L1 紧张时）
4. **tiled** — M 维分块 + 多轮 DMA（最后手段）

---

## 关键设计决策

| 决策 | 结论 | 原因 |
|------|------|------|
| mapping 不设 is_mapped | decomposition 确认后才设 | 映射只是命名翻译 |
| format = HBM 格式 | DMA load 随路转换 | 避免显式 format_convert 节点 |
| block_pad 用 format×dtype 表 | 不同格式不同对齐 | 分形块尺寸和 dtype 相关 |
| Graph.renumber() | 每个 pass 后执行 | 保持 node ID 与执行序一致 |
| opt_log 在 node.params | 每个 pass 追加 | 可视化和 debug 需要优化理由 |

---

## 关键文件速查

| 需求 | 文件 |
|------|------|
| 改 IR 数据结构 | `torch2c/common/graph_ir.py` |
| 改内存大小计算 | `torch2c/common/sizing.py` |
| 改 pass 管线顺序 | `torch2c/integration/pipeline.py` |
| 改 pass 开关 | `torch2c/common/pass_config.py` |
| 加算子映射 | `integration/config/direct_mappings.yaml` |
| 加裂解规则 | `integration/config/decompositions.yaml` |
| 加 C 签名 | `integration/config/c_api_signatures.yaml` |
| 加 C mock | `npu_cpu_mock/src/` + `npu_cpu_mock/include/npu_api.h` |
| 改硬件参数 | `integration/config/hardware_config.yaml` |
| 改 codegen 模板 | `torch2c/d_emission/codegen/` |
| 改可视化 | `torch2c/viz/` |

---

## 环境搭建

### 快速安装

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 验证

```bash
# 全量测试（~8s, 510 用例）
.venv/bin/pytest --tb=short -q

# 编译 demo
python scripts/compile_and_viz.py
open output/compile_viz/viz/pipeline.html

# C golden 验证
cd output/demo_axb/ && cmake -B build && cmake --build build && cd build && ctest -V
```

### VSCode Tasks 速查

`Cmd+Shift+P` → `Tasks: Run Task`:

| 开发 | 测试 | 编译 |
|------|------|------|
| `run:current` | `test:all` | `compile:minimal` |
| `run:current+viz` | `test:current` | `compile:full` |
| `install:dev` | `test:module` | `compile:both` |
| | `test:no-golden` | `compile:debug` |
| | `test:block-fuser` | `demo:e2e` |
| | `test:codegen` | `demo:module` |
| | `test:roofline` | `demo:viz` |

### 依赖

| 依赖 | 版本 |
|------|------|
| Python | ==3.10.* |
| PyTorch | >=2.4 (torch.export) |
| PyYAML | >=6.0 |
| NumPy | >=1.24 |

---

## 目录结构

```
torch2c/
├── a_capture/graph_capture/     ① Frontend
├── b_lowering/
│   ├── op_mapping/              ② 映射
│   └── op_decomposition/        ③ 裂解
├── c_backend/
│   ├── format_annotator/        ⑤ format 标注
│   ├── reformat_inserter/       ⑤b format 转换
│   └── validator/               ⑥ 校验
├── d_emission/
│   ├── scheduler/               ⑦ 拓扑排序
│   ├── memory_planner/          ⑧ 内存编排
│   └── codegen/                 ⑨ C 代码生成
├── optpass/                     可选 Pass
│   ├── bc_op_absorption/        ④ 参数吸收
│   ├── bc_mha_merge/            ④b MHA 优化
│   ├── c_format_planner/        ⑤a 格式规划
│   ├── c_storage_assigner/      ⑤c 存储分配
│   ├── c_block_pad/             ⑤d 块对齐
│   ├── cd_roofline/             ⑥b 算力分析
│   ├── cd_fusion/               ⑥c 算子融合
│   ├── cd_block_fuser/          ⑥c block 级融合
│   └── d_global_tiler/          ⑦b 全局分块
├── common/                      基础设施
├── integration/                 管线 + 配置 + 端到端测试
├── viz/                         可视化
├── npu_cpu_mock/                C mock
└── scripts/                     脚本

每个模块: code.py + __init__.py + tests/ + demo/(可选) + config/(可选)
```
