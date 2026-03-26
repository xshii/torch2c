# 架构深度理解

阅读本指南后你应该能回答：这个编译器怎么把一个 PyTorch 模型变成 C 代码？

## 编译流程总览

```
PyTorch nn.Module
    │
    ▼ torch.export (禁止自动分解，保留高级 op)
Graph IR (ATen ops)
    │
    ▼ op_mapping (查 direct_mappings.yaml)
Graph IR (NPU ops，但 is_mapped=False)
    │
    ▼ op_decomposition (查 decompositions.yaml，按 npu_op)
Graph IR (NPU 原子 ops，is_mapped=True)
    │
    ▼ [可选 passes: absorption, mha_merge]
    │
    ▼ format_annotator (标注每个 tensor 的 format/dtype)
    ▼ [format_planner] → reformat_inserter (插 format 转换节点)
    ▼ [storage_assigner] (hbm/local/pipe)
    ▼ [block_pad] (shape 对齐到分形块)
    ▼ validator (合法性校验)
    │
    ▼ [roofline, fusion]
    │
    ▼ scheduler (拓扑排序 + 依赖生成)
    ▼ [global_tiler] (L1 tiling)
    ▼ memory_planner (HBM/L1 地址 + DMA 计划)
    ▼ codegen (生成 .c/.h/CMakeLists.txt)
    │
    ▼
完整 C 工程目录
```

## 硬件模型

```
╔══════════════════════════════════════╗
║               NPU Core              ║
║  ┌─────────┐  ┌─────────┐          ║
║  │  Cube   │  │ Vector  │          ║
║  │ 16×16×16│  │  SIMD   │          ║
║  │  MAC    │  │  128w   │          ║
║  └────┬────┘  └────┬────┘          ║
║       │             │               ║
║  ┌────┴─────────────┴────┐          ║
║  │         L1 SRAM       │ 16 MB    ║
║  │    (计算数据暂存)      │          ║
║  └────────────┬──────────┘          ║
║          ┌────┴────┐                ║
║          │  IDMA   │ 片上搬运       ║
║          │  DMA    │ HBM↔L1        ║
║          └────┬────┘                ║
╚═══════════════╪══════════════════════╝
           ┌────┴────┐
           │   HBM   │ 4 GB
           │ (权重+IO)│
           └─────────┘
```

### 四种计算/搬运单元

| 单元 | 职责 | 典型算子 |
|------|------|----------|
| Cube | 矩阵乘加 (16×16×16 MAC) | cube_matmul, cube_matmul_bias |
| Vector | 逐元素/SIMD (128 宽) | vector_add, vector_gelu, vector_softmax |
| IDMA | 片上搬运 + transpose | idma_reshape, idma_transpose, idma_embedding |
| DMA | HBM ↔ L1 搬运 (随路格式转换) | dma_move, dma_reformat |

### Tensor 格式

**核心语义**: `tensor.format` = HBM 存储格式。DMA load 时硬件自动转换为消费者需要的 L1 格式。

| 格式 | 内存布局 | 用途 |
|------|----------|------|
| ND | 标准行优先 | 默认，Vector 输入/输出 |
| NZ (Fractal_NZ) | [⌈N/c0⌉, ⌈M/cube⌉, c0, cube] | Cube src1（权重）|
| ZZ (Fractal_Z) | [⌈M/cube⌉, ⌈N/c0⌉, cube, c0] | Cube src0（激活）|
| NN | [⌈N/c0⌉, ⌈M/cube⌉, c0, cube] | 列优先变体 |

cube_size=16（固定），c0 随 dtype 变化：fp16=16, int8=32。

### 内存策略

memory_planner 按优先级尝试：
1. **bulk** — 所有 tensor 同时放 L1（最快，无 DMA 开销）
2. **perop** — 按算子 liveness 复用 L1（标准策略）
3. **spill** — 选择性换出到 HBM（L1 紧张时）
4. **tiled** — M 维分块 + 多轮 DMA（最后手段）

## Graph IR 生命周期

```
graph_capture 创建:
  - node.op_type = "aten.xxx"
  - tensor.format = "nd"
  - node.is_mapped = False

op_mapping 后:
  - node.npu_op = "vector_xxx"
  - node.compute_unit = "vector"
  - node.is_mapped 仍为 False（mapping 不设）

op_decomposition 后:
  - 复合算子被拆为多个原子算子
  - 所有原子算子 node.is_mapped = True

format_annotator 后:
  - tensor.format 可能变为 nz/zz
  - node.format_annotation 被填入
  - tensor.dtype 被确认

block_pad 后:
  - tensor.shape 对齐到分形块（format×dtype 对齐表）
  - tensor.original_shape 保存原始值

scheduler 后:
  - graph.execution_order 按依赖排序
  - node.dependencies 被填入

memory_planner 后:
  - tensor.hbm_offset, hbm_size 被分配
  - tensor.l1_offset 被分配
  - graph.dma_plans 被生成（load/store 指令）

codegen 后:
  - 输出目录包含完整 C 工程
```

## 配置流转

```
hardware_config.yaml ──┐
direct_mappings.yaml ──┤
decompositions.yaml ───┼──→ pipeline._build_pass_configs()
c_api_signatures.yaml ─┤    │
tiling_config.yaml ────┤    ▼
cost_model_config.yaml ┤  pass_configs: dict
naming_rules.yaml ─────┘    │
                             ▼
                    各 pass 的 run(graph, config)
```

每个 pass 只看自己的 config 子集，由 pipeline 负责分发。

## 关键设计决策

| 决策 | 结论 | 原因 |
|------|------|------|
| op_mapping 不设 is_mapped | decomposition 确认后才设 | 映射只是命名翻译，裂解才确认算子原子性 |
| format = HBM 格式 | DMA load 时随路转换 | 避免显式 format_convert 节点 |
| block_pad 用 format×dtype 表 | 不同格式不同对齐 | NZ/ZZ 分形块尺寸和 dtype 相关 |
| Graph.renumber() | 每个 pass 后执行 | 保持 node ID 与执行序一致 |
| opt_log 记录在 node.params | 每个 pass 追加 | 可视化和 debug 需要优化理由 |
