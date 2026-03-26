# 块级数据流融合架构设计

## 1. 问题

当前 pass 链的四个独立贪心决策互相制约：

```
storage_assigner(⑤c) → 逐 pair 判断 local/pipe
fusion_planner(⑥c)   → 线性链检测 + fan-out>1 提升
global_tiler(⑦b)     → 逐节点独立选 tile size
memory_planner(⑧)    → L1 不够时被动 spill
```

每步局部最优但全局次优：融合不看 L1 容量、tiling 不看融合组、storage 不看 tiling。

## 2. 目标

用一个 `block_fuser` pass 替换 `fusion_planner + global_tiler`，联合决策：
- **哪些 tensor 留 L1**（storage）
- **哪些算子共享 tile 循环**（fusion）
- **tile 多大**（tiling）

核心：在融合决策时**显式建模数据在 HBM↔L1 间的搬运成本**（Blockbuster 思路）。

## 3. 不变量

以下组件**不改动**：

| 组件 | 原因 |
|------|------|
| Graph IR（Node/Tensor） | block_fuser 输出标准字段，不加新 IR 类型 |
| storage_assigner（⑤c） | pipe 是硬件约束，block_fuser 尊重其决策 |
| scheduler（⑦） | 只读 fusion 标注确保组内连续，逻辑不变 |
| memory_planner（⑧） | 消费同样的 storage + _tile_config 接口 |

**唯一需要扩展的下游**：codegen（⑨）— 对融合组生成共享 tile 循环。

## 4. 输出接口（兼容现有）

block_fuser 的输出格式和 fusion_planner + global_tiler 完全一致：

```python
# 1. storage 决策
tensor.storage = "local"  # 组内中间 tensor
tensor.storage = "hbm"    # 外部 tensor

# 2. tile 决策
node.params["_tile_config"] = {
    "tile_size": 64,
    "num_buffers": 2,
    "num_tiles": 4,
}

# 3. 融合组
graph.metadata["fusion_groups"] = [
    {"id": "fg_0", "node_ids": [...], "internal_tensors": [...], ...}
]
node.params["_fusion_group"] = "fg_0"
node.params["_fusion_role"] = "head" | "middle" | "tail"
```

## 5. 模块结构

```
torch2c/optpass/cd_block_fuser/
├── __init__.py               # run, post_validate
├── block_fuser.py            # run() 编排：构建 → 融合 → tile → 写回
├── _block_graph.py           # 块级数据流图（pass 内部临时结构）
├── _fusion.py                # 贪心融合算法
├── _tiling.py                # 组内联合 tile 决策
├── _patterns/                # 特殊 pattern 检测
│   ├── __init__.py
│   └── _attention.py         # Phase 2: attention DAG
└── tests/
    ├── test_block_fuser.py
    ├── test_block_graph.py
    ├── test_fusion.py
    └── test_tiling.py

torch2c/d_emission/codegen/
├── _fused_emitter.py         # NEW: 融合组代码生成
└── c_emitter.py              # 修改: 检测 fusion_group → 调用 fused_emitter
```

## 6. 核心数据结构

### 6.1 块级数据流图（_block_graph.py）

pass 内部临时结构，不持久化到 Graph IR。

```python
@dataclass
class DataBlock:
    """tensor 在内存层级间的搬运模型。"""
    tensor_id: str
    size_bytes: int                # padded size
    producer_id: str | None        # 生产者节点 id
    consumer_ids: list[str]        # 消费者节点 id 列表
    current_tier: str              # hbm / local / pipe（来自 storage_assigner）
    is_external: bool              # model_input / model_output / weight

    # 融合收益 = 消除这条 HBM 搬运能省多少 cycles
    @property
    def elimination_benefit(self) -> int:
        if self.is_external or self.current_tier != "hbm":
            return 0
        return 2 * self.size_bytes  # load + store（归一化为 bytes）

    # L1 压力 = 这个 tensor 在 L1 中占多少 bytes
    @property
    def l1_pressure(self) -> int:
        return self.size_bytes


@dataclass
class ComputeBlock:
    """算子在块级图中的表示。"""
    node_id: str
    compute_unit: str
    compute_cycles: int            # from roofline / cost_model
    launch_cycles: int
    input_block_ids: list[str]     # DataBlock.tensor_id
    output_block_ids: list[str]
    # tiling 信息
    tileable: bool
    tile_dim: int                  # 哪个维度可 tile
    tile_dim_size: int             # 该维度原始大小


@dataclass
class BlockGraph:
    """块级数据流图 — pass 内部临时结构。"""
    compute_blocks: dict[str, ComputeBlock]
    data_blocks: dict[str, DataBlock]
    topo_order: list[str]          # 拓扑序（compute block ids）

    @classmethod
    def from_graph(cls, graph: Graph, hw: RooflineHwParams,
                   cost_model: CostModel) -> BlockGraph:
        """从 Graph IR + roofline 数据构建。"""
        ...
```

### 6.2 融合组（_fusion.py）

```python
@dataclass
class FusionCandidate:
    """一个融合组的候选方案。"""
    node_ids: list[str]            # 拓扑序排列的节点
    internal_blocks: set[str]      # 组内 tensor（留 L1）
    external_input_blocks: set[str]   # 外部输入（DMA load）
    external_output_blocks: set[str]  # 外部输出（DMA store）
    total_benefit: int             # 消除的 HBM 搬运 bytes
    l1_peak: int                   # L1 峰值占用（未 tile 时）
```

## 7. 算法

### 7.1 融合算法（_fusion.py）

```
输入：BlockGraph, L1 容量
输出：list[FusionCandidate]

1. 初始化：每个 ComputeBlock 独立成组
2. 收集所有可融合边（DataBlock），按 elimination_benefit 降序排列
3. 对每条边：
   a. 找 producer 和 consumer 所在的组
   b. 如果已在同一组 → 跳过
   c. 尝试合并两组 → 计算合并后 L1 峰值
   d. 如果 L1 峰值 ≤ 容量（考虑最优 tile 后）→ 提交合并
   e. 否则 → 不融合，保留 HBM 搬运
4. 返回所有组（过滤掉单节点组）
```

**和现有 fusion_planner 的区别**：
- 不限制 fan-out == 1（多消费者也能融合，只要 L1 放得下）
- 不限制线性链（DAG 也行，只要拓扑序合法）
- 融合决策内嵌 L1 容量检查
- 按收益排序（优先融合大 tensor）

### 7.2 联合 Tile 决策（_tiling.py）

```
输入：FusionCandidate, L1 容量, 硬件参数
输出：tile_size, num_buffers

1. 收集组内所有可 tile 节点的 tile_dim_size
2. 确定共享 tile_dim（取最小公倍数或最严格约束）
3. 二分搜索最大 tile_size：
   a. 计算 L1 峰值 = Σ(tiled tensor size) for 所有组内 tensor
   b. 如果 peak ≤ L1_cap → 可行，尝试更大
   c. 如果 peak > L1_cap → 缩小
4. 检查是否可 ping-pong（peak × 2 ≤ L1_cap）
5. 返回 (tile_size, num_buffers=2 if ping-pong else 1)
```

**和现有 global_tiler 的区别**：
- 组内所有节点共享 tile_size（而非各自独立选）
- tile 决策和融合决策联合（融合组太大时可以选更小的 tile）

### 7.3 L1 峰值估算

```python
def estimate_l1_peak(group: FusionCandidate, tile_size: int,
                     block_graph: BlockGraph) -> int:
    """估算融合组在给定 tile_size 下的 L1 峰值占用。

    在一个 tile 迭代内，L1 同时存在：
    - 外部输入的 tiled 切片（DMA load 进来的）
    - 组内中间 tensor 的 tiled 切片（不出 L1）
    - 外部输出的 tiled 切片（等待 DMA store）
    """
    peak = 0
    for tid in (group.external_input_blocks
                | group.internal_blocks
                | group.external_output_blocks):
        db = block_graph.data_blocks[tid]
        tiled_size = _calc_tiled_size(db.size_bytes, tile_size, ...)
        peak += tiled_size
    return peak
```

## 8. Codegen 扩展

### 8.1 检测融合组（c_emitter.py 修改）

在 `_gen_grouped_body` 中，检测连续节点是否属于同一 `_fusion_group`：

```python
def _gen_grouped_body(...):
    # 将 execution_order 按 fusion_group 分段
    segments = _segment_by_fusion(order, nodes)
    for segment in segments:
        if segment.is_fused:
            # 调用融合代码生成器
            code = gen_fused_block(segment.node_ids, ...)
        else:
            # 原有逻辑：逐节点生成
            code = gen_op_block(...)
```

### 8.2 融合组代码生成（_fused_emitter.py）

```c
/* === Fusion Group fg_0: cube_matmul → vector_relu → vector_add === */
/* tiled 256→4×64, 2-buf */
{
    /* untiled loads (weights, constants) */
    dma_move(/*load*/ weight_B, l1_B, ...);

    for (int _tile = 0; _tile < 4; _tile++) {
        int _buf = _tile % 2;
        int _l1_base = _buf * BUF_SIZE;

        /* load external inputs (tiled) */
        dma_move(/*load*/ input_A + _tile * stride_A, l1_A + _l1_base, ...);

        /* --- fused compute: intermediates stay in L1 --- */

        /* node_0: cube_matmul */
        npu_cube_matmul(l1_A + _l1_base, l1_B, l1_mm_out + _l1_base, ...);

        /* node_1: vector_relu (reads l1_mm_out, writes l1_relu_out) */
        npu_vector_relu(l1_mm_out + _l1_base, l1_relu_out + _l1_base, ...);

        /* node_2: vector_add (reads l1_relu_out + bias) */
        npu_vector_add(l1_relu_out + _l1_base, l1_bias, l1_out + _l1_base, ...);

        /* store external outputs (tiled) */
        dma_move(/*store*/ l1_out + _l1_base, output + _tile * stride_out, ...);
    }
}
```

**生成规则**：
1. 外部 weight/constant → 循环前 load（untiled）
2. 外部 input → 循环内 load（tiled）
3. 内部 tensor → 不生成 DMA，只分配 L1 offset
4. 外部 output → 循环内 store（tiled）
5. 各 compute op 保持原有参数，只替换 tensor 地址为 L1 offset

## 9. Pipeline 集成

### 9.1 Pass 替换

```python
# pipeline.py _LATE_PASSES
_PassDesc("block_fuser", "⑥c", block_fuser.run, None,
          toggle=OptionalPass.BLOCK_FUSER),
```

### 9.2 互斥开关

```python
# pass_config.py
class PassConfig:
    block_fuser: bool = False      # 默认关闭（开发中）
    fusion_planner: bool = True    # 默认开启
    global_tiler: bool = True      # 默认开启

    def is_enabled(self, pass_id):
        # block_fuser 开启时自动禁用 fusion_planner + global_tiler
        if pass_id == OptionalPass.FUSION_PLANNER and self.block_fuser:
            return False
        if pass_id == OptionalPass.GLOBAL_TILER and self.block_fuser:
            return False
        return getattr(self, pass_id.name.lower())
```

### 9.3 config 传递

block_fuser 和 roofline 一样，config_key=None（传入完整 configs dict），从中读取：
- `configs["hardware"]` → L1 容量、DMA 带宽
- `configs["cost_model"]` → per-op cycles
- `configs["hardware"]["tile_override"]` → 手动 tile 覆盖

## 10. Phase 2: Attention 扩展点

`_patterns/_attention.py` 在 block_fuser 的融合阶段**之前**运行：

```python
def run(graph, config):
    block_graph = BlockGraph.from_graph(graph, hw, cost_model)

    # Phase 2 扩展点：特殊 pattern 检测
    attention_blocks = detect_attention_pattern(block_graph)
    for attn in attention_blocks:
        # 将 Q@K^T → softmax → @V 标记为强制融合
        # 计算两级 tile size (Tq, Tk)
        # 标记 online softmax 变换
        ...

    # 通用贪心融合
    groups = fuse_blocks(block_graph, l1_cap)

    # 合并 attention 融合组 + 通用融合组
    all_groups = attention_blocks + groups

    # 联合 tile 决策
    for group in all_groups:
        assign_tile_sizes(group, l1_cap, hw)

    # 写回
    apply_decisions(graph, all_groups)
```

attention 融合不依赖通用融合框架的特殊之处：
- 两级 tile（Tq × Tk）而非一级
- 需要 online softmax 数学变换
- codegen 用专门的 `_attention_emitter.py`

但它**复用**通用框架的：
- BlockGraph 数据结构
- L1 峰值估算
- 写回接口
- tile 决策的约束检查

## 11. 数据流总览

```
                    Graph IR
                       │
        ┌──────────────┼──────────────┐
        │              │              │
  storage_assigner   roofline    cost_model
   (pipe/local)     (cycles)    (per-op params)
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                  block_fuser (⑥c)
              ┌────────┼────────┐
              │        │        │
         _block_graph  │   _patterns/
         (构建)        │   (attention)
              │        │        │
              ▼        ▼        ▼
           _fusion  _tiling   (Phase 2)
              │        │        │
              └────────┼────────┘
                       │
                       ▼ (写回 Graph IR)
            ┌──────────┼──────────┐
            │          │          │
      tensor.storage  _tile_config  fusion_groups
      (local/hbm)   (tile_size)   (metadata)
            │          │          │
            └──────────┼──────────┘
                       │
                       ▼
                  scheduler (⑦)
                       │
                       ▼
                memory_planner (⑧)
                       │
                       ▼
                  codegen (⑨)
              ┌────────┼────────┐
              │        │        │
         c_emitter  _fused_    _attention_
         (per-node) emitter    emitter
                   (fusion    (Phase 2)
                    group)
```

## 12. 测试策略

| 层次 | 测试内容 | 关键断言 |
|------|----------|----------|
| _block_graph | 从 Graph 构建 BlockGraph | DataBlock.elimination_benefit 正确 |
| _fusion | 线性链融合 | 和现有 fusion_planner 结果一致 |
| _fusion | DAG 融合 | fan-out tensor 在组内 → local |
| _fusion | L1 超容不融合 | 超容量的组不被创建 |
| _tiling | 组内共享 tile | 所有节点同一 tile_size |
| _tiling | ping-pong | L1 够 2 倍 → num_buffers=2 |
| block_fuser | 端到端 | 输出接口和 fusion_planner+global_tiler 兼容 |
| block_fuser | 回退 | BLOCK_FUSER=False 时旧 pass 正常工作 |
| codegen | 融合代码生成 | 组内 tensor 无 DMA，外部 tensor 有 DMA |
| integration | 编译 + C mock | 正确性不变，DMA 指令数减少 |

## 13. 实施计划

```
Week 1: Phase 0 — codegen 消费融合标注
  ├── _fused_emitter.py（融合组共享 tile 循环）
  ├── c_emitter.py 修改（检测 fusion_group → 调用 fused_emitter）
  └── 测试：matmul→relu 融合组代码正确

Week 2: Phase 1a — block_fuser 基础
  ├── _block_graph.py（BlockGraph 构建）
  ├── _fusion.py（贪心融合 + L1 约束）
  └── 测试：线性链 + DAG + L1 超容

Week 3: Phase 1b — 联合 tiling + 集成
  ├── _tiling.py（组内共享 tile 决策）
  ├── block_fuser.py（run 编排）
  ├── pipeline.py（pass 接入 + 互斥开关）
  └── 测试：端到端 + 回退 + 集成

Week 4: Phase 2 — Attention 融合
  ├── _patterns/_attention.py（pattern detection）
  ├── online softmax 变换
  ├── _attention_emitter.py（嵌套 tile 循环）
  ├── C mock npu_vector_softmax_online
  └── 测试：单头 attention 融合 + 精度
```
