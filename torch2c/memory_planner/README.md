# memory_planner — Pass⑦：内存编排

## 职责

为所有tensor分配HBM偏移地址，为每个算子规划L1布局，生成DMA搬运计划。

## 输入

- Graph IR（校验通过，所有节点已映射）
- `config/hardware_config.yaml`

## 输出

- Graph IR（每个tensor的hbm_offset, hbm_size, l1_offset被填充）
- 附加数据：DMA计划列表（每个算子的load/store指令序列）

## 接口

```python
def run(graph: Graph, config: dict) -> tuple[Graph, list[DmaPlan]]

@dataclass
class DmaInstruction:
    op: str                    # "load" | "store"
    tensor_id: str
    hbm_offset: int
    l1_offset: int
    size_bytes: int
    src_format: str
    dst_format: str

@dataclass
class DmaPlan:
    node_id: str
    loads: list[DmaInstruction]
    stores: list[DmaInstruction]
```

## 处理逻辑

### HBM层（全局）

1. 计算每个tensor的padded_size（分形格式对齐）
2. 分析生命周期：first_use = min(消费者的执行顺序), last_use = max(消费者的执行顺序)
3. 按first_use排序，best-fit分配HBM偏移
4. last_use之后的tensor空间标记为可复用
5. 地址按 `hardware_config.memory.hbm.alignment_bytes` 对齐

### L1层（每个算子独立）

1. 按顺序排列：输入tensor → 权重tensor → 输出tensor
2. 每个tensor起始地址按 `hardware_config.memory.l1.alignment_bytes` 对齐
3. 检查总用量不超过L1容量

### DMA计划

每个算子生成固定三段：
1. load: 所有输入和权重从HBM→L1（DMA随路完成format转换，src_format=HBM存储格式，dst_format=算子期望格式）
2. （算子执行，不在DMA计划中）
3. store: 所有输出从L1→HBM

### 关键约束

- **tensor.format = HBM存储格式**，DMA load时按消费者的format_annotation需求转换
- **Reshape正常搬运**：与其他算子一样走完整DMA load/store流程，不做零拷贝优化
- **多输出算子的无消费者输出**：如layer_norm的mean/rstd，无消费者时不分配HBM空间（生命周期分析时自动跳过）

### 关键函数

- `calc_padded_size(shape, dtype, format, cube_size) -> int` — 计算padding后的字节数
- `align_up(offset, alignment) -> int` — 向上对齐

## config/hardware_config.yaml

| 存储层 | 容量 | 对齐 |
|--------|------|------|
| HBM | 4GB | 512B |
| L2 | 32MB | 256B |
| L1 | 16MB | 32B |

## demo/

**demo_input_graph.json:** 5个算子的线性链，tensor shape统一为[1,32,64], dtype=fp16

**expected_output.json:** 所有offset >= 0，相邻活跃tensor不重叠，满足对齐要求

## UT

**test_memory_planner.py:**
- `test_no_overlap`: 同时活跃的tensor地址不重叠
- `test_alignment`: HBM offset是512的倍数，L1 offset是32的倍数
- `test_reuse`: 线性链中dead tensor的空间被后续tensor复用
- `test_padded_size`: calc_padded_size对非对齐shape正确padding
- `test_dma_plan`: 每个算子有正确数量的load和store指令
- `test_l1_capacity_check`: L1溢出时抛出MemoryPlanError
