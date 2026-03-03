# scheduler — Pass⑧：计算单元调度与依赖生成

## 职责

确定算子执行顺序，生成算子间的依赖关系。

## 输入

- Graph IR（已编排完成，每个节点有compute_unit）

## 输出

- Graph IR（每个节点的schedule_order和dependencies被填充）

## 接口

```python
def run(graph: Graph, config: dict = None) -> Graph
```

## 处理逻辑（保守策略）

1. 拓扑排序确定基本执行顺序
2. 遍历相邻算子对：
   - 有数据依赖（后者的输入是前者的输出）→ 插入依赖
   - 无数据依赖且使用不同compute_unit → 可并行（不插入依赖）
   - 无数据依赖但使用相同compute_unit → 串行（插入依赖）
3. DMA操作后始终插入barrier

## 日志

- INFO: `调度完成。依赖关系: N条，可并行算子对: M`

## 关键约束

- memory_planner已用topo_sort确定基本执行顺序，scheduler在此基础上细化并行机会
- Reshape正常参与调度（compute_unit=scalar），不做特殊跳过

## demo/

**demo_input_graph.json:** 4个算子：matmul(cube) → add(vector) → matmul(cube) → gelu(vector)

**expected_output.json:** 所有4个算子串行（因为有数据依赖链），依赖关系: [(0,1), (1,2), (2,3)]

## UT

**test_scheduler.py:**
- `test_linear_chain`: 线性依赖链的依赖关系正确
- `test_parallel_opportunity`: 两个无依赖的不同compute_unit算子不产生依赖
- `test_same_unit_serialized`: 两个无依赖的相同compute_unit算子串行化
- `test_schedule_order`: 每个节点的schedule_order按拓扑排序递增
