# op_decomposition — Pass③：算子裂解

## 职责

将未映射的ATen算子按裂解规则替换为多个NPU算子节点（固定成组）。

## 输入

- Graph IR（经过op_mapping，部分节点已映射，部分未映射）
- `config/decompositions.yaml`

## 输出

- Graph IR（裂解后的节点替换原节点，中间tensor被创建）

## 接口

```python
def run(graph: Graph, config: dict) -> Graph
```

## 处理逻辑

```python
for node in graph.nodes (未映射的):
    if node.op_type in config.decompositions:
        rule = config.decompositions[node.op_type]
        # 1. 创建中间tensor
        # 2. 创建多个新节点，按order排列
        # 3. 连接输入输出（from: "source.input_N" 引用原节点输入）
        # 4. 删除原节点
        # 5. 标记新节点 is_mapped=True
```

## 日志

- INFO: `裂解完成。裂解了X个算子，新增Y个节点，新增Z个中间tensor`

## config/decompositions.yaml

裂解规则：

| 源算子 | 目标算子组 | 计算单元 |
|--------|-----------|----------|
| aten.layer_norm | npu_layernorm_part1 → npu_layernorm_part2 | vector |
| aten._softmax | npu_softmax_part1 → npu_softmax_part2 | vector |

## demo/

**demo_input_graph.json:** 含3个节点：layer_norm → mm → softmax（其中layer_norm和softmax未映射，mm已映射）

**expected_output.json:** 裂解后：layernorm_part1 → layernorm_part2 → mm → softmax_part1 → softmax_part2。节点数从3变为5，新增2个中间tensor。

## UT

**test_op_decomposition.py:**
- `test_layernorm_decompose`: layer_norm裂解为2个part
- `test_softmax_decompose`: softmax裂解为2个part
- `test_intermediate_tensor_created`: 中间tensor存在且shape正确
- `test_already_mapped_skipped`: 已映射的节点不被裂解
- `test_no_rule_preserved`: 没有裂解规则的未映射节点保持原样
