# op_absorption — Pass④：参数吸收

## 职责

将独立的算子（如mask的add）吸收为相邻算子的可选参数。

## 输入

- Graph IR（经过mapping和decomposition，所有节点都已映射）
- `config/absorptions.yaml`

## 输出

- Graph IR（被吸收的节点被删除，目标节点的absorbed_inputs被填充）

## 接口

```python
def run(graph: Graph, config: dict) -> Graph
```

## 处理逻辑

```python
for rule in config.absorptions:
    # 在图中寻找匹配pattern：
    #   1. 找到absorbed_op类型的节点
    #   2. 检查它的输出是否只有一个消费者
    #   3. 检查消费者是否是target_op类型
    #   4. 匹配成功：
    #      - 将被吸收节点的指定输入添加到目标节点的absorbed_inputs
    #      - 重连：目标节点的对应输入改为被吸收节点的非mask输入
    #      - 删除被吸收节点及其输出tensor
```

## 日志

- INFO: `吸收完成。吸收了X个算子，消除了Y个中间tensor`

## config/absorptions.yaml

吸收规则：

| 被吸收算子 | 目标算子 | 吸收参数 |
|-----------|---------|---------|
| npu_add | npu_softmax_part1 | mask（input_1 → 目标的mask参数） |

## 关键约束

- Demo模型包含attention mask（模型外部输入，`is_model_input=True`，`shape=[1, 1, 32, 32]`）
- `add(scores, mask) → softmax_part1` 的pattern确保可被匹配，实现端到端验证

## demo/

**demo_input_graph.json:** 含3个节点：add(scores, mask) → softmax_part1 → softmax_part2
- mask tensor: `is_model_input=True`，`shape=[1, 1, 32, 32]`

**expected_output.json:** 吸收后节点数从3变为2（add被删除），softmax_part1的absorbed_inputs = {"mask": "tensor_mask"}

## UT

**test_op_absorption.py:**
- `test_mask_absorption`: add被吸收进softmax_part1
- `test_absorbed_input_recorded`: softmax_part1的absorbed_inputs包含mask
- `test_intermediate_removed`: add的输出tensor被删除
- `test_no_match_preserved`: 不匹配规则的add保持不变（如残差add）
- `test_empty_rules`: absorptions为空列表时图不变
