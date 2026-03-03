# op_mapping — Pass②：算子直接映射

## 职责

将ATen算子名替换为NPU算子名（1对1直接映射），标注compute_unit。

## 输入

- Graph IR（节点的op_type为ATen名）
- `config/direct_mappings.yaml`

## 输出

- Graph IR（已映射的节点：npu_op被填充，is_mapped=True，compute_unit被填充）
- 未映射的节点保持原样（留给op_decomposition或validator处理）

## 接口

```python
def run(graph: Graph, config: dict) -> Graph
```

## 处理逻辑

```python
for node in graph.nodes:
    if node.op_type in config.mappings:
        node.npu_op = config.mappings[node.op_type].npu_op
        node.compute_unit = config.mappings[node.op_type].compute_unit
        node.is_mapped = True
    else:
        # 跳过，留给后续Pass处理
```

## 日志

- INFO: `映射完成。已映射: X, 未映射: Y`
- DEBUG: 每个算子的映射结果

## 关键约束

- **ATen算子名使用全称**：如 `aten.mm.default`（需与graph_capture实际输出一致）
- 需先跑一次torch.export确认实际算子名后修正config

## config/direct_mappings.yaml

| ATen算子（全称） | NPU算子 | 计算单元 |
|-----------------|---------|----------|
| aten.mm.default | npu_matmul | cube |
| aten.add.Tensor | npu_add | vector |
| aten.mul.Tensor | npu_mul | vector |
| aten.mul.Scalar | npu_mul_scalar | vector |
| aten.gelu.default | npu_gelu | vector |
| aten.transpose.int | npu_transpose | vector |
| aten.t.default | npu_transpose_2d | vector |
| aten.reshape.default | npu_reshape | scalar |

> 注：以上算子名为预估全称，实际开发时需用graph_capture输出修正。

## demo/

**demo_input_graph.json:** 含5个节点的小图：mm → add → mul → gelu → reshape

**expected_output.json:** 映射后npu_op分别为：npu_matmul, npu_add, npu_mul, npu_gelu, npu_reshape

## UT

**test_op_mapping.py:**
- `test_basic_mapping`: 5个算子全部映射成功
- `test_unmapped_preserved`: 含一个不在配置中的算子，验证它保持未映射
- `test_compute_unit`: 验证matmul→cube, add→vector
- `test_empty_graph`: 空图不报错
