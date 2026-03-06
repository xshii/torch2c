# validator — Pass⑥：合法性校验

## 职责

验证所有节点都已映射到支持的NPU算子。

## 输入

- Graph IR（经过mapping、decomposition、absorption）
- `config/supported_ops.yaml`

## 输出

- 通过：返回原图不变
- 失败：抛出ValidationError，包含所有未支持算子的列表

## 接口

```python
def run(graph: Graph, config: dict) -> Graph
```

## 处理逻辑

```python
unsupported = []
for node in graph.nodes:
    if node.npu_op not in config.supported_ops:
        unsupported.append(f"{node.id}: {node.op_type} (npu_op={node.npu_op})")
if unsupported:
    raise ValidationError(f"以下算子未映射: {unsupported}")
```

## config/supported_ops.yaml

支持的NPU算子列表：
- npu_matmul, npu_add, npu_mul, npu_mul_scalar
- npu_gelu, npu_transpose, npu_transpose_2d, npu_reshape
- npu_layernorm_part1, npu_layernorm_part2
- npu_softmax_part1, npu_softmax_part2

## demo/

**demo_valid_graph.json:** 全部节点都在支持列表中 → 通过

**demo_invalid_graph.json:** 含一个 `npu_unknown` 节点 → 报错

## UT

**test_validator.py:**
- `test_all_supported`: 全部通过
- `test_one_unsupported`: 报错且错误信息包含具体算子名
- `test_multiple_unsupported`: 报错且列出所有未支持算子（不只第一个）
