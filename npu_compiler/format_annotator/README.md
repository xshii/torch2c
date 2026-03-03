# format_annotator — Pass⑤：Format/Dtype标注

## 职责

根据每个NPU算子的format/dtype要求，标注每个tensor的preferred format和dtype。

## 输入

- Graph IR（所有节点已映射为NPU算子）
- `config/type_format_config.yaml`

## 输出

- Graph IR（每个节点的format_annotation被填充）

## 接口

```python
def run(graph: Graph, config: dict) -> Graph
```

## 关键约束

- **tensor.format语义**：表示该tensor在HBM中的存储格式
- **DMA随路转换**：同一tensor被不同format需求的算子消费时，不插入显式format_convert节点，由DMA搬运时自动完成格式转换
- **format_annotation扩展**：输入format/dtype、计算dtype、输出format/dtype 均可不同

## 处理逻辑

```python
for node in graph.nodes:
    req = config.op_format_requirements[node.npu_op]
    node.format_annotation = {
        "inputs": [{"format": r.format, "dtype": r.dtype} for r in req.inputs],
        "outputs": [{"format": r.format, "dtype": r.dtype} for r in req.outputs],  # 输出format/dtype可与输入不同
        "compute_dtype": req.compute_dtype,       # 计算精度，可与输入输出均不同
        "supports_format_convert": req.supports_format_convert,
        "supports_dtype_cast": req.supports_dtype_cast
    }
    # 更新tensor的HBM存储格式（由生产者决定）
    for i, tensor_id in enumerate(node.outputs):
        if i < len(req.outputs):
            graph.tensors[tensor_id].format = req.outputs[i].format
            graph.tensors[tensor_id].dtype = req.outputs[i].dtype
```

## 日志

- INFO: `Format标注完成。标注了N个节点，M个tensor`

## config/type_format_config.yaml

关键规则（含compute_dtype）：
- `npu_matmul`: 输入输出 format=nz, dtype=fp16, compute_dtype=fp16
- `npu_add/mul/gelu` 等Vector算子: format=nd, dtype=fp16, compute_dtype=fp16
- `npu_layernorm_part1/part2`: format=nd, dtype=fp32, compute_dtype=fp32

## demo/

**demo_input_graph.json:** 含2个节点：matmul(cube) → add(vector)，所有tensor初始format=nd, dtype=fp16

**expected_output.json:** matmul的输入tensor标注为format=nz, add的输入tensor标注为format=nd

## UT

**test_format_annotator.py:**
- `test_matmul_format`: matmul输入标注为nz
- `test_vector_format`: add输入标注为nd
- `test_annotation_structure`: format_annotation字段结构正确
