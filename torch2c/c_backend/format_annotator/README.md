# format_annotator — Pass⑤：Format/Dtype标注

## 职责

根据模型自身的 dtype/format 标注每个 tensor。用户可通过 `target_dtype` / `target_format` 覆盖。

## 输入

- Graph IR（所有节点已映射为NPU算子）
- config dict（可选 `target_dtype`、`target_format`）

## 输出

- Graph IR（每个节点的format_annotation被填充）

## 接口

```python
def run(graph: Graph, config: dict) -> Graph
```

config 可选键：
- `target_dtype`: 全局目标 dtype（如 "fp16"），None 则继承模型原始值
- `target_format`: 全局目标 format（如 "nz"），None 则继承模型原始值

## 关键约束

- **tensor.format语义**：表示该tensor在HBM中的存储格式
- **DMA随路转换**：同一tensor被不同format需求的算子消费时，不插入显式format_convert节点，由DMA搬运时自动完成格式转换
- **dtype/format 完全灵活**：硬件支持所有 dtype 和 format，不做静态约束

## 处理逻辑

```python
for node in graph.nodes:
    # dtype/format 来自模型自身或用户覆盖
    for tid in node.inputs:
        t = graph.get_tensor(tid)
        annotation = {"format": target_format or t.format, "dtype": target_dtype or t.dtype}
    for tid in node.outputs:
        t = graph.get_tensor(tid)
        t.format = target_format or t.format
        t.dtype = target_dtype or t.dtype
```

## 日志

- INFO: `Format标注完成。标注了N个节点，M个tensor`

## UT

**test_format_annotator.py:**
- `test_inherit_model_dtype`: 默认继承模型原始 dtype
- `test_inherit_model_format`: 默认继承模型原始 format
- `test_target_dtype_override`: target_dtype 覆盖模型 dtype
- `test_target_format_override`: target_format 覆盖模型 format
- `test_annotation_structure`: format_annotation 字段结构正确
