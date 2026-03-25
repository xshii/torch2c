# graph_capture — Pass①：图捕获

## 职责

使用 torch.export 将 PyTorch nn.Module 导出为 ATen IR 图，转换为 Graph IR。

## 环境

- PyTorch 2.4+（Python 3.10兼容）

## 关键约束

- **禁止torch.export自动分解**：传入自定义decomposition table，保留 `aten.layer_norm`、`aten._softmax` 等高级算子
- **ATen算子名使用全称**：如 `aten.mm.default`、`aten.add.Tensor`（需跑torch.export确认实际名称）
- **多输出算子全部保留**：如 `aten.layer_norm` 返回 `(output, mean, rstd)`，全部创建Tensor对象，无消费者的由memory_planner回收
- **Scalar值作为常量tensor**：`is_weight=True`，`shape=[1]`（如attention中 `scores * 0.25` 的0.25）
- **Mask输入**：Demo模型包含attention mask，作为模型外部输入，`is_model_input=True`，`shape=[1, 1, 32, 32]`

## 输入

- `model`: torch.nn.Module
- `dummy_input`: torch.Tensor（固定shape的样例输入）
- `mask`: torch.Tensor（可选，attention mask，shape=[1, 1, 32, 32]）

## 输出

- Graph IR（所有节点的op_type为ATen算子全称，如 `aten.mm.default`）
- 权重tensor标记 `is_weight=True`
- 常量scalar tensor标记 `is_weight=True`，`shape=[1]`
- 模型输入/输出tensor标记 `is_model_input/is_model_output=True`

## 接口

```python
def capture(model: nn.Module, dummy_input: torch.Tensor) -> Graph
```

## 日志

- INFO: `图捕获完成，节点数: N, tensor数: M, 权重tensor数: W`
- DEBUG: 每个节点的算子类型和输入输出tensor

## demo/

**demo_model.py:** 定义2层Encoder Transformer的PyTorch模型（小shape版，含attention mask输入）。
- hidden_size=64, num_heads=4, ffn_dim=256, seq_len=32, batch=1
- forward(x, mask) — mask shape=[1, 1, 32, 32]

**run_demo.py:** 捕获模型图并保存为JSON。

**expected_output.json:** 预期算子类型清单（全称）：
- 关键算子：aten.mm.default, aten.add.Tensor, aten.layer_norm, aten._softmax, aten.gelu 等
- 预期节点数：40~80
- 预期权重tensor数：20~40

## UT

**test_graph_capture.py:**
- `test_capture_linear`: 单个nn.Linear导出，验证有mm节点
- `test_capture_encoder`: 完整encoder导出，验证关键算子类型存在
- `test_weight_marking`: 验证权重tensor的is_weight=True
- `test_io_marking`: 验证输入输出tensor的标记正确
