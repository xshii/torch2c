# 新增算子 Checklist

添加新的 ATen → NPU 算子映射时，按此 checklist 逐项确认，避免遗漏。

## 历史教训

| Bug | 根因 | 防范 |
|-----|------|------|
| `aten.linear.default` 结果错误 | 缺少 weight 转置（linear = x @ w.T + b） | 单算子语义测试 |
| `npu_softmax`/`npu_layernorm` dtype 错误 | `type_format_config.yaml` 缺少条目 | 配置一致性测试 |

## Step 1: 确定映射策略

- [ ] **1-to-1 映射**：添加到 `integration/config/direct_mappings.yaml`
- [ ] **1-to-N 裂解**：添加到 `integration/config/decompositions.yaml`
- [ ] 确认 PyTorch 算子语义（参考 [PyTorch Docs](https://pytorch.org/docs/stable/generated/)）
- [ ] 识别隐含语义：转置、broadcast、alpha/beta 系数、多输出等

## Step 2: 配置文件（全部必须同步更新）

| 文件 | 说明 | 必须 |
|------|------|------|
| `direct_mappings.yaml` 或 `decompositions.yaml` | ATen → NPU 映射规则 | Y |
| `type_format_config.yaml` | format/dtype 标注（包括 _part1/_part2 变体） | Y |
| `c_api_signatures.yaml` | C API 参数签名 | Y |

> 自动校验：`test_config_consistency.py` 会交叉检查上述三个文件的算子集合一致性。

## Step 3: C Mock 实现

- [ ] 在 `npu_cpu_mock/src/` 中实现 CPU 模拟
- [ ] 在 `npu_cpu_mock/include/npu_api.h` 中声明函数签名
- [ ] 添加单算子 C 测试到 `integration/tests/demo_ut/test_c_ops.py`

## Step 4: 语义测试

- [ ] 在 `integration/tests/test_op_semantics.py` 中添加最小模型测试
- [ ] 对比 PyTorch reference 与编译后 C 输出
- [ ] 精度要求：max_abs_diff < 2e-2 (FP16), cosine > 0.999

## Step 5: 特殊处理（如有）

- [ ] **需要 weight 转置的算子**：在 `graph_capture/config/capture_rules.yaml` 的 `weight_transpose_ops` 中注册
- [ ] **裂解算子**：确认 _part1 和 _part2 都在 `type_format_config.yaml` 中
- [ ] **带标量参数的算子**：确认 `c_api_signatures.yaml` 中 param source 正确

## Step 6: 验证

```bash
# 配置一致性
pytest torch2c/integration/tests/test_config_consistency.py -v

# 单算子语义
pytest torch2c/integration/tests/test_op_semantics.py -v

# 单算子 C 精度
pytest torch2c/integration/tests/demo_ut/test_c_ops.py -v

# 端到端 golden 比对
pytest torch2c/integration/tests/test_pipeline.py::TestCGoldenComparison -v

# 全量测试
pytest --tb=short
```
