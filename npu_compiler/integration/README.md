# integration — 管线串联与端到端测试

## 职责

串联所有Pass模块，实现从PyTorch模型到C工程的完整编译流水线。**本模块最后开发。**

## 流水线

```
graph_capture → op_mapping → op_decomposition → op_absorption
    → format_annotator → validator → memory_planner → scheduler → codegen
```

## 接口

```python
def compile(model: nn.Module, dummy_input: torch.Tensor, config_dir: str) -> str
    """
    完整编译流水线。
    返回生成的C工程输出目录路径。
    """
```

## 关键约束

- **config与模块局部完全相同**：integration/config/ 是各模块config/的副本，集成时统一从此处加载
- **Demo数据**：各Agent先按README规格自行构造demo JSON，后续用graph_capture真实输出修正

## config/

汇总所有模块的配置文件（与各模块局部config完全相同）：

| 配置文件 | 用途 |
|---------|------|
| direct_mappings.yaml | ATen→NPU 1对1映射 |
| decompositions.yaml | 裂解规则 |
| absorptions.yaml | 吸收规则 |
| c_api_signatures.yaml | NPU C接口函数签名 |
| type_format_config.yaml | dtype/format枚举与算子要求 |
| hardware_config.yaml | 硬件存储参数 |
| model_config.yaml | 模型参数 |
| codegen_config.yaml | 代码生成选项 |

## demo/

**encoder_model.py:** 2层Encoder Transformer模型定义

**run_full_demo.py:** 执行完整编译流水线，生成C工程

**validate_output.py:** 验证生成的C工程正确性

## 验收标准

| 检查项 | 方法 |
|--------|------|
| 所有模块UT通过 | `pytest --tb=short` |
| 端到端管线跑通 | `python integration/demo/run_full_demo.py` 无报错 |
| C代码语法正确 | `gcc -fsyntax-only -include npu_mock.h output/src/model_graph.c` |
| C侧UT通过 | `cd output && cmake . && make && ctest` |
| 算子数量正确 | model_graph.c中62个算子块 |
| 日志完整 | 每个Pass有入口/出口INFO日志 |

## UT

**test_pipeline.py:**
- 端到端流水线集成测试
