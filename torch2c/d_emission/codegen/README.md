# codegen — Pass⑨：C代码生成

## 职责

根据编排结果和配置，生成完整的可编译C工程。

## 输入

- Graph IR（完整编排完成）
- DMA计划列表
- `config/c_api_signatures.yaml`
- `config/codegen_config.yaml`
- 权重数据（numpy数组）
- golden数据（PyTorch跑出的输入输出）

## 输出

完整C工程目录：

```
output/
├── CMakeLists.txt
├── src/
│   ├── model_graph.c / .h
│   ├── model_memory.h
│   ├── model_params.h
│   └── model_weights.h
├── utils/
│   ├── data_loader.c / .h
│   ├── data_dumper.c / .h
│   ├── comparator.c / .h
│   └── tensor_utils.h
├── golden/
│   ├── input_0.bin / .desc
│   └── output_0.bin / .desc
├── tests/
│   ├── test_data_loader.c
│   ├── test_comparator.c
│   ├── test_memory_layout.c
│   └── CMakeLists.txt
├── main.c
└── README.md
```

## 文件说明

| 文件 | 职责 |
|------|------|
| c_emitter.py | 生成model_graph.c/h — 主执行逻辑，每个算子生成三段式代码块：DMA搬入 → 算子调用 → DMA搬出 |
| weight_exporter.py | 将PyTorch权重导出为C静态数组（model_weights.h） |
| golden_exporter.py | 将PyTorch输入输出导出为二进制文件 + 描述文件 |
| utils_emitter.py | 生成辅助工具C代码：data_loader, data_dumper, comparator |
| mock_emitter.py | 已废弃（npu_mock.h 间接层已消除，生成代码直接 #include npu_api.h） |
| cmake_emitter.py | 生成CMakeLists.txt，支持mock模式和真实SDK模式 |

## 模板使用

`templates/` 目录下的 `.tmpl` 文件使用Python f-string插值：

| 占位符 | 说明 |
|--------|------|
| {op_id} | 算子编号 |
| {npu_op} | NPU算子名 |
| {compute_unit} | 计算单元名 |
| {params} | 展开的参数列表 |
| {l1_offset} | L1偏移 |
| {hbm_offset} | HBM偏移 |
| {size} | 搬运大小 |
| {src_fmt} / {dst_fmt} | format枚举（DMA随路转换：src=HBM存储格式，dst=算子期望格式） |

## 关键约束

- **npu_transpose 4D接口**：增加ndim和dims参数，支持高维tensor转置
- **DMA随路格式转换**：load指令的src_fmt=tensor的HBM存储格式，dst_fmt=消费者算子期望格式
- **absorbed_inputs处理**：如softmax_part1吸收mask后，通过optional_params中的mask参数传入（否则传NULL）

## demo/

**demo_input_plan.json:** 3个算子的完整编排结果（含HBM/L1偏移、DMA计划、依赖关系）

**run_demo.py:** 加载plan，生成C工程到demo_output/，然后执行 `gcc -fsyntax-only` 验证语法

## UT

**test_c_emitter.py:**
- `test_op_block_generation`: 单个算子生成正确的三段式代码
- `test_param_filling`: 参数根据signature正确填入
- `test_syntax_check`: 生成的C代码通过gcc语法检查

**test_utils_emitter.py:**
- `test_comparator_generation`: comparator.c语法正确
- `test_data_loader_generation`: data_loader.c语法正确
