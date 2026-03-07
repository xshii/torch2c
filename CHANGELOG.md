# Changelog

## [Unreleased]

### C Codegen Architecture Overhaul

#### npu_tensor_t API: `void* ptr` 直接指针
- `npu_tensor_t.addr` (uint32_t) 改为 `npu_tensor_t.ptr` (void*)，消除全局 `npu_l1_base` 和地址移位
- `npu_api.h` / `npu_mock.h` / `mock_emitter.py` / `test_c_ops.py` 同步更新
- 移除 `c_api_signatures.yaml` 中的 `addr_shift` 配置

#### 结构体管理 tensor descriptor
- 生成 `model_tensors_t` 结构体，按使用阶段划分子结构体：
  `inputs` / `weights` / `layer0_self_attn` / `layer0` / ... / `outputs`
- `model_run` 中通过 designated initializer 一次性初始化所有 tensor
- 子函数通过 `model_tensors_t*` 指针访问，消除跨函数重复声明

#### 语义化变量命名
- 权重：state_dict key 缩写 (`l0_sa_q_proj_weight`)
- 中间结果：producer 算子类型 + 编号 (`mm_bias_2`, `softmax_15`)
- 输入/输出：`in_32` / `out_107`

#### 函数封装（按 Python 类层级）
- 从 `torch.export` 的 `nn_module_stack` 提取模块路径
- `Node` 新增 `module_path` 字段，`graph_capture` 填充
- 按模块路径分组生成 `static void` 函数：`layer0_self_attn()`, `layer0()`, ...

#### 宏定义消除魔法数字
- Tensor spec 宏：`T_FP16_NZ(base, off)`, `T_FP32_ND(base, off)` 等
- 模型维度宏：`D_MODEL`, `SEQ_LEN`, `DIM_FF`, `BATCH`
- op 调用中自动替换匹配的字面量为维度宏

### Other
- `npu()` 标注系统：支持 `NpuSpec` / `npu_input()` / `@torch2c_config` 装饰器
- `format_npu_annotations()` 图级标注摘要
- `inspect()` 快速诊断入口
