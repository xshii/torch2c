# 新增 NPU 算子清单

按顺序完成以下步骤。每步都有具体的文件路径和代码片段，直接复制粘贴后修改 TODO 部分。

---

## 0. 前置信息收集

开始前确认以下信息（填在这里方便后面引用）：

| 字段 | 值 |
|------|------|
| ATen 算子名 | `aten.TODO.default`（先跑 `torch.export` 确认） |
| NPU 算子名 | `TODO_op_name`（前缀规则：cube_/vector_/idma_/dma_） |
| 计算单元 | TODO：cube / vector / idma |
| 输入 tensor 数 | TODO：1 / 2 / ... |
| 输出 tensor 数 | TODO：通常 1 |
| 额外参数 | TODO：如 scalar、axis 等 |
| 输出 shape 规则 | TODO：与输入相同 / 有变化 |

---

## 1. 算子映射

**文件**: `torch2c/integration/config/direct_mappings.yaml`

```yaml
# TODO: 添加到文件末尾
aten.TODO.default:
  npu_op: TODO_op_name
  compute_unit: TODO_unit  # cube / vector / idma
```

---

## 2. C API 签名

**文件**: `torch2c/integration/config/c_api_signatures.yaml`

在对应 section（`cube_ops` / `vector_ops` / `idma_ops`）下添加：

```yaml
  # TODO: 选择合适的签名模式

  # ── 一元算子模式 ──
  TODO_op_name:
    unit: TODO_unit
    params:
      - { name: "input",         type: "tensor_desc", source: "tensor.input_0" }
      - { name: "out",           type: "tensor_desc", source: "tensor.output_0" }
      - { name: "count",         type: "int",         source: "tensor.input_0.count" }
      - { name: "compute_dtype", type: "dtype_enum",  source: "param.compute_dtype", default: "fp16" }

  # ── 二元算子模式 ──
  # TODO_op_name:
  #   unit: TODO_unit
  #   params:
  #     - { name: "a",             type: "tensor_desc", source: "tensor.input_0" }
  #     - { name: "b",             type: "tensor_desc", source: "tensor.input_1" }
  #     - { name: "out",           type: "tensor_desc", source: "tensor.output_0" }
  #     - { name: "count",         type: "int",         source: "tensor.input_0.count" }
  #     - { name: "compute_dtype", type: "dtype_enum",  source: "param.compute_dtype", default: "fp16" }

  # ── 带标量参数模式 ──
  # TODO_op_name:
  #   unit: TODO_unit
  #   params:
  #     - { name: "input",         type: "tensor_desc", source: "tensor.input_0" }
  #     - { name: "out",           type: "tensor_desc", source: "tensor.output_0" }
  #     - { name: "scalar",        type: "float",       source: "param.scalar" }
  #     - { name: "count",         type: "int",         source: "tensor.input_0.count" }
  #     - { name: "compute_dtype", type: "dtype_enum",  source: "param.compute_dtype", default: "fp16" }
```

source 类型速查：
- `tensor.input_N` / `tensor.output_N` — tensor descriptor
- `tensor.input_N.count` — 元素总数
- `tensor.input_N.ndim` — 维度数
- `tensor.input_N.shape` — shape 数组（type 用 `int_array`）
- `tensor.input_N.hbm_size` — 字节数
- `param.xxx` — node.params 中的值
- `param.compute_dtype` — 计算精度

---

## 3. Tiling 配置

**文件**: `torch2c/integration/config/tiling_config.yaml`

```yaml
# TODO: 添加
TODO_op_name:
  tile_dim: -2          # tiling 维度（-2 = 倒数第二维）
  min_tile: 1
```

---

## 4. 命名规则

**文件**: `torch2c/integration/config/naming_rules.yaml`

```yaml
# TODO: 添加
TODO_op_name:
  short_name: TODO_short  # codegen 中的简短名
```

---

## 5. 代价模型

**文件**: `torch2c/integration/config/cost_model_config.yaml`

```yaml
# TODO: 添加
TODO_op_name:
  flops_per_element: TODO_N  # 每元素浮点运算数估算
```

---

## 6. C Mock 声明

**文件**: `npu_cpu_mock/include/npu_api.h`

在对应 section 添加函数声明：

```c
/* TODO: 选择合适的签名 */

/* 一元 */
void TODO_op_name(TidInfo tid, npu_tensor_t input, npu_tensor_t out,
                  int count, npu_dtype_t compute_dtype);

/* 二元 */
/* void TODO_op_name(TidInfo tid, npu_tensor_t a, npu_tensor_t b,
                     npu_tensor_t out, int count, npu_dtype_t compute_dtype); */
```

---

## 7. C Mock 实现

**文件**: `npu_cpu_mock/src/npu_compute_TODO.c`（新建或追加到已有文件）

直接使用 `templates/new_op_c_mock_template.c` 作为起点，复制后修改：
1. 把所有 `TODO_OP_NAME` 替换为你的算子名
2. 修改函数签名匹配步骤 6 的声明
3. 填写计算逻辑（`float result = ...` 那一行）

> **关键提醒**：
> - 用 `npu_t_ptr(t)` 获取指针，**不要用** `t.addr`
> - 用 `npu_read_compute()` / `npu_write_compute()` 做精度转换
> - `NPU_TRACE_BEGIN` 和 `NPU_TRACE_END` 参数必须一致

---

## 8. 验证配置一致性

```bash
.venv/bin/pytest torch2c/integration/tests/test_config_consistency.py -v
```

如果报错，说明某个配置表漏了新算子，回去补全。

---

## 9. 单元测试

在相关模块的 `tests/` 下添加测试，确保：
- mapping 能正确映射
- codegen 能生成正确的 C 调用
- C mock 能编译通过并产出正确结果

---

## 10. 全量回归

```bash
.venv/bin/pytest --tb=short -q
```

---

## 附录：如果算子需要裂解

有些复合算子（如 layernorm -> part1 + part2）需要裂解规则。

**文件**: `torch2c/integration/config/decompositions.yaml`

```yaml
decompositions:
  TODO_op_name:                      # 裂解前的 npu_op
    steps:
      - npu_op: TODO_op_name_part1
        compute_unit: TODO_unit
      - npu_op: TODO_op_name_part2
        compute_unit: TODO_unit
```

裂解后的每个子算子也需要完成上面的步骤 2-8。
