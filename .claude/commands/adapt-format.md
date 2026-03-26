# Format/Dtype 适配指南

当需要修改 tensor 格式、对齐规则、内存大小计算时，参考本指南。

## 核心概念

### format×dtype 对齐表

定义在 `torch2c/integration/config/hardware_config.yaml` 的 `block_pad.alignment` 中：

```yaml
block_pad:
  alignment:           # format → dtype → [dim[-2]对齐, dim[-1]对齐]
    nd:
      fp16: [1, 16]    # 行不对齐，列对齐到 SIMD 宽度
      int8: [1, 32]
    nz:
      fp16: [16, 16]   # Fractal_NZ: 行=c0, 列=cube_size
      int8: [32, 16]   # int8 c0=32
    zz:
      fp16: [16, 16]   # Fractal_Z: 行=cube_size, 列=c0
      int8: [16, 32]   # int8 c0=32
    nn:
      fp16: [16, 16]
      int8: [32, 16]
  fallback: [16, 16]
  single_dim: 256
```

### 计算对齐值的两种方式

```python
# 方式 1: 从配置解析（block_pad pass 内部用）
from torch2c.optpass.c_block_pad.block_pad import parse_alignment_table, get_align_rule
table, fallback = parse_alignment_table(config)
rule = get_align_rule(table, fallback, fmt="nz", dtype="fp16")
# rule.dim_neg2=16, rule.dim_neg1=16

# 方式 2: 便捷函数（其他模块用）
from torch2c.common.sizing import get_dim_align
align = get_dim_align("nz", "fp16")  # → (16, 16)
align = get_dim_align("zz", "int8")  # → (16, 32)
```

### 计算 padded 字节数

```python
from torch2c.common.sizing import calc_padded_size, get_dim_align

size = calc_padded_size(
    shape=[1, 30, 60],
    dtype="fp16",
    fmt="nz",
    dim_align=get_dim_align("nz", "fp16"),  # (16, 16)
)
# 30 → 32 (对齐到16), 60 → 64 (对齐到16)
# size = 1 * 32 * 64 * 2 = 4096 bytes
```

## 修改场景

### 场景 1: 添加新的 dtype

1. 在 `hardware_config.yaml` 的 `block_pad.alignment` 每个 format 下添加新 dtype
2. 在 `torch2c/common/sizing.py` 的 `_DEFAULT_ALIGNMENT` 表中添加对应项
3. 在 `torch2c/common/dtypes.py` 中添加 dtype 信息（bytes、C enum、numpy 映射）
4. 在 `fractal.c0_by_dtype` 中添加 c0 值
5. 跑测试：`pytest torch2c/optpass/c_block_pad/tests/ -v`

### 场景 2: 添加新的 format

1. 在 `hardware_config.yaml`:
   - `block_pad.alignment` 添加新 format section
   - `format_capabilities` 添加各单元对新 format 的支持
2. 在 `torch2c/common/sizing.py`:
   - `_DEFAULT_ALIGNMENT` 添加新 format
   - `calc_padded_size` 的分形格式判断中添加新 format
3. 在 `torch2c/d_emission/codegen/_helpers.py`:
   - `FORMAT_MAP` 添加新 format → C enum 映射
4. 在 `npu_cpu_mock/include/npu_api.h`:
   - `npu_format_t` 枚举添加新值
5. 跑全量测试

### 场景 3: 修改对齐规则

1. 修改 `hardware_config.yaml` 的 `block_pad.alignment`
2. **同步修改** `torch2c/common/sizing.py` 的 `_DEFAULT_ALIGNMENT`（两者必须一致）
3. 跑 block_pad 测试：`pytest torch2c/optpass/c_block_pad/tests/ -v`
4. 跑全量回归

### 场景 4: format_annotator 规则修改

format_annotator 决定每个 tensor 应该用什么 format：

```
torch2c/c_backend/format_annotator/format_annotator.py
torch2c/integration/config/hardware_config.yaml → format_capabilities
```

format_capabilities 定义了每个计算单元支持的格式：

```yaml
format_capabilities:
  cube:
    src0: [nd, zz]       # 激活可以是 ND 或 ZZ
    src1: nz              # 权重必须是 NZ
    dst: [nd, nz, zz]
  vector:
    src: [nd]             # Vector 只支持 ND
    dst: [nd]
  idma:
    src: [nd, nz, zz, nn] # IDMA 支持所有
    dst: [nd, nz, zz, nn]
```

## 关键文件

| 文件 | 职责 |
|------|------|
| `torch2c/common/sizing.py` | `calc_padded_size` + `get_dim_align` |
| `torch2c/optpass/c_block_pad/block_pad.py` | shape 对齐 pass |
| `torch2c/c_backend/format_annotator/format_annotator.py` | format 标注 |
| `torch2c/optpass/c_format_planner/format_planner.py` | 全局格式优化 |
| `torch2c/c_backend/reformat_inserter/reformat_inserter.py` | 格式转换节点 |
| `torch2c/d_emission/codegen/_helpers.py` | FORMAT_MAP (format → C enum) |
| `torch2c/integration/config/hardware_config.yaml` | 硬件参数 |
| `docs/tensor_formats.md` | 格式详解文档 |

## 注意事项

- `_DEFAULT_ALIGNMENT` (sizing.py) 和 `block_pad.alignment` (hardware_config.yaml) **必须保持同步**
- 修改 format 后必须检查 `format_capabilities` 是否需要更新
- 所有 `calc_padded_size` 调用都必须用 `get_dim_align(t.format, t.dtype)` 而非硬编码
- DMA 随路转换意味着 HBM 格式和 L1 格式可以不同，不需要显式转换节点
