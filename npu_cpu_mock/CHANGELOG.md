# Changelog

## v2.0.0 — API 重构

### Breaking Changes

- **所有 op 第一个参数统一为 `TidInfo tid`**，携带 task_id / dep_task_id / dep_unit，用于流水线调度。
- **删除 legacy API**：`npu_dma_load`、`npu_dma_store`、`npu_dma_barrier`、`npu_set_dependency`、`npu_barrier` 全部移除，由新的 tensor API 取代。
- **删除 `npu_sync.c`**，同步语义已融入 `TidInfo`。
- **DMA 接口统一为 `npu_tensor_t` 参数**，不再使用裸指针 + dtype 分离传参。

### New Features

- **`TidInfo` 结构体**：`{ task_id, dep_task_id, dep_unit }`，每个 op 统一携带调度信息（CPU mock 上 no-op）。
- **`npu_compute_unit_t` 枚举**：`NPU_UNIT_CUBE` / `NPU_UNIT_VECTOR` / `NPU_UNIT_SCALAR`。
- **`compute_dtype` 精度控制**：所有计算 op 真正使用 `compute_dtype` 参数。通过 `npu_read_compute()` / `npu_write_compute()` inline helper 在读写时截断到计算精度，模拟低精度计算单元行为。
- **`dma_move(tid, dst, src, count)`**：通用 DMA 搬运，src/dst dtype 不同时自动随路类型转换，dtype 相同时走 memcpy 快速路径。
- **`idma_move(tid, dst, src, count)`**：L1 → pipe 搬运，与 `dma_move` 功能相同，语义上表示内部 DMA。
- **`dma_reformat(tid, input, out, count)`**：格式/dtype 转换。

### Bug Fixes

- **修复 `dma_reformat` 参数传反**：`tensor_move(input, out)` 修正为 `tensor_move(out, input)`。
- **修复 `cube_matmul_bias` 精度回退**：提取 `matmul_core()` 返回 float 累加器数组，bias 在 float 精度上直接相加，避免中间经过 `out.dtype` 截断。

### Refactoring

- **`npu_l1_base` 从 `npu_sync.c` 移到 `npu_dtype_utils.c`**，语义更合理。
- **`cube_matmul` / `cube_matmul_bias` 共享 `matmul_core()`**，消除矩阵乘内循环重复。
- **`vector_transpose` 索引计算去重**，同 dtype / 异 dtype 分支合并为一个循环。
- **`scalar_reshape` / `scalar_broadcast` / `scalar_copy` 共享 `scalar_memcpy()`**。
