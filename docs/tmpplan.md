# Clean Code / Clean Architecture 重构计划 (Round 3)

## Context

Round 2 的 10 项 (P0~P3) + ruff/mypy 部署 + 魔鬼数字消除已全部完成，104 Python 测试通过。
本轮聚焦 **函数/模块过长**、**校验覆盖缺口**、**协议落地**、**硬编码** 四类遗留问题。

---

## Phase 1 (R3-1): 超长函数拆分 — 违反 < 50 行规则

### R3-1a: 拆分 `_decompose_node()` (100 行)

**文件**：`op_decomposition/op_decomposition.py:50`
**问题**：100 行，做 7 件事（创建中间 tensor、构建新节点 inputs/outputs、更新 consumer、更新 execution_order、替换节点）。
**方案**：提取 3 个子函数。

| 新函数 | 职责 | 行数 |
|--------|------|------|
| `_create_intermediates()` | 创建中间 tensor | ~10 |
| `_build_step_nodes()` | 按 steps 构建新节点列表（含 extra_inputs 解析） | ~35 |
| `_rewire_graph()` | 更新 consumer/producer/execution_order、替换节点 | ~40 |
| `_decompose_node()` | 编排器 | ~15 |

### R3-1b: 拆分 `compile()` (90 行)

**文件**：`integration/pipeline.py:160`
**问题**：90 行，串联 9 个 Pass + golden 导出 + codegen 子模块。
**方案**：提取 codegen 阶段为 `_run_codegen()`。

| 新函数 | 职责 | 行数 |
|--------|------|------|
| `_run_codegen()` | Pass ⑨ 全部子 emitter + weight + golden | ~35 |
| `compile()` | 降至 ~55 行 |

### R3-1c: 拆分 `emit_main_c()` (74 行)

**文件**：`codegen/c_project/main_emitter.py:22`
**问题**：74 行，混合 tensor 过滤 + load/compare 代码生成。
**方案**：提取 2 个子函数。

| 新函数 | 职责 | 行数 |
|--------|------|------|
| `_gen_load_inputs()` | 生成 load_inputs C 代码块 | ~15 |
| `_gen_compare_outputs()` | 生成 compare_outputs C 代码块 | ~25 |
| `emit_main_c()` | 降至 ~30 行 |

### R3-1d: 拆分 `_try_absorb()` (62 行)

**文件**：`op_absorption/op_absorption.py:28`
**问题**：62 行，匹配 + 重连 + 清理混在一起。
**方案**：提取内循环体为 `_absorb_one()`。

| 新函数 | 职责 | 行数 |
|--------|------|------|
| `_absorb_one()` | 执行单次吸收（重连 consumer、清理旧 tensor） | ~30 |
| `_try_absorb()` | 遍历 + 匹配 + 调用 `_absorb_one()` | ~25 |

### R3-1e: 拆分 `_allocate_hbm()` (56 行)

**文件**：`memory_planner/memory_planner.py:116`
**问题**：56 行，释放过期块和 best-fit 分配混在同一循环体。
**方案**：提取 `_release_expired()` 和 `_best_fit_alloc()`。

| 新函数 | 职责 | 行数 |
|--------|------|------|
| `_release_expired()` | 释放 lifetime 结束的 tensor 到 free_blocks | ~15 |
| `_best_fit_alloc()` | 从 free_blocks 找最优空闲块 | ~15 |
| `_allocate_hbm()` | 降至 ~30 行 |

---

## Phase 2 (R3-2): 超长模块拆分 — 违反 < 300 行规则

### R3-2a: memory_planner.py 拆分 (448 行 → 2 文件)

**文件**：`memory_planner/memory_planner.py`（448 行，超过 300 行限制）
**方案**：将 DMA 相关逻辑提取到 `memory_planner/_dma.py`。

| 文件 | 内容 | 行数 |
|------|------|------|
| `memory_planner.py` | HBM 分配 + L1 布局 + `run()` + `post_validate()` | ~250 |
| `_dma.py` | `DmaInstruction`、`DmaPlan`、`_get_dst_format`、`_build_dma_plan`、`_build_bulk_dma` | ~150 |
| `__init__.py` | 补充导出 `DmaPlan`, `DmaInstruction` |

### R3-2b: graph_capture.py 精简 (339 行 → < 300)

**文件**：`graph_capture/graph_capture.py`（339 行）
**问题**：`_PARAM_RENAMES` + `_DTYPE_MAP` + `_DIM_TO_SIZE_OPS` 等配置数据占 ~40 行，与逻辑代码混合。
**方案**：将映射表移入 `graph_capture/_constants.py`（~40 行），主文件降至 ~300 行。

---

## Phase 3 (R3-3): 校验覆盖缺口

### R3-3a: 补 op_absorption post_validate

**问题**：`_MIDDLE_PASSES` 中 op_absorption 的 `validate_fn=None`，是唯一没有 post_validate 的中间 Pass。
**方案**：新增 `op_absorption.post_validate(graph)`，校验被吸收节点已清理、absorbed_inputs 引用的 tensor 存在。

| 文件 | 操作 |
|------|------|
| `op_absorption/op_absorption.py` | 新增 `post_validate()` (~10 行) |
| `integration/pipeline.py` | `_MIDDLE_PASSES` 中 op_absorption 填入 `validate_fn` |
| `op_absorption/tests/test_op_absorption.py` | 新增 2 个用例 |

### R3-3b: 补 scheduler post_validate

**问题**：scheduler (Pass ⑧) 无 `post_validate`，pipeline 也没调 `_run_post_validation`。
**方案**：新增 `scheduler.post_validate(graph)`，校验所有节点有 `schedule_order` 且 `dependencies` 非 None。pipeline 中补调。

| 文件 | 操作 |
|------|------|
| `scheduler/scheduler.py` | 新增 `post_validate()` (~10 行) |
| `integration/pipeline.py` | Pass ⑧ 后补 `_run_post_validation(collector, "scheduler", graph, scheduler.post_validate)` |
| `scheduler/tests/test_scheduler.py` | 新增 2 个用例 |

### R3-3c: validator 诊断接入 DiagnosticCollector

**问题**：validator (Pass ⑥) 直接 `raise ValidationError`，跳过 DiagnosticCollector 收集。
**方案**：pipeline 中用 try/except 包裹，捕获后写入 collector 再 re-raise。

| 文件 | 操作 |
|------|------|
| `integration/pipeline.py` | Pass ⑥ 包裹 try/except，异常写入 collector |

---

## Phase 4 (R3-4): 硬编码 & 协议落地

### R3-4a: graph_capture 硬编码算子规则配置化

**文件**：`graph_capture/graph_capture.py:22-26, 238-241`
**问题**：`_PARAM_RENAMES` 和 `_ADDMM_REORDER` 将算子特殊逻辑硬编码在 Python 中，与其他 Pass 的 YAML 配置模式不一致。
**方案**：新建 `graph_capture/config/capture_rules.yaml`，存放 param_renames 和 input_reorder 规则，代码通用解析。

| 文件 | 操作 |
|------|------|
| **新建** `graph_capture/config/capture_rules.yaml` | param_renames + input_reorder 规则 |
| `graph_capture/graph_capture.py` | 删除硬编码，加载 YAML 解析 |
| `integration/config/capture_rules.yaml` | 同步生产配置 |

### R3-4b: pipeline `_run_golden` 硬编码 fp16

**文件**：`integration/pipeline.py:131-133`
**问题**：`astype(np.float16)` 硬编码，应从模型输入推断或参数化。
**方案**：从 graph 的 model_input tensor 的 dtype 推断转换精度。

| 文件 | 操作 |
|------|------|
| `integration/pipeline.py` | `_run_golden` 增加 `dtype` 参数，默认 "fp16"，由 `compile()` 传入 |

### R3-4c: DiagnosticCollector 错误未拦截编译

**文件**：`integration/pipeline.py:142-154, 219`
**问题**：`_run_post_validation()` 始终调 `collector.warn()`，从不调 `collector.error()`。`collector.summary()` 仅日志输出，`has_errors()` 从未检查。编译可在校验失败时照常完成。
**方案**：post_validate 返回的错误调 `collector.error()`；`collector.summary()` 后检查 `has_errors()`，若有则 raise。

| 文件 | 操作 |
|------|------|
| `integration/pipeline.py` | `_run_post_validation` 改用 `collector.error()`；`compile()` 末尾检查 `has_errors()` |

### R3-4d: 补齐 `__init__.py` 导出

**问题**：`graph_capture`、`op_mapping`、`op_decomposition` 的 `__init__.py` 为空，`scheduler` 缺少 `post_validate` 导出。pipeline 直接 `from xxx import xxx_module` 访问子模块内部。
**方案**：统一补齐 `__init__.py` 导出 `run`、`post_validate`（如有）。

| 文件 | 操作 |
|------|------|
| `graph_capture/__init__.py` | 导出 `capture`, `post_validate` |
| `op_mapping/__init__.py` | 导出 `run`, `post_validate` |
| `op_decomposition/__init__.py` | 导出 `run`, `post_validate` |
| `op_absorption/__init__.py` | 补充导出 `post_validate`（R3-3a 新增后） |
| `scheduler/__init__.py` | 补充导出 `post_validate`（R3-3b 新增后） |

### R3-4e: graph_capture 内部函数补类型标注

**文件**：`graph_capture/graph_capture.py:166,178,282,326`
**问题**：`_handle_placeholder`、`_handle_getitem`、`_handle_call`、`_handle_output` 参数无类型标注，mypy 无法检查。
**方案**：补上 `graph: Graph`、`fx_node: torch.fx.Node`、`fx_map: dict[str, str | list[str]]` 等类型。

| 文件 | 操作 |
|------|------|
| `graph_capture/graph_capture.py` | 4 个 `_handle_*` 函数补类型标注 |

### R3-4f: mypy union-attr 热点修复

**问题**：41 个 mypy 错误，集中在 `Optional[int]` 字段（hbm_offset 等）。测试代码中 `get_tensor()` 返回 `Tensor | None` 未 narrow。
**方案**：仅修复生产代码中的 2 处（`memory_planner.py:142` arg-type、`weight_exporter.py:64` tuple 类型），测试代码用 assert narrow。

| 文件 | 操作 |
|------|------|
| `memory_planner/memory_planner.py` | align_up 调用处加 `assert t.hbm_size is not None` |
| `codegen/weight_exporter.py` | entries 改用 NamedTuple 消除 `[2]` 索引 |
| 各 test 文件 | `get_tensor()` 后加 `assert t is not None` (~15 处) |

---

## 提交策略

```
Commit 1: R3-1a ~ R3-1e        (超长函数拆分)
    ↓ pytest 全绿
Commit 2: R3-2a + R3-2b        (超长模块拆分)
    ↓ pytest 全绿
Commit 3: R3-3a ~ R3-3c        (校验覆盖缺口)
    ↓ pytest 全绿
Commit 4: R3-4a ~ R3-4c        (配置化 + 硬编码 fp16 + DiagnosticCollector 加固)
    ↓ pytest 全绿
Commit 5: R3-4d ~ R3-4f        (__init__.py 导出 + 类型标注 + mypy 修复)
    ↓ pytest 全绿 + ruff check 全绿 + mypy 错误 < 10
```

## 最终验证

```bash
ruff check torch2c/
mypy torch2c/
python3 -m pytest --tb=short -q
cd npu_cpu_mock/build && cmake .. && make && ctest --output-on-failure
```
