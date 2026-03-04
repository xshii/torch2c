# Clean Code / Clean Architecture 重构计划

## Context

前一轮 4 项整改已完成（WI-1~4），全量 133+6 测试通过。本轮针对架构层面的 10 项 Clean Code / Clean Architecture 优化，按优先级分 4 阶段执行，每阶段结束跑全量测试保证绿色。

---

## Phase 1 (P0): 基础改进 — 无交叉依赖，可并行

### P0-1: 错误处理统一

**问题**：5 种不同错误策略（raise/warning/静默）混用。
**方案**：`common/errors.py` 新增 `Severity` 枚举、`CompileDiagnostic`（frozen dataclass）、`DiagnosticCollector`（warn/error/has_errors/summary）。pipeline.py 内部创建 collector，每个 Pass 后收集诊断，统一日志输出。保留现有异常类。

| 文件 | 操作 |
|------|------|
| `common/errors.py` | 新增 3 个类型 (~35 行) |
| `common/__init__.py` | 导出新类型 |
| `integration/pipeline.py` | 创建 collector，替换 validate_phase 循环为 `_run_post_validation()` |
| **新建** `common/tests/test_errors.py` | 3 个用例 |

### P0-2: 拆分 `_handle_call`

**问题**：`graph_capture.py` `_handle_call` ~80 行做 5 件事。
**方案**：提取 3 个子函数，`_handle_call` 降至 ~15 行编排器。

| 新函数 | 职责 | 行数 |
|--------|------|------|
| `_parse_call_args()` | 解析 FX args/kwargs → op, nid, input_tids, params | ~35 |
| `_normalize_op_inputs()` | addmm 重排 + 参数重命名 | ~12 |
| `_create_call_outputs()` | 创建输出 tensor + 更新 fx_map | ~20 |
| `_handle_call()` | 调用上述 3 个 + 更新 consumer + add_node | ~15 |

文件：`graph_capture/graph_capture.py`，无测试变更（12 个现有用例通过 `capture()` 覆盖）。

---

## Phase 2 (P1): 架构改进 — 依赖 Phase 1

### P1-1: validate_phase 解耦 — 核心层不再知道 pipeline 阶段

**问题**：`graph_ir.py` 包含 `_PHASE_VALIDATORS` 和 5 个阶段校验函数，核心实体反向依赖管线知识（违反 CA 依赖规则）。
**方案**：每个 Pass 模块导出 `post_validate(graph) -> list[str]`，pipeline 直接调用。graph_ir.py 只保留结构性 `validate()`。

| 文件 | 操作 |
|------|------|
| `common/graph_ir.py` | **删除** `validate_phase()`、5 个 `_validate_*`、`_PHASE_VALIDATORS` |
| `graph_capture/graph_capture.py` | 新增 `post_validate()` (迁移自 `_validate_graph_capture`) |
| `op_mapping/op_mapping.py` | 新增 `post_validate()` |
| `op_decomposition/op_decomposition.py` | 新增 `post_validate()` |
| `format_annotator/format_annotator.py` | 新增 `post_validate()` |
| `memory_planner/memory_planner.py` | 新增 `post_validate()` |
| `integration/pipeline.py` | 替换 `graph.validate_phase("xxx")` 为 `module.post_validate(graph)` |
| `common/tests/test_graph_ir.py` | **删除** `TestValidatePhase` 类 |
| 各 Pass test 文件 (5个) | 迁入对应的 post_validate 测试用例 |

### P1-2: 裂解规则声明化

**问题**：`op_decomposition.py` 硬编码 `if step["npu_op"] == "npu_layernorm_part2"`。
**方案**：YAML 新增 `extra_inputs` 字段（如 `["original.0"]`），Python 通用解析。

| 文件 | 操作 |
|------|------|
| `integration/config/decompositions.yaml` | layernorm_part2 加 `extra_inputs: ["original.0"]` |
| `op_decomposition/config/decompositions.yaml` | 同上 |
| `op_decomposition/op_decomposition.py` | 硬编码 if → 通用 extra_inputs 解析 (~6 行) |

### P1-3: 提取 SourceResolver 类

**问题**：`c_emitter.py` 的 3 个函数（`_resolve_param`/`_find_tensor`/`_extract_tensor_field`）耦合紧密。
**方案**：封装为 `SourceResolver` 类，方法：`resolve(param)`、`find_tensor(key)`、`_resolve_tensor_ref()`、`_resolve_param_ref()`、`_extract_field()`。`_gen_op_call` 改用 resolver。

| 文件 | 操作 |
|------|------|
| `codegen/c_emitter.py` | 3 个函数 → `SourceResolver` 类 (~50 行) |

---

## Phase 3 (P2): 结构优化 — 依赖 Phase 2

### P2-1: Pass Protocol 定义

**新建** `common/pass_protocol.py`：

```python
@runtime_checkable
class CompilerPass(Protocol):
    def run(self, graph: Graph, config: dict) -> Graph: ...
```

文档性质。memory_planner（返回 tuple）和 scheduler（config 可选）不强制适配。

| 文件 | 操作 |
|------|------|
| **新建** `common/pass_protocol.py` | ~20 行 |
| `common/__init__.py` | 导出 `CompilerPass` |

### P2-2: Graph 字段归属文档

`graph_ir.py` 模块 docstring 新增 **Field Ownership Table**：明确每个字段由哪个 Pass 写入、谁可以读。纯文档，无代码变更。

### P2-3: 统一 DTYPE 元信息

**问题**：`DTYPE_BYTES`（memory_planner）和 `DTYPE_MAP`（codegen）分别维护。
**方案**：**新建** `common/dtypes.py`。

| 文件 | 操作 |
|------|------|
| **新建** `common/dtypes.py` | `DtypeInfo(bytes, c_enum)` + `DTYPE_INFO` + `dtype_bytes()` + `dtype_c_enum()` (~35 行) |
| `memory_planner/memory_planner.py` | 删除 `DTYPE_BYTES`，改用 `dtype_bytes()` |
| `codegen/_helpers.py` | `DTYPE_MAP` 改为从 `DTYPE_INFO` 派生 |
| `common/__init__.py` | 导出新类型 |
| **新建** `common/tests/test_dtypes.py` | 3 个用例 |

---

## Phase 4 (P3): 管线优化

### P3-1: 声明式管线

**方案**：定义 `_PassDesc(name, number, run_fn, config_key, validate_fn)` 列表，`compile()` 中 7 个中间 Pass 改为循环。graph_capture (Pass 1) 和 codegen (Pass 9) 保持特殊处理。~50 行重复 → ~20 行循环。

| 文件 | 操作 |
|------|------|
| `integration/pipeline.py` | 新增 `_PassDesc` + `_MIDDLE_PASSES` 列表，compile() 重写中间段 |

### P3-2: 配置收敛 (仅文档)

每个模块本地 `config/` 的 YAML 头部加注释：`# TEST FIXTURE — 生产配置在 integration/config/`。

### P3-3: DRY 配置加载 (不动)

各模块 `load_xxx_config()` 仅供 demo 脚本使用，保留不改。

---

## 新增文件汇总

| 文件 | 用途 | 行数 |
|------|------|------|
| `common/dtypes.py` | 统一 dtype 元信息 | ~35 |
| `common/pass_protocol.py` | CompilerPass Protocol | ~20 |
| `common/tests/test_errors.py` | DiagnosticCollector 测试 | ~25 |
| `common/tests/test_dtypes.py` | Dtype 元信息测试 | ~15 |

## 提交策略

```
Commit 1: P0-1 + P0-2       (错误处理 + 拆分函数)
    ↓ pytest 全绿
Commit 2: P1-1               (validate_phase 解耦 — 改动最多，独立提交)
    ↓ pytest 全绿
Commit 3: P1-2 + P1-3 + P2-2 + P2-3  (声明裂解 + SourceResolver + 文档 + dtypes)
    ↓ pytest 全绿
Commit 4: P2-1               (Pass Protocol)
    ↓ pytest 全绿
Commit 5: P3-1 + P3-2        (声明式管线 + 配置注释)
    ↓ pytest 全绿 (最终验证：133+ Python + 6 C)
```

## 最终验证

```bash
cd npu_cpu_mock/build && cmake .. && make && ctest --output-on-failure
.venv/bin/python3 -m pytest --tb=short -q
```
