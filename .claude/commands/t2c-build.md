# /t2c-build — 智能开发引擎

> 合并自: iterate + tdd + spec-test + debug
> 用法: `/t2c-build [mode] [target]`
> 模式: `fix` | `feat` | `refactor` | `perf` | `test` | `debug`
> 示例: `/t2c-build fix block_pad对齐bug` | `/t2c-build feat 新增softmax融合` | `/t2c-build test roofline`

你是 torch2c NPU 编译器的开发引擎。根据用户的 mode 和 target，自动执行完整的开发循环。
$ARGUMENTS 包含 mode 和 target 信息。若为空，询问用户要做什么。

---

## 0. 模式路由

根据 $ARGUMENTS 的第一个词选择模式：

| 关键词 | 模式 | 核心循环 |
|--------|------|----------|
| `fix` / `bug` / `修` | BUG FIX | 复现 → 定位 → 修复 → 回归 |
| `feat` / `add` / `加` / `新` | FEATURE | 设计确认 → TDD → 集成 → 回归 |
| `refactor` / `重构` | REFACTOR | 确保测试覆盖 → 小步重构 → 回归 |
| `perf` / `优化` / `性能` | PERFORMANCE | 量化现状 → 找瓶颈 → 优化 → 验证 |
| `test` / `测试` | TEST DESIGN | 规格提取 → 用例设计 → 实现 → 覆盖率检查 |
| `debug` / `排查` | DEBUG | 症状分类 → 缩小范围 → 逐层定位 |
| 其他 | AUTO | 分析意图后自动选择模式 |

---

## 1. 通用入口（所有模式共享）

### 1.1 上下文收集

```bash
# 当前状态
git status
git log --oneline -5
.venv/bin/pytest --tb=short -q 2>&1 | tail -5  # 快速确认基线

# 如果 target 指定了模块
grep -rn "target关键词" torch2c/ --include="*.py" | head -20
```

### 1.2 影响分析

改动前，明确回答：
- **改什么**：哪些文件、哪些函数
- **为什么**：动机和预期效果
- **风险**：可能破坏什么（列出相关的下游 pass 和测试）
- **验证**：怎么确认改对了

**方案决策需要用户同意**，代码实现自行执行。

---

## 2. BUG FIX 模式

### 2.1 复现（Red）

```python
# 写一个必然失败的测试来精确复现 bug
def test_regression_xxx():
    """回归：描述 bug 现象。"""
    g = _make_graph(...)
    result = run(g, config)
    assert result == expected  # 当前会 FAIL
```

```bash
.venv/bin/pytest path/to/test.py::test_regression_xxx -v
# 确认 FAILED
```

### 2.2 定位

按以下优先级缩小范围：

1. **错误类型分类**（先看报错信息）：

| 错误类型 | 排查方向 | 关键文件 |
|----------|----------|----------|
| `MappingError` | 映射表缺算子 | `direct_mappings.yaml` |
| `DecompositionError` | 裂解规则缺失 | `decompositions.yaml` |
| `ValidationError` | format/dtype 不匹配 | `format_annotator` + `c_api_signatures.yaml` |
| `CodegenError` | 签名参数不匹配 | `c_api_signatures.yaml` + `c_emitter.py` |
| `MemoryPlanError` | L1 溢出 | `hardware_config.yaml` memory.l1 |
| C 编译失败 | mock 实现 bug | `npu_cpu_mock/src/` |
| Golden FAIL | FP16 精度 / mock 逻辑 | 逐 pass 对比中间结果 |
| Config 一致性 | 某个表漏了算子 | 按报错补全对应 YAML |

2. **逐 pass 缩小**：`debug_dump=True` 编译，对比每个 pass 后的 JSON 快照
3. **日志提升**：`NPU_LOG_LEVEL=DEBUG .venv/bin/pytest ... -v -s`

### 2.3 修复（Green）

- 最小修复，不顺手重构
- 确认复现测试通过

### 2.4 回归

```bash
.venv/bin/pytest --tb=short -q
```

---

## 3. FEATURE 模式

### 3.1 设计确认

用 1-3 句话描述方案，等待用户确认：
- 改哪些文件
- 新增什么接口
- 是否需要新配置

### 3.2 TDD 循环

**严格 Red → Green → Refactor**：

```
对每个功能点:
  ① 写测试（明确断言，不是 assert result）
  ② 跑测试，确认 Red（FAILED）
  ③ 写最小实现，确认 Green（PASSED）
  ④ 重构（在测试保护下）
  ⑤ 跑模块测试确认无回归
```

### 3.3 测试设计方法（从规格出发）

**不要从实现出发写测试**。从规格提取可测试属性：

| 问题 | 产出的测试 |
|------|------------|
| 正常情况下结果是什么？ | `test_{property}_normal` |
| 输入非法时应该怎样？ | `test_{property}_rejects_invalid` |
| 边界值是什么？ | `test_{property}_boundary` |
| 空/零/极端情况？ | `test_{property}_edge_case` |
| 和其他规格有交叉吗？ | `test_{property}_with_{other}` |

对于**配置驱动的逻辑**，用决策表参数化：

```python
CASES = [
    # (input, expected)
    ("nd", "fp16", (1, 16)),
    ("nz", "int8", (32, 16)),
]

@pytest.mark.parametrize("fmt,dtype,expected", CASES,
                         ids=[f"{c[0]}_{c[1]}" for c in CASES])
def test_alignment(fmt, dtype, expected):
    assert get_dim_align(fmt, dtype) == expected
```

对于**pass 逻辑**，必须包含：
- `test_basic` — 正常变换
- `test_no_op` — 不满足条件时不变换
- `test_idempotent` — 跑两次结果相同
- `test_post_validate_clean` — post_validate 返回空

### 3.4 编码规范

- 每个函数 < 50 行，模块核心代码 < 300 行
- 用 `common.get_logger`（不用 print）、`common.errors`（不用裸 raise）
- 用 `opt_log` 记录每个优化决策
- 改了配置就跑 `pytest torch2c/integration/tests/test_config_consistency.py`

### 3.5 集成验证

```bash
# 模块测试
.venv/bin/pytest torch2c/{module}/tests/ -v --tb=short

# 全量回归（510 用例）
.venv/bin/pytest --tb=short -q
```

### 3.6 自查清单

- [ ] 全量测试通过
- [ ] 每个函数 < 50 行，模块 < 300 行
- [ ] 用了 common 的 logger/errors
- [ ] opt_log 记录了优化决策
- [ ] 改了 config → config 一致性测试通过
- [ ] 改了 C mock → golden 比对通过

---

## 4. REFACTOR 模式

1. 确保重构区域有**充分测试覆盖**（没有就先补）
2. 小步重构，每步都跑测试
3. **不改行为，只改结构**
4. 保持 API 兼容（或一次性更新所有调用方）

---

## 5. PERFORMANCE 模式

1. **量化现状**：`pass_timing.json`、内存使用、C mock 运行时间
2. **找瓶颈**：profile，不猜
3. **优化 + 验证效果**：前后对比
4. **确认正确性未退化**：全量测试

---

## 6. TEST DESIGN 模式

### 五种测试方法

| 方法 | 适用场景 | 核心思路 |
|------|----------|----------|
| **规格驱动** | 有明确的硬件/协议规格 | 每条规格 → 一组 assert |
| **场景驱动** | 端到端集成 | 模型 × 硬件配置 × pass 组合 矩阵 |
| **不变量** | 全局属性 | 任何 pass 后都必须满足的属性 |
| **决策表** | 配置驱动逻辑 | parametrize 穷举输入组合 |
| **回归** | Bug fix | 每个 fix 附带复现测试 |

### 场景矩阵模板

```
场景 = 模型复杂度 × 内存压力 × 精度模式
```

| 编号 | 模型 | L1 | 预期策略 | 验证重点 |
|------|------|----|----------|----------|
| ST1 | Linear (AX+B) | 充裕 | bulk | 最简编译链 |
| ST3 | Linear (AX+B) | 紧张 | tiled | tiling 正确性 |
| ST5 | 2-layer MLP | 中等 | perop | L1 liveness 复用 |
| ST6 | MHA (4 头) | 紧张 | tiled | 注意力 tiling |

### 不变量检查（任何 pass 后）

```python
def assert_graph_invariants(graph):
    # 每个 tensor 的 producer/consumer 引用有效
    # execution_order 中的节点都存在
    # 每个节点的输入输出 tensor 都存在
    # 无孤立 tensor
```

### 命名规范

```
test_{功能}_{场景}
test_pad_shape_nd_fp16          # 功能 + 具体场景
test_skip_when_already_aligned  # 边界条件
```

---

## 7. DEBUG 模式

### 快速分类（根据症状）

收到错误信息后，按上面 2.2 的错误分类表定位方向。

### 三级缩小范围

1. **pass 级**：`debug_dump=True` 编译，对比前后 JSON 快照
2. **节点级**：查看 `node.params["_opt_log"]` 中的优化决策记录
3. **tensor 级**：`NPU_LOG_LEVEL=DEBUG` 查看每个 tensor 的 shape/format 变化

### 调试工具速查

| 工具 | 命令 |
|------|------|
| 全量测试 | `.venv/bin/pytest --tb=short -q` |
| 单文件测试 | `.venv/bin/pytest ${file} -v -s` |
| 详细日志 | `NPU_LOG_LEVEL=DEBUG .venv/bin/pytest ... -v -s` |
| 单个测试 | `.venv/bin/pytest path::Class::method -v -s` |
| 停在首个失败 | `.venv/bin/pytest -x --tb=long` |
| 跳 golden | `--deselect torch2c/integration/tests/test_pipeline.py::TestCGoldenComparison` |
| debug 编译 | `python scripts/compile_and_viz.py` (产出 debug/ 快照 + viz/ 可视化) |
| C mock 调试 | 编译时加 `-DNPU_DEBUG_LEVEL=2` |

### 常见坑

```python
# tensor 访问：用 graph.tensors[tid]，不要缓存引用后修改
# C mock：用 .ptr 不用 .addr，用 npu_t_ptr() / npu_read_compute() / npu_write_store()
# padded size：用 get_dim_align(t.format, t.dtype)，不要硬编码 (16, 16)
```

---

## 8. 通用结束流程

所有模式完成后：

1. **全量回归**：`.venv/bin/pytest --tb=short -q`
2. **如改了 config**：`pytest torch2c/integration/tests/test_config_consistency.py`
3. **报告**：列出改了什么文件、测试结果、关键决策

提交规范（用户要求时）：
```
类型: feat / fix / refactor / test / docs / chore / perf
消息: 简短描述（中英均可）
```
