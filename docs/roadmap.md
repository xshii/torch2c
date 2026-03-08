# 开发路线图

> 基于 MVP 现状分析，提出可扩展性、可测性、可维护性的改进方向。
> 按优先级分三期，每期内按价值排序。

## 当前状态 (MVP 已完成)

- 2 层 Encoder Transformer，3-head MHA，固定 shape
- 9 Pass 管线端到端跑通，精度比对通过
- 19 个直接映射 + 2 个裂解规则
- 26 个 Python 测试文件 (~7200 行测试代码)
- C mock 覆盖全部算子 + ctest

---

## 第一期：基础设施加固（提升定位效率与开发信心）

### 1.1 Graph IR 快照与 diff

**问题**：Pass 间的 IR 变化不可观测，出问题需手动 dump 对比。

**方案**：
- 在 `pipeline.py` 每个 Pass 前后自动 snapshot（`graph.to_dict()`），存入 `output/debug/`
- 提供 `graph_diff(before, after)` 工具：输出 新增/删除/修改 的 node/tensor
- debug 模式下默认启用，production 模式关闭

**影响范围**：`common/graph_ir.py` + `integration/pipeline.py`

### 1.2 conftest.py + 共享 fixtures

**问题**：26 个测试文件各自构造测试图，重复代码多，修改 IR 字段需改多处。

**方案**：
- 顶层 `torch2c/conftest.py`：提供 `sample_graph`, `hw_config`, `all_configs` fixtures
- 利用已有的 `common/testing.py` 的 `make_linear_chain()` 封装为 fixture
- 各模块的 `_make_*_graph()` 逐步迁移为参数化 fixture

**影响范围**：测试代码，不影响生产代码

### 1.3 配置 schema 校验

**问题**：`load_config()` 仅验证 required_keys 存在，不验证值类型和结构。错误配置在运行时才暴露。

**方案**：
- 为每个 YAML 配置定义 JSON Schema 或 pydantic model
- `load_config()` 加载后自动校验
- 错误信息精确到字段路径

**影响范围**：`common/config_loader.py` + 各 config/ 目录

### 1.4 测试覆盖率报告

**问题**：无法量化哪些代码路径缺少测试。

**方案**：
- `pyproject.toml` 添加 `pytest-cov` 配置
- CI 中生成覆盖率报告，设置最低阈值 (建议 80%)
- 重点补充：error handling 路径、边界条件

**影响范围**：`pyproject.toml` + CI 配置

---

## 第二期：可扩展性提升（支撑更多算子和模型）

### 2.1 Pass 注册机制

**问题**：新增 Pass 需修改 `pipeline.py` 的 `_MIDDLE_PASSES` 硬编码列表。

**方案**：
```python
# 方案 A：装饰器注册
@register_pass(order=2, config_key="mapping")
def op_mapping_pass(graph, config): ...

# 方案 B：YAML 配置 Pass 顺序（更灵活）
# pipeline_config.yaml
passes:
  - {name: op_mapping, order: 2, module: torch2c.op_mapping}
  - {name: op_decomposition, order: 3, module: torch2c.op_decomposition}
```

**注意**：当前 `_PassDesc` 声明式设计已经不错，短期内不急需改。当 Pass 数量超过 15 个时考虑。

### 2.2 算子注册表

**问题**：新增算子需同时修改 `direct_mappings.yaml` + `c_api_signatures.yaml` + C mock，容易遗漏。

**方案**：
- 建立算子注册表（single source of truth），一个算子的所有信息集中定义
- 提供 `validate_op_registry` 脚本：检查映射表、签名表、mock 实现三者一致性
- 现有 `test_config_consistency.py` 扩展为自动化检查

**影响范围**：新增脚本 + 测试

### 2.3 参数化裂解规则

**问题**：当前裂解规则中 "中间 tensor shape = 第一个输入 shape" 是硬编码假设，不适用于所有算子。

**方案**：
- 裂解规则中允许指定 shape 推导函数
- 提供常见 shape 推导模板：`same_as_input_0`, `reduce_last_dim`, `broadcast_shape`
- 默认仍为 `same_as_input_0`

**影响范围**：`op_decomposition.py` + `decompositions.yaml`

### 2.4 多模型支持

**问题**：当前仅支持固定 shape 的 2 层 Encoder。

**方案**：
- 支持动态 layer 数量（已基本支持）
- 支持 Decoder（causal mask、cross-attention）
- 支持 MLP-only 模型
- 每种模型模板作为 `integration/demo/models/` 中的 benchmark

**影响范围**：demo + 测试，核心管线理论上无需改

---

## 第三期：工程成熟度（CI/CD、性能、文档）

### 3.1 CI/CD 管线

**问题**：当前无自动化 CI，依赖手动运行测试。

**方案**：
```yaml
# .github/workflows/ci.yml
jobs:
  python-test:
    - pytest torch2c/ --cov --cov-report=xml
    - pytest torch2c/integration/tests/ -v
  c-test:
    - cd npu_cpu_mock && cmake -B build && cmake --build build
    - cd npu_cpu_mock/build && ctest -V
  system-test:
    - pytest torch2c/integration/demo/demo_st/ -v
```

### 3.2 性能基线

**问题**：无法感知编译时间回退。

**方案**：
- 记录每个 Pass 的执行时间（在 `_run_middle_passes` 中 instrument）
- 记录内存使用峰值
- 建立 benchmark 模型的性能基线
- CI 中对比回退预警

### 3.3 Graph IR 不可变性

**问题**：当前 Graph/Node/Tensor 均为 mutable dataclass，任何 Pass 都可以修改任何字段，违反字段所有权表难以检测。

**方案**：
- **短期**：在 `_run_post_validation` 中添加字段所有权检查（对比 snapshot）
- **长期**：将 IR 改为 immutable + builder 模式（每个 Pass 返回新 Graph）

**评估**：长期方案改动大、收益不确定，建议先做短期方案。

### 3.4 错误恢复与诊断增强

**问题**：`DiagnosticCollector` 仅收集字符串，缺少结构化信息。

**方案**：
- `CompileDiagnostic` 增加 `node_id`, `tensor_id`, `suggestion` 字段
- 诊断信息可序列化输出为 JSON
- 常见错误附带修复建议

### 3.5 插件式 C 后端

**问题**：codegen 直接生成 CPU mock 代码，未来需支持真实 NPU 后端。

**方案**：
- 抽象 `Backend` 接口：`emit_op_call()`, `emit_dma()`, `emit_header()`
- 当前 mock 实现为 `MockBackend`
- 真实 NPU 实现为 `NpuBackend`
- 通过配置切换

---

## 优先级总结

| 优先级 | 项目 | 价值 | 工作量 |
|--------|------|------|--------|
| **P0** | Graph IR 快照与 diff | 大幅提升问题定位效率 | 小 |
| **P0** | conftest.py + 共享 fixtures | 减少测试维护成本 | 小 |
| **P1** | 配置 schema 校验 | 提前发现配置错误 | 中 |
| **P1** | 算子注册表一致性检查 | 防止遗漏 | 小 |
| **P1** | 测试覆盖率报告 | 量化质量 | 小 |
| **P2** | CI/CD 管线 | 自动化验证 | 中 |
| **P2** | 参数化裂解规则 | 支撑新算子 | 中 |
| **P2** | 性能基线 | 防止回退 | 中 |
| **P3** | Pass 注册机制 | Pass > 15 个时需要 | 中 |
| **P3** | Graph IR 不可变性 | 长期架构健康 | 大 |
| **P3** | 插件式 C 后端 | 支撑真实硬件 | 大 |

---

## 技术债清单

| 编号 | 描述 | 所在文件 | 风险 |
|------|------|---------|------|
| TD-1 | 模块局部 config/ 是 integration/config/ 的手动副本，易不一致 | 各模块 config/ | 中 |
| TD-2 | `load_config()` 无 schema 校验，仅检查 key 存在 | `common/config_loader.py` | 中 |
| TD-3 | 裂解中间 tensor shape 硬编码为输入 shape | `op_decomposition.py` | 低 (MVP) |
| TD-4 | 无 conftest.py，测试间无共享 fixture | 各 tests/ | 低 |
| TD-5 | `absorptions.yaml` 在 integration 中为空（mock 不支持融合） | `integration/config/` | 低 |
| TD-6 | codegen 与 mock 后端耦合，无抽象层 | `codegen/` | 低 (MVP) |
| TD-7 | 错误诊断无结构化信息（仅字符串） | `common/errors.py` | 低 |
