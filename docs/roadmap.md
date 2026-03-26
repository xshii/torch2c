# 开发路线图

## 当前状态

- 目录重构完成：a_capture / b_lowering / c_backend / d_emission / optpass
- 17 个 pass（9 必需 + 8 可选），441 个测试用例
- op_mapping/op_decomposition 解耦：mapping 纯命名，decomposition 按 npu_op 查裂解表
- 4 种 tensor 格式：ND / NZ / ZZ / NN
- opt_log 优化决策日志系统
- Graph.renumber() 节点重编号
- Pass 耗时统计
- Pipeline 可视化（HTML 流程图 + 4 列甬道图 + DMA 节点）
- Demo 模型：AX+B / Embedding / 单层 Attention / 2 层 Encoder

---

## 已完成 ✅

| 项目 | 状态 |
|------|------|
| Graph IR 快照与 diff | ✅ debug_dump 模式 |
| Pass 耗时统计 | ✅ _pass_timing |
| Pipeline 可视化 | ✅ pipeline_viz + pass_detail_viz |
| 目录重构（4 阶段 + optpass） | ✅ |
| op_mapping/decomposition 解耦 | ✅ npu_op 查裂解表 |
| transpose 移到 IDMA | ✅ |
| opt_log 优化决策记录 | ✅ 13 个 pass |
| Graph.renumber() | ✅ |
| 4 种格式 (ND/NZ/ZZ/NN) | ✅ 全链路 |
| MHA merge pass | ✅ |
| npu_mock.h 消除 | ✅ |
| format_planner 实际标注 (NZ/ZZ) | ✅ tiebreaker + 权重格式优化 |
| DMA 随路格式转换 | ✅ src_format/dst_format 全链路 |
| global_tiler vector 算子支持 | ✅ 20+ 算子 |

---

## TODO

### P0 — 功能补全

| 项目 | 描述 |
|------|------|
| decompositions.yaml 实效性 | layernorm/softmax 裂解后下游 pass 需适配（c_api_signatures 缺 part1/part2） |

### P1 — 可扩展性

| 项目 | 描述 |
|------|------|
| Pass 自动注册 | 新增 pass 无需改 pipeline.py，从 _PassDesc 自动发现 |
| 算子注册表一致性 | 一个算子的映射/签名/mock/tiling/naming 集中定义 |
| 配置 schema 校验 | load_config 加载后自动验证值类型和结构 |
| 参数化裂解规则 | 支持自定义 shape 推导（不只是 same_as_input_0） |

### P2 — 工程质量

| 项目 | 描述 |
|------|------|
| conftest.py 共享 fixtures | 减少测试重复代码 |
| 测试覆盖率报告 | pytest-cov + 最低阈值 80% |
| CI/CD | GitHub Actions: pytest + ctest + ST |
| 性能基线 | 编译时间回退预警 |

### P3 — 架构演进

| 项目 | 描述 |
|------|------|
| Graph IR 不可变性 | immutable + builder 模式 |
| 插件式 C 后端 | 抽象 Backend 接口，支持真实 NPU |
| 动态 shape | 编译期 shape 符号化 |
| 多核调度 | task_id → 物理核映射 |

---

## 技术债

| 编号 | 描述 | 风险 |
|------|------|------|
| TD-1 | 模块局部 config/ 是 integration/config/ 的手动副本 | 中 |
| TD-2 | load_config() 无 schema 校验 | 中 |
| ~~TD-3~~ | ~~format_planner 标注逻辑未完整实现~~ | ✅ 已修复 |
| ~~TD-4~~ | ~~codegen DMA 未使用 tensor.format 做随路转换~~ | ✅ 已修复 |
| TD-5 | c_api_signatures 缺少 layernorm_part1/part2 等裂解产物的签名 | 中 |
| TD-6 | codegen 与 mock 后端耦合，无抽象层 | 低 |
