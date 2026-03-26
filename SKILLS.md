# SKILLS.md — torch2c AI Skill Pack v2

> **3 个 all-in-one skills** 覆盖 NPU 编译器开发的全部工作流。
> 适用于 Claude Code、GLM CLI、Cursor、Windsurf 等 AI 编码工具。

---

## Skill 索引

| Skill | 命令 | 类型 | 定位 | 文件 |
|-------|------|------|------|------|
| **t2c-build** | `/t2c-build` | workflow | 日常开发引擎 | `.claude/commands/t2c-build.md` |
| **t2c-scaffold** | `/t2c-scaffold` | workflow | 扩展脚手架 | `.claude/commands/t2c-scaffold.md` |
| **t2c-arch** | `/t2c-arch` | reference | 架构参考 | `.claude/commands/t2c-arch.md` |

**触发规则**: `.claude/skill-rules.json`（关键词 + 文件模式 + intent 正则自动触发）

---

## Skill 1: t2c-build — 智能开发引擎 (PUA-Powered)

**合并自**: iterate + tdd + spec-test + debug + **PUA 质量引擎**（原 4 skill + PUA → 1 个）

**触发词**: fix / bug / feat / refactor / perf / test / debug / loop / TDD / 报错 / 调试 / 用例设计

**7 种模式**:

| 模式 | 触发词 | 核心循环 |
|------|--------|----------|
| `fix` | fix, bug, 修, 报错 | 复现(Red) → 方法论5步定位 → 修复(Green) → 冰山扫描 → 回归 |
| `feat` | feat, add, 加, 新增 | 设计确认 → TDD循环 → 集成 → 回归 |
| `refactor` | refactor, 重构 | 确保测试覆盖 → 小步重构 → 回归 |
| `perf` | perf, optimize, 性能 | 量化 → 找瓶颈 → 优化 → 验证 |
| `test` | test, TDD, spec, 用例 | 规格提取 → 5种方法设计 → 实现 |
| `debug` | debug, 调试, 排查 | 症状分类 → 方法论5步 → 定位 |
| `loop` | loop, 自动, 迭代 | 自主迭代，零人工干预，直到完成 |

**PUA 质量引擎（融合自 pua-skills）**:
- **三条红线**: 闭环验证(必须贴pytest输出) / 事实驱动(不猜) / 穷尽一切(5步走完才能放弃)
- **通用方法论 5 步**: 闻味道 → 揪头发 → 照镜子 → 执行新方案 → 复盘
- **压力升级**: L0信任→L1温和→L2灵魂拷问→L3严格审查→L4最终尝试（按失败次数）
- **冰山法则**: 修一个问题，扫一类问题。一个进来，一类出去
- **Owner 意识**: 主动发现问题，不等用户指出
- **抗合理化**: 9 种常见借口的反击机制
- **7 项检查清单**: L3+ 失败时强制完成的排查清单
- **体面退出**: 穷尽一切后输出结构化失败报告

**开发能力（融合自 iterate/tdd/spec-test/debug）**:
- **TDD 严格循环**: Red → Green → Refactor，每步验证
- **5 种测试方法**: 规格驱动 / 场景矩阵 / 不变量 / 决策表 / 回归
- **错误快速分类**: 按错误类型 → 定位文件（MappingError→YAML, ValidationError→format 等）
- **debug_dump 全链路**: 每个 pass 前后 JSON 快照 + 可视化

---

## Skill 2: t2c-scaffold — 脚手架引擎

**合并自**: add-op + add-pass + adapt-format（原 3 个 skill → 1 个）

**触发词**: 加算子 / 新增 op / add pass / 格式 / format / dtype / 对齐

**3 种类型**:

| 类型 | 触发词 | 产出 |
|------|--------|------|
| `op` | 加算子, add op | 10 步: mapping → signature → tiling → naming → cost → mock → test |
| `pass` | 加 pass | 7 步: 目录 → 入口 → toggle → pipeline → test |
| `format` | 格式, dtype, 对齐 | 4 步: YAML → sizing → codegen → mock |

**融合能力**:
- **算子全链路**: 一次完成 mapping/signature/tiling/naming/cost_model/C mock 7 个配置表
- **Pass 骨架**: 自动生成 run()/post_validate() + __init__.py + tests/
- **一致性守护**: 完成后自动跑 test_config_consistency.py
- **裂解支持**: 复合算子自动生成 decompositions.yaml 条目
- **Python cost fn**: 高精度成本需求时自动注册 @register_cost_fn

---

## Skill 3: t2c-arch — 架构大脑

**合并自**: arch + debug(参考部分) + setup（原 3 个 skill → 1 个）

**触发词**: 架构 / pipeline / 硬件 / Graph IR / format / 配置 / setup / 环境 / VSCode

**速查内容**:
- 17 pass 编译流程总览（4 阶段 9 必须 + 8 可选）
- 硬件模型（Cube/Vector/IDMA/DMA + HBM/L1）
- Graph IR 生命周期（Node/Tensor 各字段在每个阶段的状态变化）
- Tensor 格式系统（ND/NZ/ZZ/NN + format×dtype 对齐表）
- 配置系统（7 个 YAML + 流转路径）
- 内存策略（bulk → perop → spill → tiled 优先级）
- 关键设计决策速查表
- 环境搭建 + VSCode Tasks 速查
- 关键文件路径速查

---

## Guardrails（自动守护规则）

定义在 `skill-rules.json` 中，不需要手动触发：

| 规则 | 级别 | 触发条件 | 动作 |
|------|------|----------|------|
| **config-consistency** | block | 任何 `config/*.yaml` 被修改 | 必须跑一致性测试 |
| **alignment-hardcode** | warn | 代码中出现 `calc_padded_size(..., (16, 16))` | 提示用 `get_dim_align()` |
| **c-mock-addr** | warn | C mock 中出现 `.addr` | 提示用 `.ptr` 或 `npu_t_ptr()` |

---

## 安装方式

### Claude Code（原生支持）

```
.claude/commands/t2c-build.md     → /t2c-build
.claude/commands/t2c-scaffold.md  → /t2c-scaffold
.claude/commands/t2c-arch.md      → /t2c-arch
```

无需额外配置，放在 `.claude/commands/` 下自动识别为 slash command。

### GLM CLI

```python
# 1. 读取 SKILLS.md 获取 skill 索引
# 2. 读取 skill-rules.json 获取触发规则
# 3. 用户输入匹配 promptTriggers.keywords / intentPatterns
# 4. 匹配到 → 读取对应 .claude/commands/t2c-*.md 注入 system prompt
# 5. guardrails 在文件修改时自动检查

import json, re
rules = json.load(open('.claude/skill-rules.json'))
for name, skill in rules['skills'].items():
    for pattern in skill['promptTriggers']['intentPatterns']:
        if re.search(pattern, user_input, re.IGNORECASE):
            skill_content = open(f'.claude/commands/{name}.md').read()
            system_prompt += f'\n\n{skill_content}'
            break
```

### Cursor / Windsurf

将 SKILLS.md 的精简版写入 `.cursorrules`。

### 通用 AI 工具

`CLAUDE.md`（项目规则） + `SKILLS.md`（skill 索引） 作为 system prompt 注入。

---

## 架构工具箱（Sprint 1-4 新增 API）

写新 pass 时优先使用以下 API，降低出错概率：

| API | 用途 | 替代 |
|-----|------|------|
| `node.roofline` | typed 读写 `params["_roofline"]` | `node.params.get("_roofline")` |
| `ComputeUnit.CUBE` | 类型安全常量 | 裸字符串 `"cube"` |
| `TensorFormat.NZ` | 类型安全常量 | 裸字符串 `"nz"` |
| `Storage.LOCAL` | 类型安全常量 | 裸字符串 `"local"` |
| `FormatAnnotation.uniform(2,1)` | 结构化标注 | 手动构建 dict |
| `graph.rewire_input(nid, port, tid)` | 原子接线 | 手动更新 consumer 列表 |
| `graph.insert_node_before(target, node)` | 插入节点 | 手动 splice execution_order |
| `graph.single_consumer(tid)` | 查找唯一消费者 | 自写 `_find_single_consumer()` |
| `graph.intermediates()` | 中间 tensor 迭代器 | 自写过滤逻辑 |
| `graph_transaction(graph)` | 异常自动回滚 | 无 |
| `GraphBuilder` + `LAST` | 测试图构建 | 50 行手动 Tensor/Node |
| `MhaMergeConfig.from_raw(d)` | 类型安全 config | `config.get("key", default)` |
| `_PassDesc(..., kind="analysis")` | 声明 pass 类型 | 无 |
| `_PassDesc(..., requires=..., provides=...)` | 声明依赖 | 隐含在列表顺序 |

详见 `code_todo.md` 和 `graph_ir.py` 源码。

---

## 编程规范速查

| 规则 | 说明 |
|------|------|
| 函数 < 50 行 | 超过就拆子函数 |
| 模块核心代码 < 300 行 | 超过就拆模块 |
| 用 `common.get_logger(__name__)` | 不用 print |
| 用 `common.errors` 的异常类 | 不用裸 raise |
| 用 `opt_log` 记录优化决策 | 可视化依赖它 |
| 用 `get_dim_align()` 获取对齐值 | 不硬编码 |
| 用 `ComputeUnit.CUBE` 等 Enum | 不用裸字符串 |
| 用 `node.roofline` 等 descriptor | 不用 `params["_roofline"]` |
| 用 `graph.rewire_input()` 接线 | 不手动更新 consumer 列表 |
| 用 `GraphBuilder` 写测试 | 不手动创建 50 行 Graph |
| Config 改完跑一致性测试 | `test_config_consistency.py` |
| Python 3.10 + PyTorch 2.4+ | 不用 3.11+ 特性 |
| C99 标准 | mock 实现遵守 |
| 每个 pass 后 renumber | pipeline 自动做 |

## 危险操作清单

- 不要用 `tensor.addr`（正确是 `.ptr`）
- 不要硬编码 `calc_padded_size(..., (16, 16))`（用 `get_dim_align`）
- 不要在 op_mapping 中设 `is_mapped=True`（decomposition 才设）
- 不要跳过 config 一致性检查
- 不要在 pass 中修改 config（pass 是 graph → graph 的纯变换）
- 不要 `print` 调试信息（用 `logger.debug`）
- 不要用 `node.params["_roofline"]`（用 `node.roofline` descriptor）
- 不要硬编码 `"cube"` / `"nz"` / `"local"`（用 Enum 常量）

## VSCode Tasks 速查

| 开发 | 测试 | 编译 |
|------|------|------|
| `run:current` | `test:all` | `compile:minimal` |
| `run:current+viz` | `test:current` | `compile:full` |
| `install:dev` | `test:module` | `compile:both` |
| | `test:no-golden` | `compile:debug` |
| | `test:block-fuser` | `demo:e2e` |
| | `test:codegen` | `demo:module` |
| | `test:roofline` | `demo:viz` |
