# SKILLS.md — torch2c AI 协作技能集

> 本文件定义了 AI 助手在 torch2c 项目中工作时需要掌握的核心技能。
> 适用于任何 AI 编码助手（Claude Code、Cursor、Copilot、Windsurf 等）。
> 详细操作指南见 `.claude/commands/` 下的各 skill 文件。

---

## Skill 1: 理解架构 (`/arch`)

**触发词**: "架构"、"流水线"、"pipeline"、"编译流程"、"怎么编译的"

**能力要求**:
- 理解 4 阶段 9 必须 + 8 可选 pass 的编译流水线
- 理解 Graph IR 模型（Graph/Node/Tensor 及其字段生命周期）
- 理解硬件模型（Cube/Vector/IDMA/DMA 四单元 + HBM/L1 两级存储）
- 理解 tensor format 语义（format = HBM 格式，DMA 随路转换）
- 理解配置流转（YAML → pipeline._build_pass_configs → pass.run）

**参考文件**: `CLAUDE.md`、`docs/architecture.md`、`docs/tensor_formats.md`、`.claude/commands/arch.md`

---

## Skill 2: 新增算子 (`/add-op`)

**触发词**: "加算子"、"新增 op"、"支持 xxx 算子"、"add operator"

**能力要求**:
- 知道新增一个算子需要改哪些文件（10 个步骤）
- 知道 ATen op → NPU op 的映射关系
- 知道 C API 签名的 source 类型（tensor.input_N、param.xxx、dtype_enum 等）
- 知道 C mock 的编写规范（.ptr 不是 .addr、NPU_DBG_T、npu_read_compute）
- 知道一致性检查（test_config_consistency.py）

**必改文件清单**:
1. `torch2c/integration/config/direct_mappings.yaml`
2. `torch2c/integration/config/c_api_signatures.yaml`
3. `torch2c/integration/config/tiling_config.yaml`
4. `torch2c/integration/config/naming_rules.yaml`
5. `torch2c/integration/config/cost_model_config.yaml`
6. `npu_cpu_mock/include/npu_api.h`
7. `npu_cpu_mock/src/npu_compute_xxx.c`

**参考文件**: `.claude/commands/add-op.md`

---

## Skill 3: 新增 Pass (`/add-pass`)

**触发词**: "加 pass"、"新增优化"、"add pass"、"优化 pass"

**能力要求**:
- 知道 pass 的位置前缀规则（bc_/c_/cd_/d_）
- 知道 pass 接口规范（`run(graph, config) -> Graph` + `post_validate(graph) -> list[str]`）
- 知道 OptionalPass 枚举和 pipeline 注册
- 知道 opt_log 记录优化决策
- 知道 graph.renumber() 在每个 pass 后自动执行

**必改文件清单**:
1. `torch2c/optpass/{prefix}_{name}/{name}.py` (新建)
2. `torch2c/optpass/{prefix}_{name}/__init__.py` (新建)
3. `torch2c/optpass/{prefix}_{name}/tests/test_{name}.py` (新建)
4. `torch2c/common/pass_config.py` (加 toggle)
5. `torch2c/integration/pipeline.py` (注册 pass)

**参考文件**: `.claude/commands/add-pass.md`

---

## Skill 4: 调试 (`/debug`)

**触发词**: "报错"、"失败"、"debug"、"调试"、"不通过"、"为什么 fail"

**能力要求**:
- 根据错误类型快速定位问题域（MappingError/ValidationError/CodegenError 等）
- 使用 debug_dump 模式生成 pass 中间快照
- 使用可视化（pipeline.html）查看图变化
- 逐 pass 缩小范围
- C mock 调试（NPU_DEBUG_LEVEL=2）
- calc_padded_size 必须用 get_dim_align（不能硬编码对齐值）

**参考文件**: `.claude/commands/debug.md`

---

## Skill 5: Format/Dtype 适配 (`/adapt-format`)

**触发词**: "格式"、"format"、"对齐"、"alignment"、"block_pad"、"dtype"、"NZ"、"ZZ"

**能力要求**:
- 理解 format×dtype 二维对齐表
- 知道 `get_dim_align(fmt, dtype)` 是唯一的对齐值获取方式
- 知道 `_DEFAULT_ALIGNMENT` (sizing.py) 和 `block_pad.alignment` (YAML) 必须同步
- 知道添加新 format/dtype 的完整步骤
- 知道 format_capabilities 和 DMA 随路转换的关系

**参考文件**: `.claude/commands/adapt-format.md`、`docs/tensor_formats.md`

---

## Skill 6: TDD 工作流 (`/tdd`)

**触发词**: "测试"、"TDD"、"test"、"用例"、"怎么测"

**能力要求**:
- Red → Green → Refactor 循环
- 测试模式模板（Pass 测试、Config 一致性测试、端到端测试）
- 测试命名规范
- pytest 命令（单文件、单方法、-s、-x、--deselect）
- 全量回归：`pytest --tb=short -q`（当前 448 用例）

**参考文件**: `.claude/commands/tdd.md`

---

## Skill 7: 持续迭代 (`/iterate`)

**触发词**: "改动"、"修改"、"重构"、"迭代"、"workflow"

**能力要求**:
- 改动前：理解现有代码 + 确认范围
- 改动中：TDD + 增量验证
- 改动后：自查清单（8 项）+ 提交规范
- 场景化：bug fix / 加功能 / 重构 / 性能优化

**参考文件**: `.claude/commands/iterate.md`

---

## Skill 8: 环境搭建 (`/setup`)

**触发词**: "安装"、"环境"、"setup"、"配置开发环境"、"怎么跑起来"

**能力要求**:
- 知道 Python 3.10 + venv + `pip install -e ".[dev]"` 安装流程
- 知道 VSCode Tasks 的使用方式
- 知道如何验证环境正确（全量测试 + demo 编译）

**参考文件**: `QUICKSTART.md`、`.claude/commands/setup.md`

---

## VSCode Tasks 速查

`Cmd+Shift+P` → `Tasks: Run Task`：

| Task | 作用 | 典型场景 |
|------|------|----------|
| `test:all` | 全量 pytest（448 用例） | 改完代码回归验证 |
| `test:current` | pytest 当前文件 | 开发时快速验证 |
| `test:module` | pytest 指定模块 | 缩小排查范围 |
| `run:current` | 运行当前 .py | 跑 demo/脚本 |
| `run:current+viz` | 运行 + 打开 pipeline.html | 编译后看可视化 |
| `compile:debug` | 编译 DemoEncoder（debug_dump） | 全链路调试 |
| `demo:e2e` | 完整编译链 demo | 端到端验证 |
| `demo:st` | ST1-ST6 场景测试 | bulk/spill/tiling 验证 |
| `demo:module` | 选模块跑 demo | 单步查看 pass 中间结果 |
| `demo:viz` | 内存 lifetime 可视化 | 排查内存分配 |
| `install:dev` | 安装开发依赖 | 环境初始化 |

## 日志与调试工具速查

| 工具 | 怎么用 | 看什么 |
|------|--------|--------|
| `NPU_LOG_LEVEL=DEBUG` | 环境变量加在命令前 | 每个 tensor/节点的详细变化 |
| `debug_dump=True` | compile() 参数 | 每个 pass 前后的 JSON 快照 |
| `pipeline.html` | 浏览器打开 | 流水线 + 甬道图 + opt_log |
| `lifetime.html` | 浏览器打开 | HBM/L1 内存时序图 |
| `schedule.html` | 浏览器打开 | 调度时序 |
| `opt_log` | node.params["_opt_log"] | 每个节点为什么被优化 |
| `_pass_timing` | configs["_pass_timing"] | 每个 pass 耗时 ms |
| `NPU_DEBUG_LEVEL=2` | C 编译 -D 宏 | 算子输入输出值 |
| `debug.yaml` | config 目录下 | torch trace / memory layout 开关 |

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
| Config 改完跑一致性测试 | `test_config_consistency.py` |
| Python 3.10 + PyTorch 2.4+ | 不用 3.11+ 特性 |
| C99 标准 | mock 实现遵守 |
| 每个 pass 后 renumber | pipeline 自动做 |

## 危险操作清单（不要做）

- 不要用 `tensor.addr`（正确是 `.ptr`）
- 不要硬编码 `calc_padded_size(..., (16, 16))`（用 `get_dim_align`）
- 不要在 op_mapping 中设 `is_mapped=True`（decomposition 才设）
- 不要跳过 config 一致性检查
- 不要在 pass 中修改 config（pass 是 graph → graph 的纯变换）
- 不要 `print` 调试信息（用 `logger.debug`）
