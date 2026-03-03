# 多 Agent 并行开发实操指南

基于 `docs/ordr.md` 第 11 节，使用 Claude Code + git worktree 的具体操作步骤。

## 依赖结构

```
Phase 0: common (Python) + npu_cpu_mock (C)  ← 同分支并行，零文件交集
  │
  ├── Agent 1: graph_capture + op_mapping + op_decomposition
  ├── Agent 2: op_absorption + format_annotator + validator
  ├── Agent 3: memory_planner + scheduler
  └── Agent 4: codegen
          │
      integration (最后串联)
```

- common 和 npu_cpu_mock 分属 Python / C，目录完全不重叠，可在同一分支同时开发
- 4 组 Pass 模块只依赖 common、互不依赖，合并无冲突

## 分支策略

```
main
 ├── feature/common-dev     → common (Python)          ← worktree 并行
 ├── feature/npu-cpu-mock   → npu_cpu_mock (C)         ← worktree 并行
 │     合并到 main 后
 ├── feature/agent1  → graph_capture + op_mapping + op_decomposition  ← worktree 并行
 ├── feature/agent2  → op_absorption + format_annotator + validator   ← worktree 并行
 ├── feature/agent3  → memory_planner + scheduler                    ← worktree 并行
 ├── feature/agent4  → codegen                                       ← worktree 并行
 │     合并到 main 后
 └── feature/integration → pipeline + 端到端测试
```

## 第一步：用 worktree 并行开发 common + npu_cpu_mock

两个模块修改目录完全不重叠（Python vs C），用 worktree 隔离避免 git 操作冲突：

| 终端 | worktree 分支 | 负责目录 | 语言 | 验证命令 |
|------|-------------|---------|------|---------|
| 终端 A | feature/common-dev | `npu_compiler/common/` | Python | `pytest npu_compiler/common/tests/` |
| 终端 B | feature/npu-cpu-mock | `npu_cpu_mock/` | C | `cd npu_cpu_mock && cmake -B build && cmake --build build && cd build && ctest` |

```bash
# 0. 确保在 main 分支，创建 worktree
git checkout main
git worktree add .claude/worktrees/common -b feature/common-dev
git worktree add .claude/worktrees/mock -b feature/npu-cpu-mock
```

### 终端 A — common (Python)

```bash
cd .claude/worktrees/common && claude
```

Prompt:

```
阅读 npu_compiler/common/README.md，开发 common 模块全部代码和测试。
具体：graph_ir.py、logger.py、config_loader.py、errors.py 及对应 tests/。
pytest npu_compiler/common/tests/ 全绿后 commit。
```

### 终端 B — npu_cpu_mock (C)

```bash
cd .claude/worktrees/mock && claude
```

Prompt:

```
阅读 npu_cpu_mock/README.md，实现全部 17 个 NPU API 的 CPU mock 和 C 单元测试。
具体：include/ 头文件、src/ 全部实现、tests/ 全部测试、CMakeLists.txt。
cd npu_cpu_mock && cmake -B build && cmake --build build && cd build && ctest 全绿后 commit。
```

### 合并到 main

两个终端完成后：

```bash
cd /path/to/torch2c
git checkout main
git merge feature/common-dev
git merge feature/npu-cpu-mock

# 验证
pytest npu_compiler/common/tests/ --tb=short
cd npu_cpu_mock && cmake -B build && cmake --build build && cd build && ctest --output-on-failure
```

## 第二步：创建 4 个 worktree 并行开发

```bash
git checkout main

git worktree add .claude/worktrees/agent1 -b feature/agent1
git worktree add .claude/worktrees/agent2 -b feature/agent2
git worktree add .claude/worktrees/agent3 -b feature/agent3
git worktree add .claude/worktrees/agent4 -b feature/agent4
```

4 个终端分别启动 Claude Code，每个给对应的 prompt：

### 终端 1 — Agent 1: graph_capture + op_mapping + op_decomposition

```bash
cd .claude/worktrees/agent1 && claude
```

Prompt:

```
你负责开发 graph_capture、op_mapping、op_decomposition 三个模块。

前置阅读（必读）：
1. npu_compiler/common/ 下全部代码（graph_ir.py, logger.py, config_loader.py, errors.py）
2. 每个模块的 README：npu_compiler/graph_capture/README.md、npu_compiler/op_mapping/README.md、npu_compiler/op_decomposition/README.md
3. docs/ordr.md 第 5.2 节（Graph IR JSON 格式）
4. 各模块 config/ 目录下的 yaml 配置文件
5. docs/ordr.md 第 16.2、16.4、16.5 节（补充决策）

交付要求：每个模块的代码文件 + UT + demo，pytest 全绿后 commit。
代码原则：每模块核心 < 300 行，每函数 < 50 行，使用 common 的 logger/config_loader/errors。
```

### 终端 2 — Agent 2: op_absorption + format_annotator + validator

```bash
cd .claude/worktrees/agent2 && claude
```

Prompt:

```
你负责开发 op_absorption、format_annotator、validator 三个模块。

前置阅读（必读）：
1. npu_compiler/common/ 下全部代码（graph_ir.py, logger.py, config_loader.py, errors.py）
2. 每个模块的 README：npu_compiler/op_absorption/README.md、npu_compiler/format_annotator/README.md、npu_compiler/validator/README.md
3. docs/ordr.md 第 5.2 节（Graph IR JSON 格式）
4. 各模块 config/ 目录下的 yaml 配置文件
5. docs/ordr.md 第 16.3、16.7 节（补充决策）

交付要求：每个模块的代码文件 + UT + demo，pytest 全绿后 commit。
代码原则：每模块核心 < 300 行，每函数 < 50 行，使用 common 的 logger/config_loader/errors。
```

### 终端 3 — Agent 3: memory_planner + scheduler

```bash
cd .claude/worktrees/agent3 && claude
```

Prompt:

```
你负责开发 memory_planner、scheduler 两个模块。

前置阅读（必读）：
1. npu_compiler/common/ 下全部代码（graph_ir.py, logger.py, config_loader.py, errors.py）
2. 每个模块的 README：npu_compiler/memory_planner/README.md、npu_compiler/scheduler/README.md
3. docs/ordr.md 第 5.2 节（Graph IR JSON 格式）
4. 各模块 config/ 目录下的 yaml 配置文件（hardware_config.yaml）

交付要求：每个模块的代码文件 + UT + demo，pytest 全绿后 commit。
代码原则：每模块核心 < 300 行，每函数 < 50 行，使用 common 的 logger/config_loader/errors。
```

### 终端 4 — Agent 4: codegen

```bash
cd .claude/worktrees/agent4 && claude
```

Prompt:

```
你负责开发 codegen 模块。

前置阅读（必读）：
1. npu_compiler/common/ 下全部代码（graph_ir.py, logger.py, config_loader.py, errors.py）
2. npu_compiler/codegen/README.md
3. docs/ordr.md 第 5.2 节（Graph IR JSON 格式）
4. codegen 的 config/ 目录下的 yaml（c_api_signatures.yaml, codegen_config.yaml）和 templates/
5. docs/ordr.md 第 16.4 节（补充决策：transpose 4D 接口）

交付要求：代码文件 + UT + demo，pytest 全绿后 commit。
代码原则：核心 < 300 行，每函数 < 50 行，使用 common 的 logger/config_loader/errors。
```

## 第三步：合并 4 个 feature 分支

```bash
cd /path/to/torch2c
git checkout main

git merge feature/agent1
git merge feature/agent2
git merge feature/agent3
git merge feature/agent4

# 验证：全量测试
pytest --tb=short
cd npu_cpu_mock/build && ctest --output-on-failure
```

如有合并冲突（正常不会），手动解决后 `git add . && git commit`。

## 第四步：开发 integration

```bash
git checkout -b feature/integration
claude
```

Prompt:

```
你负责开发 integration 模块，将所有 Pass 串联成完整编译 pipeline。

前置阅读（必读）：
1. npu_compiler/common/ 下全部代码
2. npu_compiler/integration/README.md
3. 各 Pass 模块的代码（graph_capture, op_mapping, op_decomposition, op_absorption, format_annotator, validator, memory_planner, scheduler, codegen）
4. docs/ordr.md 第 3.1 节（整体流程）

交付要求：
1. pipeline.py — 串联全部 Pass
2. 端到端 demo：python -m npu_compiler.integration.demo.run_full_demo
3. pytest 全绿后 commit
```

完成后合并：

```bash
cd /path/to/torch2c
git checkout main && git merge feature/integration
```

## 第五步：清理 worktree

每个阶段合并后及时清理：

```bash
# 第一步合并后
git worktree remove .claude/worktrees/common
git worktree remove .claude/worktrees/mock
git branch -d feature/common-dev feature/npu-cpu-mock

# 第二步完成后
git worktree remove .claude/worktrees/agent1
git worktree remove .claude/worktrees/agent2
git worktree remove .claude/worktrees/agent3
git worktree remove .claude/worktrees/agent4
```

## 集成验收检查清单

| 检查项 | 命令 |
|--------|------|
| 全量 UT 通过 | `pytest --tb=short` |
| 端到端跑通 | `python -m npu_compiler.integration.demo.run_full_demo` |
| C mock UT 通过 | `cd npu_cpu_mock/build && ctest` |
| C 生成代码语法 | `gcc -fsyntax-only -include npu_api.h output/src/model_graph.c` |
| 算子数 = 62 | 检查 model_graph.c |
| 日志完整 | 每个 Pass 有入口/出口 INFO |

## 风险缓解

| 风险 | 措施 |
|------|------|
| Graph IR 理解不一致 | prompt 中指明阅读 graph_ir.py + ordr.md 5.2 节 |
| demo JSON 格式不统一 | 先用 graph_capture 生成真实 JSON 分发 |
| 某个 agent 失败 | 可在对应终端单独重跑 |
| 配置不匹配 | integration/config/ 为 single source of truth |
