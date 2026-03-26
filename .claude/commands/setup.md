# 环境搭建与 Skills 安装指南

## 一、开发环境搭建

### 1. Python 环境

```bash
# 必须 Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

VSCode Task: `install:dev`

### 2. 验证环境

```bash
# 全量测试（约 8 秒，448 用例）
.venv/bin/pytest --tb=short -q

# 编译 demo 模型
python scripts/demo_axb.py

# 验证 C 工程
cd output/demo_axb/
cmake -B build && cmake --build build
cd build && ctest -V
```

### 3. 依赖版本

| 依赖 | 版本 |
|------|------|
| Python | ==3.10.* |
| PyTorch | >=2.4（需支持 torch.export） |
| PyYAML | >=6.0 |
| NumPy | >=1.24 |
| Flask | 可选（远端可视化） |

## 二、AI Skills 安装指南

### Claude Code

Skills 已通过以下文件自动生效：

```
CLAUDE.md                        # 项目规则（自动加载到每次对话）
SKILLS.md                        # 技能索引（AI 参考）
.claude/commands/
├── add-op.md                    # /add-op — 新增算子
├── add-pass.md                  # /add-pass — 新增 pass
├── debug.md                     # /debug — 调试指南
├── tdd.md                       # /tdd — TDD 工作流
├── iterate.md                   # /iterate — 持续迭代
├── arch.md                      # /arch — 架构深度理解
├── adapt-format.md              # /adapt-format — 格式适配
└── setup.md                     # /setup — 本文件
```

使用方式：在 Claude Code 中输入 `/add-op` 等斜杠命令即可触发对应 skill。

### Cursor

将以下内容追加到 `.cursorrules`（如果用 Cursor 的话）：

```
请阅读项目根目录的 CLAUDE.md 和 SKILLS.md，
它们定义了本项目的架构规则、编程规范、调试方法。
具体操作指南在 .claude/commands/ 目录下。
```

### 其他 AI 工具

1. 将 `CLAUDE.md` 作为系统提示或项目上下文
2. 将 `SKILLS.md` 作为参考文档
3. 需要具体操作时，读取 `.claude/commands/` 下的对应文件

## 三、VSCode 配置

### Tasks

已在 `.vscode/tasks.json` 中配置好，`Cmd+Shift+P` → `Tasks: Run Task` 即可使用。

关键 task:

| 日常开发 | 测试调试 | 编译运行 |
|----------|----------|----------|
| `install:dev` | `test:all` | `compile:debug` |
| `run:current` | `test:current` | `demo:e2e` |
| `run:current+viz` | `test:module` | `demo:module` |

### 推荐 VSCode 扩展

- Python (ms-python) — 必须
- Pylance — 类型检查
- C/C++ — 查看 npu_cpu_mock 源码

## 四、项目关键文件地图

```
开始接触项目时，按以下顺序阅读：

1. CLAUDE.md               ← 项目规则 + 架构总览（必读）
2. SKILLS.md                ← AI 技能索引
3. docs/tensor_formats.md   ← 格式系统详解
4. docs/architecture.md     ← 架构设计
5. QUICKSTART.md            ← 快速开始
6. TODO.md                  ← 待做事项
7. docs/roadmap.md          ← 路线图 + 技术债
```

## 五、快速验证一切正常

```bash
# 1. 安装
source .venv/bin/activate
pip install -e ".[dev]"

# 2. 全量测试
.venv/bin/pytest --tb=short -q
# 预期: 448 passed

# 3. 编译 demo
python scripts/demo_axb.py
# 预期: output/demo_axb/ 目录下有 C 工程

# 4. 编译 + 可视化
python scripts/compile_and_viz.py
open output/compile_viz/viz/pipeline.html
# 预期: 浏览器打开流水线可视化

# 5. C golden 验证
cd output/demo_axb/
cmake -B build && cmake --build build
cd build && ctest -V
# 预期: 100% tests passed
```
