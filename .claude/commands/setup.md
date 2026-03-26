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

## 二、Pre-commit 护栏安装

```bash
pip install pre-commit
pre-commit install
```

安装后每次 `git commit` 会自动检查：
- Python lint（ruff）
- 禁止 `print()`（源码中应用 logger）
- 禁止 `.addr`（C mock 应用 `.ptr`）
- 禁止 `calc_padded_size(..., (16, 16))` 硬编码（应用 `get_dim_align`）
- 函数 > 50 行、文件 > 300 行（warning）

手动跑全量检查：

```bash
pre-commit run --all-files
```

单独跑规范检查器：

```bash
python scripts/check_conventions.py
```

## 三、代码模板使用

`templates/` 目录下有 4 个骨架文件，新增算子或 pass 时直接复制修改：

| 模板文件 | 用途 | 使用方式 |
|----------|------|----------|
| `new_pass_template.py` | 新 pass 骨架 | 复制到 `torch2c/optpass/{prefix}_{name}/{name}.py`，改 TODO |
| `new_pass_test_template.py` | 新 pass 测试骨架 | 复制到 `tests/test_{name}.py`，改 TODO |
| `new_op_c_mock_template.c` | C mock 骨架 | 复制到 `npu_cpu_mock/src/`，改 TODO |
| `new_op_checklist.md` | 新增算子 10 步清单 | 打开跟着做，每步有可粘贴代码 |

示例（新增 pass）：

```bash
# 1. 创建目录
mkdir -p torch2c/optpass/c_my_pass/tests

# 2. 复制模板
cp templates/new_pass_template.py torch2c/optpass/c_my_pass/my_pass.py
cp templates/new_pass_test_template.py torch2c/optpass/c_my_pass/tests/test_my_pass.py

# 3. 编辑 TODO 占位符
# 4. 注册到 pass_config.py + pipeline.py
# 5. 跑测试
.venv/bin/pytest torch2c/optpass/c_my_pass/tests/ -v
```

## 四、AI Skills 安装指南

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

### Cursor / GLM

`.cursorrules` 已创建，Cursor 会自动加载。内容是 CLAUDE.md 的精简版，
危险操作在最前面，适合能力较弱的模型。

配合 `templates/` 目录下的骨架文件使用效果更好。

### 其他 AI 工具

1. 将 `CLAUDE.md` 作为系统提示或项目上下文
2. 将 `SKILLS.md` 作为参考文档
3. 需要具体操作时，读取 `.claude/commands/` 下的对应文件

## 五、VSCode 配置

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

## 六、项目关键文件地图

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

## 七、快速验证一切正常

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
