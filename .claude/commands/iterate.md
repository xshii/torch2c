# 持续迭代工作流

在本项目中做任何改动时，遵循以下标准流程确保质量。

## 改动前

### 1. 理解当前状态

```bash
# 查看未完成的 TODO
cat TODO.md

# 查看当前测试状态
.venv/bin/pytest --tb=short -q

# 查看最近提交
git log --oneline -10
```

### 2. 确认改动范围

先读相关代码，理解现有实现再动手：

```bash
# 找到相关文件
grep -rn "关键词" torch2c/ --include="*.py" | head -20

# 读具体文件
cat torch2c/{module}/{file}.py
```

## 改动中

### 3. TDD 循环

每个功能点：
1. 写测试 → 确认 Red
2. 写实现 → 确认 Green
3. 重构 → 确认还是 Green

### 4. 增量验证

```bash
# 每改一个文件就跑对应模块测试
.venv/bin/pytest torch2c/{module}/tests/ -v --tb=short

# 每完成一个功能点就跑全量
.venv/bin/pytest --tb=short -q
```

### 5. 记录优化决策

如果改动涉及 pass 逻辑，用 opt_log 记录：

```python
from torch2c.common.opt_log import log_opt
log_opt(node, "pass_name", "做了什么", "为什么这么做")
```

## 改动后

### 6. 自查清单

- [ ] 全量测试通过（`pytest --tb=short -q`）
- [ ] 没有引入新的 warning
- [ ] 每个函数 < 50 行
- [ ] 模块核心代码 < 300 行
- [ ] 用了 common 的 logger（不是 print）
- [ ] 用了 common 的 errors（不是裸 raise）
- [ ] 改了 config 的话，config 一致性测试通过
- [ ] 改了 C mock 的话，golden 比对通过

### 7. 提交

```bash
git add <specific files>
git commit -m "type: 简短描述

详细说明（如果需要）

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

commit type: `feat` / `fix` / `refactor` / `test` / `docs` / `chore`

### 8. 推送

```bash
git push origin main
```

## 常见迭代场景

### 场景 A: 修 bug

1. 复现 bug（写测试）
2. 定位根因（读代码 + debug）
3. 最小修复
4. 确认测试通过
5. 检查有无类似 bug

### 场景 B: 加功能

1. 确认需求（用户意图）
2. 设计方案（需用户同意）
3. TDD 实现
4. 更新文档（如影响用户接口）

### 场景 C: 重构

1. 确保有充分测试覆盖
2. 小步重构，每步都跑测试
3. 不改行为，只改结构
4. 保持 API 兼容（或一次性更新所有调用方）

### 场景 D: 性能优化

1. 先量化现状（pass 耗时、内存使用）
2. 找瓶颈（不要猜）
3. 优化 + 验证效果
4. 确认正确性未退化

## 关键约束

- **不要一次改太多文件** — 改了就测，测了再改下一个
- **不要跳过测试** — 没有 green 不 commit
- **不要猜 API** — 先 grep/read 确认再写代码
- **改配置后必跑一致性测试** — `pytest torch2c/integration/tests/test_config_consistency.py`
