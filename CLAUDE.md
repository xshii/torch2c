# CLAUDE.md — 项目规则

## 工作流
- 方案决策需要用户同意，其余（写代码、跑测试、文件操作等）自行执行，不必确认
- 每个模块核心代码 < 300 行，每个函数 < 50 行
- 使用 common 的 logger、config_loader、errors

## 技术约束
- Python 3.10, PyTorch 2.4+, C99
- 测试：pytest (Python), ctest (C)
- 精度：max_abs_diff < 1e-3 (FP16), cosine > 0.999

## 项目结构
- `npu_compiler/` — Python 编译器包
- `npu_cpu_mock/` — NPU C API 的 CPU 模拟实现（测试框架）
- `docs/ordr.md` — 需求文档（权威来源）
- `docs/develop.md` — 多 Agent 并行开发指南
