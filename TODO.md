# TODO

## 功能扩展

- [ ] **multi-head merge pass**：决定多头注意力是合并计算还是拆分计算
  - 合并减少 block padding 浪费（总维度对齐一次 vs 每头各自对齐）
  - 合并提升 cube 利用率（更大矩阵）
  - 拆分利于 L1 容量（单头 tensor 更小，可能免 tiling）
  - 决策因子：padding overhead + tiling overhead，选总 overhead 更小的方案
  - 输出：调整图中 reshape/transpose 节点模式
