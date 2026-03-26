# TODO

## P0 — 功能补全

- [ ] **裂解产物签名**：c_api_signatures.yaml 缺少 vector_layernorm_part1/part2、vector_softmax_part1/part2

## P1 — 块级数据流融合（block_fuser）

基于 Blockbuster (MSFT 2025) / RedFuser (Ali, ASPLOS 2026) / PolyBlocks 2026 的思路，
用块级数据流建模替代当前贪心 pass 链，联合决策 fusion + tiling + storage。

### Phase 0：Codegen 消费融合标注（前置条件）

- [ ] **codegen 融合代码生成**：`c_emitter._gen_grouped_body` 检测同一 `_fusion_group`
      的连续节点，生成共享 tile 循环（一个外层 for 包裹组内所有节点），
      组内 `storage=local` 的 tensor 跳过 DMA load/store
- [ ] **测试**：matmul→relu→add 融合组生成单一 tile 循环，DMA 只搬外部 IO

### Phase 1：block_fuser pass（替换 fusion_planner ⑥c + global_tiler ⑦b）

- [ ] **BlockEdge / BlockNode 数据结构**：
      显式建模每条 tensor 在 HBM↔L1 间的搬运成本（elimination_benefit）
      和 L1 占用压力（l1_pressure），而非当前的"是否单消费者"二元判断
- [ ] **贪心融合算法**：按 elimination_benefit 降序遍历所有边，
      尝试融合 producer↔consumer，每步检查 L1 容量约束。
      支持 DAG 结构（不限于线性链）
- [ ] **组内联合 tile 决策**：融合组内所有节点共享 tile_size，
      二分搜索最大可行 tile（L1 峰值 ≤ 容量），支持 ping-pong
- [ ] **写回兼容接口**：输出 tensor.storage / _tile_config / fusion_groups，
      下游 memory_planner / codegen 不需要改动
- [ ] **OptionalPass.BLOCK_FUSER 开关**：可回退到旧 fusion_planner + global_tiler
- [ ] **测试**：线性链融合（同现有）、DAG 融合、L1 超容降级、tile 一致性

### Phase 2：Attention DAG 融合

- [ ] **attention pattern detection**：识别 Q@K^T → softmax → @V 的 DAG 结构，
      构建 AttentionBlock 数据结构（qk_matmul / softmax / sv_matmul）
- [ ] **两级 tile size 决策**：Tq（Q 行块）× Tk（K 列块），
      约束 partial_scores[Tq,Tk] + Q_tile + KT_tile + V_tile + accum ≤ L1
- [ ] **online softmax 变换**：编译器层面将 full softmax 替换为增量式
      online softmax（running_max + running_sum + correction）
- [ ] **codegen 嵌套 tile 循环**：生成 outer(Tq) × inner(Tk) 双层循环，
      中间 partial_scores 不落 HBM
- [ ] **C mock npu_vector_softmax_online**：online softmax CPU 模拟实现
- [ ] **c_api_signatures + cost_model 新增 fused_attention 相关算子**
- [ ] **测试**：单头 attention 融合、tile size 计算、online softmax 精度

## P2 — 可扩展性

- [ ] 算子注册表一致性检查（映射/签名/mock/tiling/naming 集中定义）
- [ ] 配置 schema 校验（load_config 自动验证类型和结构）
- [ ] 参数化裂解规则（支持自定义 shape 推导）

## P3 — 工程质量

- [ ] conftest.py 共享 fixtures
- [ ] 测试覆盖率 ≥ 80%
- [ ] CI/CD（GitHub Actions）

## P4 — 架构演进

- [ ] 动态 Shape（编译期 shape 符号化）
- [ ] 多核调度（task_id → 物理核映射）
- [ ] RL/LLM 驱动编译器调优（Compiler-R1 路线）

## 已完成 ✅

- [x] mha_merge pass（④b）
- [x] 目录重构（a_capture/b_lowering/c_backend/d_emission/optpass）
- [x] op_mapping/decomposition 解耦（npu_op 查裂解表）
- [x] transpose 移到 IDMA
- [x] 4 种格式 ND/NZ/ZZ/NN
- [x] opt_log 优化决策记录
- [x] Graph.renumber()
- [x] Pass 耗时统计
- [x] Pipeline 可视化（HTML + 甬道图 + DMA 节点）
- [x] npu_mock.h 消除
- [x] vector_relu + idma_embedding
- [x] Demo: AX+B / Embedding / 单层 Attention
- [x] format_planner 4-tier 偏好（score → consumer port pref → producer dst pref → default ND）
- [x] DMA 随路格式转换（src_format/dst_format 全链路）
- [x] global_tiler 扩展 vector 算子支持（20+ ops）
- [x] fusion_planner 激进 L1 驻留（fan-out>1 tensor → local）
- [x] 3-tier cost model（Python fn > YAML per-op > YAML unit default）
- [x] cost_model_config.yaml（per-op flops_multiplier + launch_cycles）
- [x] Graph.metadata 正式字段（替代 object.__setattr__）
- [x] mha_merge 统一 cost_model 参数来源

## 参考文献

| 系统 | 论文/项目 | 核心思路 |
|------|-----------|----------|
| TileLang | [arXiv 2504.17577](https://arxiv.org/abs/2504.17577) | tile 级 DSL，70 行写 FlashAttention，比 FA3 快 1.36x |
| Blockbuster | [arXiv 2505.07829](https://arxiv.org/abs/2505.07829) | 块级数据流建模，显式内存层级间搬运，规则融合 |
| RedFuser | [ASPLOS 2026](https://arxiv.org/abs/2603.10026) | 级联归约自动融合（softmax+GEMM），2-5x 加速 |
| PolyBlocks | [arXiv 2603.06731](https://arxiv.org/abs/2603.06731) | MLIR 基础设施，多级 tiling + fusion + scratchpad |
| AttentionEngine | [arXiv 2502.15349](https://arxiv.org/abs/2502.15349) | attention 分解为 scoring+aggregation，跨硬件优化 |
| FastAttention | [arXiv 2410.16663](https://arxiv.org/abs/2410.16663) | 昇腾两级 tiling，10.7x 加速 |
| Compiler-R1 | [NeurIPS 2025](https://arxiv.org/abs/2506.15701) | RL 训练 LLM 做编译器自动调优 |
| CUDA Tile IR | [GitHub](https://github.com/NVIDIA/cuda-tile) | NVIDIA 官方 tile 级 MLIR IR |
