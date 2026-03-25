# TODO

## P0 — 功能补全

- [ ] **format_planner 实际标注**：当前全标 ND，应根据 format_capabilities 给 cube 权重标 NZ、激活标 ZZ
- [ ] **DMA 随路格式转换**：codegen 生成 DMA load 时使用 tensor.format 设置 dst format
- [ ] **裂解产物签名**：c_api_signatures.yaml 缺少 vector_layernorm_part1/part2、vector_softmax_part1/part2

## P1 — 可扩展性

- [ ] 算子注册表一致性检查（映射/签名/mock/tiling/naming 集中定义）
- [ ] 配置 schema 校验（load_config 自动验证类型和结构）
- [ ] 参数化裂解规则（支持自定义 shape 推导）

## P2 — 工程质量

- [ ] conftest.py 共享 fixtures
- [ ] 测试覆盖率 ≥ 80%
- [ ] CI/CD（GitHub Actions）

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
