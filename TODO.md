# TODO

## 正确性（优先修复）

- [x] **memory_planner 策略降级吞异常**：只 catch `MemoryPlanError`，其他异常透传；每次降级打 warning 日志
- [x] **策略降级状态泄漏**：`_reset_offsets()` 增加清理 `node.params["_tile_info"]`
- [x] **HBM lifetime 遗漏 absorbed_inputs**：`_hbm_alloc.py` 合并 `consumer_node_ids` 和 `absorbed_inputs` 中的消费者
- [x] **op_absorption 后 consumer 列表陈旧**：`remove_node()` 自动清理相关 tensor 的 consumer/producer 引用
- [x] **decomposition 中间 tensor shape 可配置**：添加 `inter_shape` 字段到 decompositions.yaml，支持 `reduce_last` 和自定义 shape 规则

## 架构（代码质量）

- [x] **c_emitter.py 拆分**：tiling 逻辑拆到 `_tiled_emitter.py`（c_emitter 293 行，_tiled_emitter 195 行）
- [x] **DMA 生成函数合并**：`_gen_tiled_dma_line` 和 `_gen_tiled_dma_line_db` 合并为一个函数，L1 偏移作为可选参数
- [x] **shape 临时修改加 try-finally**：`gen_tiled_op_block` 用 try-finally 包裹确保恢复
- [x] **shape 索引越界检查**：`_extract_field` 添加负索引和正索引的边界校验

## 性能与扩展

- [x] **调度器依赖改为结构冒险**：同 compute_unit + 共享 tensor 才串行，无共享 tensor 则可并行
- [x] **tiling 维度 per-op 可配**：`_TILEABLE_OPS` 改为 dict，每个 op 指定切分维度偏移
- [x] **手动 tiling 参数接口**：`compile()` 添加 `tile_override` 参数，支持指定 tile_size / num_buffers
- [x] **double buffer 自动触发**：`_find_tile_size_for_multi_buffer` 缩小 tile_size 以启用 ping-pong

## 易用性

- [x] **未映射 op 报错含模型层路径**：validator 报错附加 `module_path` 信息
- [x] **per-node 精度诊断**：`debug_dump=True` 时导出 per-node 中间结果到 `debug/intermediates/`，含 max_abs_diff 和 cosine 日志

## 可视化

- [x] ~~Schedule 甘特图去重：过滤 tiled op 的 summary 条~~
- [x] **DMA 箭头 tiling 展开**：per-tile 独立箭头，summary 条不生成出向依赖线

## 测试

- [x] ~~全量 ST 回归：ST1~ST9 全部通过~~
- [x] **结构冒险调度测试**：添加同单元共享 tensor 串行化测试
- [x] **添加 `demo:st6` VSCode task**
- [x] **添加 absorbed_inputs 对 HBM lifetime 影响的专项测试**
- [x] **添加策略降级全链路测试**（bulk → perop → spill → tiled）
- [x] **添加 batch>1 tiled bmm 的 ST 用例**（ST8: 4 头 MHA）

## 功能扩展

- [x] **完整 MHA 模型**（ST9）：Q/K/V 投影 + softmax + 加权求和 + output projection
- [x] **支持 `aten.slice.Tensor` 和 `aten.cat.default`**：NPU 映射 + C mock 实现
