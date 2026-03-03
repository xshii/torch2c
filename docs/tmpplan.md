
╭────────────────────────────────────────────────────────────────────────────────╮
│ Plan to implement                                                              │
│                                                                                │
│ 整改优化方案：4 项架构改进                                                     │
│                                                                                │
│ Context                                                                        │
│                                                                                │
│ 集成调试中发现 5 类跨模块问题（权重错配、参数命名、dim 语义、memcpy            │
│ 字节/元素混淆、配置双份不同步）。                                              │
│ 根因：4 个 Agent 并行开发时接口约定不充分，缺乏 Pass 边界校验。                │
│ 本方案按用户决策执行 4 项整改，目标：消除已知缺陷类别，防止新算子重复踩坑。    │
│                                                                                │
│ ---                                                                            │
│ WI-1: 配置单一源（删除 codegen 副本）                                          │
│                                                                                │
│ 改动范围小，其他 WI 依赖它先完成                                               │
│                                                                                │
│ ┌───────────────────────────────────────────────────┬────────────────────────┐ │
│ │                       文件                        │          操作          │ │
│ ├───────────────────────────────────────────────────┼────────────────────────┤ │
│ │ npu_compiler/codegen/config/c_api_signatures.yaml │ 删除                   │ │
│ ├───────────────────────────────────────────────────┼────────────────────────┤ │
│ │                                                   │ _DEFAULT_CONFIG_DIR    │ │
│ │ npu_compiler/codegen/_helpers.py:33               │ 改指向                 │ │
│ │                                                   │ integration/config     │ │
│ ├───────────────────────────────────────────────────┼────────────────────────┤ │
│ │ npu_compiler/codegen/tests/test_c_emitter.py:16   │ _CONFIG_DIR 改指向     │ │
│ │                                                   │ integration/config     │ │
│ ├───────────────────────────────────────────────────┼────────────────────────┤ │
│ │ 同文件 test_optional_mask_null (L127)             │ 删除 — integration     │ │
│ │                                                   │ 配置无 optional_params │ │
│ ├───────────────────────────────────────────────────┼────────────────────────┤ │
│ │ 同文件 test_softmax_with_absorbed_mask (L180)     │ 删除 — 同上            │ │
│ ├───────────────────────────────────────────────────┼────────────────────────┤ │
│ │ 同文件 test_softmax_without_mask_is_null (L198)   │ 删除 — 同上            │ │
│ └───────────────────────────────────────────────────┴────────────────────────┘ │
│                                                                                │
│ 验证：pytest npu_compiler/codegen/tests/ && pytest                             │
│ npu_compiler/integration/tests/                                                │
│                                                                                │
│ ---                                                                            │
│ WI-2: Mock memcpy 统一用字节数                                                 │
│                                                                                │
│ 3 个 memcpy 函数改为接收字节数，与 DMA 函数一致                                │
│                                                                                │
│ C 侧                                                                           │
│                                                                                │
│ ┌───────────────────────────────────────────┬───────────────────────────────┐  │
│ │                   文件                    │             改动              │  │
│ ├───────────────────────────────────────────┼───────────────────────────────┤  │
│ │                                           │ 3 个函数参数名 count/hidden → │  │
│ │ npu_cpu_mock/include/npu_api.h            │  size（语义：字节数），保留   │  │
│ │                                           │ dtype 不删                    │  │
│ ├───────────────────────────────────────────┼───────────────────────────────┤  │
│ │ npu_cpu_mock/src/npu_compute_transpose.c  │ memcpy(out, input,            │  │
│ │ npu_reshape                               │ (size_t)size); 去掉 *         │  │
│ │                                           │ npu_dtype_size                │  │
│ ├───────────────────────────────────────────┼───────────────────────────────┤  │
│ │ npu_cpu_mock/src/npu_compute_softmax.c    │ memcpy(out, inter,            │  │
│ │ npu_softmax_part2                         │ (size_t)size); 去掉 *         │  │
│ │                                           │ npu_dtype_size                │  │
│ ├───────────────────────────────────────────┼───────────────────────────────┤  │
│ │ npu_cpu_mock/src/npu_compute_norm.c       │ memcpy(out, inter,            │  │
│ │ npu_layernorm_part2                       │ (size_t)size); 去掉 *         │  │
│ │                                           │ npu_dtype_size                │  │
│ └───────────────────────────────────────────┴───────────────────────────────┘  │
│                                                                                │
│ C 测试（已有的 mock UT）                                                       │
│                                                                                │
│ ┌───────────────────────────┬───────────────────────────────────────────────┐  │
│ │           文件            │                     改动                      │  │
│ ├───────────────────────────┼───────────────────────────────────────────────┤  │
│ │ test_softmax.c:52         │ 已传 4*sizeof(float)=16 字节 →                │  │
│ │                           │ 不改（现在实现正确了）                        │  │
│ ├───────────────────────────┼───────────────────────────────────────────────┤  │
│ │ test_norm.c:59            │ 已传 8*sizeof(float)=32 字节 → 不改           │  │
│ ├───────────────────────────┼───────────────────────────────────────────────┤  │
│ │ test_transpose.c          │ 当前传元素数 6 → 改为 6 * (int)sizeof(float)  │  │
│ │ test_reshape              │                                               │  │
│ └───────────────────────────┴───────────────────────────────────────────────┘  │
│                                                                                │
│ 配置 YAML                                                                      │
│                                                                                │
│ integration/config/c_api_signatures.yaml 3 处 source 改动：                    │
│                                                                                │
│ ┌───────────────────┬─────┬────────────────────────┬───────────────────────┐   │
│ │       算子        │ 参  │       旧 source        │       新 source       │   │
│ │                   │ 数  │                        │                       │   │
│ ├───────────────────┼─────┼────────────────────────┼───────────────────────┤   │
│ │ npu_reshape       │ siz │ tensor.input_0.elem_co │ tensor.input_0.hbm_si │   │
│ │                   │ e   │ unt                    │ ze                    │   │
│ ├───────────────────┼─────┼────────────────────────┼───────────────────────┤   │
│ │ npu_softmax_part2 │ siz │ tensor.input_0.elem_co │ tensor.input_0.hbm_si │   │
│ │                   │ e   │ unt                    │ ze                    │   │
│ ├───────────────────┼─────┼────────────────────────┼───────────────────────┤   │
│ │ npu_layernorm_par │ siz │ tensor.input_0.elem_co │ tensor.input_0.hbm_si │   │
│ │ t2                │ e   │ unt                    │ ze                    │   │
│ └───────────────────┴─────┴────────────────────────┴───────────────────────┘   │
│                                                                                │
│ Python 集成测试                                                                │
│                                                                                │
│ integration/tests/demo_ut/test_c_ops.py — 所有手写 C 调用改传字节数：          │
│ - TestReshape: npu_reshape(x, out, {n*2}, ...) (FP16=2 bytes)                  │
│ - TestSoftmax: npu_softmax_part2(inter, out, {count*2}, ...)                   │
│ - TestLayerNorm: npu_layernorm_part2(inter, x, out, {count*2}, ...)            │
│ - TestAttentionBlock: npu_softmax_part2(sm_inter, attn, {seq*seq*2}, ...)      │
│                                                                                │
│ 验证：cd npu_cpu_mock/build && cmake .. && make && ctest + pytest              │
│ integration/tests/                                                             │
│                                                                                │
│ ---                                                                            │
│ WI-3: 修正上游模块（移除 pipeline 补丁）                                       │
│                                                                                │
│ 将 4 个适配函数回归到 graph_capture / op_decomposition                         │
│                                                                                │
│ 3.1 addmm 输入重排 → graph_capture.py                                          │
│                                                                                │
│ _handle_call() 中，在构建 input_tids 之后、创建 Node 之前：                    │
│ if op == "aten.addmm.default" and len(input_tids) >= 3:                        │
│     input_tids = [input_tids[1], input_tids[2], input_tids[0]]                 │
│                                                                                │
│ 3.2 参数重命名 → graph_capture.py                                              │
│                                                                                │
│ 模块级定义 _PARAM_RENAMES 字典（从 pipeline.py 迁移），在 _handle_call() 中    │
│ params 构建完成后应用：                                                        │
│ renames = _PARAM_RENAMES.get(op)                                               │
│ if renames:                                                                    │
│     for old_key, new_key in renames.items():                                   │
│         if old_key in params and new_key not in params:                        │
│             params[new_key] = params.pop(old_key)                              │
│                                                                                │
│ 3.3 负索引 dim 解析 → graph_capture.py                                         │
│                                                                                │
│ 在 capture() 中，graph 构建完成后、validate() 之前调用                         │
│ _resolve_negative_dims(graph)。                                                │
│ 函数体从 pipeline.py 整体迁移（含 softmax dim→size 转换逻辑）。                │
│                                                                                │
│ 3.4 layernorm part2 缺失输入 → op_decomposition.py                             │
│                                                                                │
│ _decompose_node() L69-72，对 layernorm_part2 步骤追加原始输入：                │
│ if i == 0:                                                                     │
│     inputs = list(node.inputs)                                                 │
│ else:                                                                          │
│     inputs = [intermediates[i - 1]]                                            │
│     if step["npu_op"] == "npu_layernorm_part2" and node.inputs:                │
│         inputs.append(node.inputs[0])                                          │
│                                                                                │
│ 同时在 L96-102 的 consumer 更新循环之后，为 part2 引用的原始输入更新           │
│ consumer_node_ids。                                                            │
│                                                                                │
│ 3.5 pipeline.py 清理                                                           │
│                                                                                │
│ 删除：_PARAM_RENAMES、_normalize_graph()、_resolve_negative_dims()、_fix_layer │
│ norm_decomp()                                                                  │
│ 删除 compile() 中的 3 处调用。                                                 │
│ 保留 _propagate_input_dtypes()（format_annotator 的遗留问题，范围外）。        │
│                                                                                │
│ 测试更新                                                                       │
│                                                                                │
│ ┌───────────────────────────┬───────────────────────────────────────────────┐  │
│ │         测试文件          │                   新增用例                    │  │
│ ├───────────────────────────┼───────────────────────────────────────────────┤  │
│ │ graph_capture/tests/test_ │ test_addmm_inputs_reordered、test_param_renam │  │
│ │ graph_capture.py          │ es_transpose、test_negative_dim_resolved      │  │
│ ├───────────────────────────┼───────────────────────────────────────────────┤  │
│ │ op_decomposition/tests/te │ test_layernorm_part2_has_orig_input           │  │
│ │ st_op_decomposition.py    │                                               │  │
│ └───────────────────────────┴───────────────────────────────────────────────┘  │
│                                                                                │
│ 验证：pytest npu_compiler/graph_capture/tests/ && pytest                       │
│ npu_compiler/op_decomposition/tests/ && pytest npu_compiler/integration/tests/ │
│                                                                                │
│ ---                                                                            │
│ WI-4: Pass 边界 Schema 校验                                                    │
│                                                                                │
│ Graph.validate_phase() — 在每个 Pass 之后检查关键字段                          │
│                                                                                │
│ graph_ir.py 新增                                                               │
│                                                                                │
│ def validate_phase(self, phase: str) -> list[str]:                             │
│     errors = self.validate()                                                   │
│     validator = _PHASE_VALIDATORS.get(phase)                                   │
│     if validator:                                                              │
│         errors.extend(validator(self))                                         │
│     return errors                                                              │
│                                                                                │
│ 5 个阶段校验函数（模块级私有）：                                               │
│                                                                                │
│ ┌──────────────────┬─────────────────────────────────────────────────────────┐ │
│ │       阶段       │                        校验规则                         │ │
│ ├──────────────────┼─────────────────────────────────────────────────────────┤ │
│ │ graph_capture    │ 权重 tensor 必须有 name；模型输入必须有                 │ │
│ │                  │ dtype/shape；节点必须有 op_type                         │ │
│ ├──────────────────┼─────────────────────────────────────────────────────────┤ │
│ │ op_mapping       │ is_mapped=True 的节点必须有 npu_op                      │ │
│ ├──────────────────┼─────────────────────────────────────────────────────────┤ │
│ │ op_decomposition │ 所有节点必须 is_mapped、有 npu_op、有 compute_unit      │ │
│ ├──────────────────┼─────────────────────────────────────────────────────────┤ │
│ │ format_annotator │ 有 producer 的 tensor 必须有 dtype                      │ │
│ ├──────────────────┼─────────────────────────────────────────────────────────┤ │
│ │ memory_planner   │ 有 consumer 或是 output 的 tensor 必须有                │ │
│ │                  │ hbm_offset/hbm_size/l1_offset                           │ │
│ └──────────────────┴─────────────────────────────────────────────────────────┘ │
│                                                                                │
│ pipeline.py 集成                                                               │
│                                                                                │
│ 每个 Pass 之后调用 graph.validate_phase("xxx")，校验不通过时                   │
│ logger.warning（不抛异常，避免阻断现有流程）。                                 │
│                                                                                │
│ 测试                                                                           │
│                                                                                │
│ common/tests/test_graph_ir.py 新增 TestValidatePhase                           │
│ 类，覆盖每个阶段的正向和反向用例。                                             │
│                                                                                │
│ 验证：pytest npu_compiler/common/tests/ && pytest                              │
│ npu_compiler/integration/tests/                                                │
│                                                                                │
│ ---                                                                            │
│ 执行顺序                                                                       │
│                                                                                │
│ WI-1 (配置单一源)                                                              │
│   ↓                                                                            │
│ WI-2 (Mock 字节数)  ←  依赖 WI-1（只改一份 yaml）                              │
│   ↓                                                                            │
│ WI-4 (Schema 校验)  ←  独立，纯新增                                            │
│   ↓                                                                            │
│ WI-3 (上游修正)     ←  最高风险，受益于 WI-4 的校验保护                        │
│                                                                                │
│ 最终验证                                                                       │
│                                                                                │
│ cd npu_cpu_mock/build && cmake .. && make && ctest --output-on-failure         │
│ .venv/bin/python3 -m pytest --tb=short -q   # 全量 120 测试                    │
╰───────────────────────────────────────────────────────────────────