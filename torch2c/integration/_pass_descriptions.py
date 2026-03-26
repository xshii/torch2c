"""_pass_descriptions — Pass 文本描述与拓扑导出（从 pipeline.py 提取）。"""

from __future__ import annotations

_PASS_DESC: dict[str, dict[str, str]] = {
    "graph_capture": {
        "input": "PyTorch nn.Module + dummy_input",
        "output": "Graph IR（ATen 算子图，含 shape/dtype/权重）",
        "desc": "通过 torch.export 捕获模型前向图，保留高级算子（layer_norm、softmax 等），"
                "不做自动分解。标注权重、模型输入输出、attention mask。",
    },
    "op_mapping": {
        "input": "ATen 算子图（op_type = aten.xxx）",
        "output": "NPU 算子图（npu_op = cube_xxx / vector_xxx / idma_xxx）",
        "desc": "1:1 映射 ATen 算子到 NPU 指令。Cube 单元处理矩阵乘（16×16×16 MAC），"
                "Vector 处理逐元素/归约（SIMD），IDMA 处理数据搬运（reshape/transpose）。"
                "未命中映射表的算子保留 is_mapped=False，留给 op_decomposition 裂解。",
    },
    "op_decomposition": {
        "input": "部分未映射的复合算子 + 已映射的逐元素算子",
        "output": "全部映射为 NPU 原子算子",
        "desc": "两阶段处理：(1) 裂解 — 将 is_mapped=False 的复合算子按 decompositions.yaml "
                "规则拆为多步原子操作；(2) 广播展开 — 逐元素算子（vector_add 等）的两个输入 "
                "shape 不匹配时插入 idma_broadcast 节点。",
    },
    "op_absorption": {
        "input": "独立的 bias/add 节点",
        "output": "bias 融入 matmul，减少节点数",
        "desc": "将 bias 加法吸收到前序 matmul 中（cube_matmul → cube_matmul_bias），"
                "省去独立 Vector 运算 + 中间 tensor 的 HBM 读写。权重 transpose 也通过 DMA "
                "ND→NZ 随路转换吸收。",
    },
    "mha_merge": {
        "input": "MHA 投影链（matmul→view→transpose→reshape）",
        "output": "保持 merged 或拆分为 per-head 投影",
        "desc": "对每个 MHA block 做成本分析：比较合并投影（padding 少但可能需要 tiling）"
                "和拆分投影（padding 多但免 tiling）的总开销，选择更优方案。",
    },
    "format_annotator": {
        "input": "格式未标注的 tensor（默认 nd）",
        "output": "每个 tensor 标注 format（nd/nz）和 dtype（fp16/fp32）",
        "desc": "根据算子标注（@npu 装饰器）和全局目标 dtype，为每个 tensor 设置 NPU 存储格式。"
                "Cube 权重需要 NZ 分形格式，激活默认 ND。",
    },
    "format_planner": {
        "input": "初始格式标注",
        "output": "全局最优格式分配",
        "desc": "考虑硬件格式能力约束（如 Cube src1 必须 NZ），分析 tensor 的生产者-消费者链路，"
                "选择使运行时格式转换次数最少的全局方案。",
    },
    "reformat_inserter": {
        "input": "可能存在格式冲突的图",
        "output": "插入显式格式转换节点",
        "desc": "在相邻算子 format 不匹配处插入 reformat 节点（通过 DMA 随路转换实现）。"
                "如果 format_planner 已消除冲突，则不插入。",
    },
    "storage_assigner": {
        "input": "所有 tensor 默认 storage=hbm",
        "output": "符合 bypass 条件的 tensor 标记为 local/pipe",
        "desc": "分析生产者-消费者的计算单元对，如果满足硬件 bypass 条件（如 cube→vector），"
                "则 tensor 不回写 HBM，直接走 L1 local buffer 或 pipe 直连，"
                "省去 2 次 DMA 搬运的带宽开销。",
    },
    "block_pad": {
        "input": "原始 shape 的 tensor",
        "output": "shape 对齐到硬件块尺寸的 tensor",
        "desc": "Cube 以 16×16 分形块为计算粒度，要求 tensor 最后维对齐到 16 的倍数、"
                "最后两维乘积对齐到 256。未对齐的 tensor 补零对齐，避免硬件处理边界碎片。",
    },
    "validator": {
        "input": "完成标注的图",
        "output": "校验通过或报错",
        "desc": "检查所有节点的 npu_op 是否在 c_api_signatures 支持列表中，"
                "确保图在目标硬件上可执行。不修改图。",
    },
    "roofline_analyzer": {
        "input": "已标注的图",
        "output": "每个节点标注计算/访存瓶颈",
        "desc": "计算每个算子的算术强度（FLOPs/Bytes），与硬件 roofline 拐点比较。"
                "计算受限的算子应优化计算效率，访存受限的应减少数据搬运（tiling/bypass/融合）。",
    },
    "block_fuser": {
        "input": "独立算子 + roofline 分析",
        "output": "融合组 + tiling 参数",
        "desc": "Block 级数据流融合：按 HBM 节省量排序、贪心合并，受 L1 容量约束。"
                "联合 tiling，输出兼容 fusion_planner + global_tiler 接口。",
    },
    "fusion_planner": {
        "input": "独立的算子序列",
        "output": "标注融合组，中间 tensor 不落 HBM",
        "desc": "识别可融合的算子对（如 cube→vector 单消费者链），将它们标记为同一融合组。"
                "组内中间 tensor 留在 L1 不回写 HBM，减少 DMA 搬运。",
    },
    "scheduler": {
        "input": "无序的节点集合",
        "output": "确定 schedule_order 和 task_id",
        "desc": "基于数据依赖的拓扑排序，分配执行序号。"
                "无依赖的节点分配到不同 task_id，标记可并行。",
    },
    "global_tiler": {
        "input": "可能超出 L1 的大 tensor",
        "output": "标注 tiling 参数（tile_size, num_buffers）",
        "desc": "评估每个算子的 L1 峰值占用，如果超出 L1 容量则切分为多个 tile。"
                "双 buffer ping-pong 模式让 DMA 搬运与计算流水线重叠，隐藏访存延迟。",
    },
    "memory_planner": {
        "input": "未分配地址的 tensor",
        "output": "每个 tensor 的 HBM offset/size + DMA 计划",
        "desc": "基于生命周期分析分配 HBM 地址，避免冲突。"
                "生成 DMA load/store 指令序列，确定每个算子的数据搬运计划。",
    },
    "codegen": {
        "input": "完整编排的图 + DMA 计划",
        "output": "C99 工程（model_graph.c, CMakeLists.txt 等）",
        "desc": "将图翻译为 C 代码：每个算子生成三段式代码块（DMA 搬入 → NPU 算子调用 → DMA 搬出）。"
                "同时导出权重头文件和 golden 测试数据。",
    },
}


def get_pass_topology(
    optimization_passes: list,
    annotation_passes: list,
    late_passes: list,
) -> dict:
    """导出 pass 拓扑结构，供可视化自动生成。

    Returns:
        {"phases": [...], "passes": [...]}
        每个 pass: {"name", "number", "phase", "optional", "input", "output", "desc"}
    """
    phases = [
        {"id": "a_capture", "label": "Capture"},
        {"id": "b_lowering", "label": "Lowering"},
        {"id": "c_backend", "label": "Backend"},
        {"id": "d_emission", "label": "Emission"},
    ]

    passes: list[dict] = []
    passes.append({
        "name": "graph_capture", "number": "①",
        "phase": "a_capture", "optional": False,
        **_PASS_DESC.get("graph_capture", {}),
    })

    phase_map = [
        (optimization_passes, "b_lowering"),
        (annotation_passes, "c_backend"),
        (late_passes, "d_emission"),
    ]
    for pass_list, phase_id in phase_map:
        for p in pass_list:
            passes.append({
                "name": p.name,
                "number": p.number,
                "phase": phase_id,
                "optional": p.toggle is not None,
                **_PASS_DESC.get(p.name, {}),
            })

    passes.append({
        "name": "codegen", "number": "⑨",
        "phase": "d_emission", "optional": False,
        **_PASS_DESC.get("codegen", {}),
    })

    return {"phases": phases, "passes": passes}
