"""Tiled DMA/op code generation: 从 c_emitter 拆出的 tiling 相关逻辑。"""
from __future__ import annotations

from ._helpers import DTYPE_C_ENUM_MAP, FORMAT_MAP


def _gen_tiled_dma_line(
    instr: dict, var: str = "_tile", indent: str = "",
    l1_off_expr: str | None = None,
) -> str:
    """生成 tiled DMA 调用：HBM 偏移随 tile 索引偏移。

    l1_off_expr 非 None 时使用动态 L1 偏移（多级缓冲）。
    batch > 1 且 tiled dim != dim 0 时，生成 per-batch 循环。
    """
    dt = DTYPE_C_ENUM_MAP.get(instr.get("dtype", "fp16"), "NPU_DTYPE_FP16")
    sf = FORMAT_MAP.get(instr.get("src_format", "nd"), "NPU_FORMAT_ND")
    df = FORMAT_MAP.get(instr.get("dst_format", "nd"), "NPU_FORMAT_ND")
    ho, sz = instr["hbm_offset"], instr["size_bytes"]
    stride = instr["tile_stride"]
    bc = instr.get("batch_count")

    # 决定 L1 基地址表达式
    if l1_off_expr is not None:
        l1_base = f"l1 + ({l1_off_expr})"
    else:
        l1_base = f"l1 + {instr['l1_offset']}"

    if bc and bc > 1:
        hbs = instr["hbm_batch_stride"]
        lbs = instr["l1_batch_stride"]
        per_batch_sz = sz // bc
        hbm_expr = f"hbm + {ho} + _b * {hbs} + {var} * {stride}"
        l1_expr = f"{l1_base} + _b * {lbs}"
        if instr["op"] == "load":
            dst = f"(npu_tensor_t){{{l1_expr}, {dt}, {df}}}"
            src = f"(npu_tensor_t){{{hbm_expr}, {dt}, {sf}}}"
        else:
            dst = f"(npu_tensor_t){{{hbm_expr}, {dt}, {df}}}"
            src = f"(npu_tensor_t){{{l1_expr}, {dt}, {sf}}}"
        call = f"dma_move((TidInfo){{0}}, {dst}, {src}, {per_batch_sz});"
        return (f"for (int _b = 0; _b < {bc}; _b++)\n"
                f"{indent}    {call}")

    hbm_expr = f"hbm + {ho} + {var} * {stride}"
    if instr["op"] == "load":
        dst = f"(npu_tensor_t){{{l1_base}, {dt}, {df}}}"
        src = f"(npu_tensor_t){{{hbm_expr}, {dt}, {sf}}}"
    else:
        dst = f"(npu_tensor_t){{{hbm_expr}, {dt}, {df}}}"
        src = f"(npu_tensor_t){{{l1_base}, {dt}, {sf}}}"
    return f"dma_move((TidInfo){{0}}, {dst}, {src}, {sz});"


def _build_buf_offset_expr(
    tid: str, tiled_tids: set, l1_layout: dict, extra_l1_offsets: list,
    num_buffers: int, var: str = "_tile",
) -> str | None:
    """构建 _tile 依赖的 L1 偏移 C 表达式。返回 None 表示非 tiled tensor。"""
    if tid not in tiled_tids or num_buffers <= 1:
        return None
    offsets = _collect_buf_offsets(tid, l1_layout, extra_l1_offsets)
    if num_buffers == 2:
        return f"(({var} % 2 == 0) ? {offsets[0]} : {offsets[1]})"
    arr = ", ".join(str(o) for o in offsets)
    return f"((const int[]){{{arr}}})[{var} % {num_buffers}]"


def _build_next_buf_offset_expr(
    tid: str, tiled_tids: set, l1_layout: dict, extra_l1_offsets: list,
    num_buffers: int, var: str = "_tile",
) -> str | None:
    """构建下一个 tile 使用的缓冲偏移表达式（用于 prefetch）。"""
    if tid not in tiled_tids or num_buffers <= 1:
        return None
    offsets = _collect_buf_offsets(tid, l1_layout, extra_l1_offsets)
    if num_buffers == 2:
        return f"(({var} % 2 == 0) ? {offsets[1]} : {offsets[0]})"
    arr = ", ".join(str(o) for o in offsets)
    return f"((const int[]){{{arr}}})[({var} + 1) % {num_buffers}]"


def _collect_buf_offsets(tid: str, l1_layout: dict, extra_l1_offsets: list) -> list[int]:
    """收集 tensor 在所有 buffer 中的 L1 偏移。"""
    ping = l1_layout[tid]
    return [ping] + [buf[tid] for buf in extra_l1_offsets if tid in buf]


def gen_tiled_op_block(
    node, tensors, dma_plan, sig, npu_op, tile_info,
    c_names, struct_prefix, dim_replace, l1_layout,
    gen_dma_line_fn, gen_op_call_fn,
) -> str:
    """生成 tiled 算子代码：untiled DMA → for 循环 { tiled DMA + op + tiled DMA }。

    gen_dma_line_fn / gen_op_call_fn: 从 c_emitter 注入，避免循环导入。
    """
    num_tiles = tile_info["num_tiles"]
    tile_size = tile_info["tile_size"]
    tiled_tids = set(tile_info.get("tiled_tensors", {}).keys())
    num_buffers = tile_info.get("num_buffers", 1)
    extra_l1_offsets = tile_info.get("extra_l1_offsets", [])
    indent, indent2, indent3 = "    ", "        ", "            "

    # 分离 tiled / untiled DMA 指令
    untiled_loads, tiled_loads = [], []
    for instr in dma_plan.get("loads", []):
        (tiled_loads if instr.get("tile_stride") else untiled_loads).append(instr)

    untiled_stores, tiled_stores = [], []
    for instr in dma_plan.get("stores", []):
        (tiled_stores if instr.get("tile_stride") else untiled_stores).append(instr)

    buf_label = f" {num_buffers}-buf" if num_buffers > 1 else ""
    lines = [f"{indent}/* === {node.id}: {npu_op} ({node.compute_unit or '?'})"
             f" tiled {tile_info['original_size']}→{num_tiles}×{tile_size}{buf_label} === */"]

    # untiled loads (before loop)
    for instr in untiled_loads:
        lines.append(f"{indent}{gen_dma_line_fn(instr)}")

    # 临时修改 tiled tensor 的 shape + hbm_size（try-finally 确保恢复）
    saved_attrs: dict[str, tuple[list[int], int | None]] = {}
    for tid, dim_idx in tile_info.get("tiled_tensors", {}).items():
        t = tensors.get(tid)
        if t:
            saved_attrs[tid] = (list(t.shape), t.hbm_size)
            t.shape = list(t.shape)
            t.shape[dim_idx] = tile_size
            t.hbm_size = None

    try:
        if num_buffers > 1 and l1_layout:
            _gen_multi_buffer_loop(
                lines, tiled_loads, tiled_stores, tiled_tids,
                l1_layout, extra_l1_offsets, num_buffers, num_tiles,
                node, tensors, sig, npu_op, c_names, struct_prefix,
                dim_replace, gen_dma_line_fn, gen_op_call_fn,
                indent, indent2, indent3,
            )
        else:
            _gen_single_buffer_loop(
                lines, tiled_loads, tiled_stores, num_tiles,
                node, tensors, sig, npu_op, c_names, struct_prefix,
                dim_replace, l1_layout, gen_op_call_fn,
                indent, indent2,
            )
    finally:
        # 恢复原始 shape + hbm_size
        for tid, (shape, hbm_size) in saved_attrs.items():
            tensors[tid].shape = shape
            tensors[tid].hbm_size = hbm_size

    # untiled stores (after loop)
    for instr in untiled_stores:
        lines.append(f"{indent}{gen_dma_line_fn(instr)}")

    return "\n".join(lines)


def _gen_multi_buffer_loop(
    lines, tiled_loads, tiled_stores, tiled_tids,
    l1_layout, extra_l1_offsets, num_buffers, num_tiles,
    node, tensors, sig, npu_op, c_names, struct_prefix,
    dim_replace, gen_dma_line_fn, gen_op_call_fn,
    indent, indent2, indent3,
) -> None:
    """多级缓冲：prologue + prefetch/compute/store 循环。"""
    # Prologue: 预加载 tile 0 到 buf 0
    lines.append(f"{indent}/* prologue: preload tile 0 */")
    for instr in tiled_loads:
        lines.append(f"{indent}{gen_dma_line_fn(instr)}")

    # op_call 使用动态 L1 偏移
    db_l1_layout = dict(l1_layout)
    for tid in tiled_tids:
        expr = _build_buf_offset_expr(
            tid, tiled_tids, l1_layout, extra_l1_offsets, num_buffers)
        if expr:
            db_l1_layout[tid] = expr
    op_call = gen_op_call_fn(npu_op, sig, node, tensors, c_names, struct_prefix,
                             dim_replace, db_l1_layout)

    lines.append(f"{indent}for (int _tile = 0; _tile < {num_tiles}; _tile++) {{")
    # Prefetch next tile
    lines.append(f"{indent2}/* prefetch next tile */")
    lines.append(f"{indent2}if (_tile + 1 < {num_tiles}) {{")
    for instr in tiled_loads:
        tid = instr["tensor_id"]
        next_expr = _build_next_buf_offset_expr(
            tid, tiled_tids, l1_layout, extra_l1_offsets, num_buffers)
        lines.append(f"{indent3}{_gen_tiled_dma_line(instr, '(_tile + 1)', indent3, next_expr)}")
    lines.append(f"{indent2}}}")

    # Compute current tile
    lines.append(f"{indent2}{op_call};")

    # Store current tile
    for instr in tiled_stores:
        tid = instr["tensor_id"]
        cur_expr = _build_buf_offset_expr(
            tid, tiled_tids, l1_layout, extra_l1_offsets, num_buffers)
        lines.append(f"{indent2}{_gen_tiled_dma_line(instr, '_tile', indent2, cur_expr)}")

    lines.append(f"{indent}}}")


def _gen_single_buffer_loop(
    lines, tiled_loads, tiled_stores, num_tiles,
    node, tensors, sig, npu_op, c_names, struct_prefix,
    dim_replace, l1_layout, gen_op_call_fn,
    indent, indent2,
) -> None:
    """单缓冲 tiled 循环。"""
    op_call = gen_op_call_fn(npu_op, sig, node, tensors, c_names, struct_prefix,
                             dim_replace, l1_layout)
    lines.append(f"{indent}for (int _tile = 0; _tile < {num_tiles}; _tile++) {{")
    for instr in tiled_loads:
        lines.append(f"{indent2}{_gen_tiled_dma_line(instr, '_tile', indent2)}")
    lines.append(f"{indent2}{op_call};")
    for instr in tiled_stores:
        lines.append(f"{indent2}{_gen_tiled_dma_line(instr, '_tile', indent2)}")
    lines.append(f"{indent}}}")
