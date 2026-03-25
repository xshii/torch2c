"""c_emitter — 生成 model_graph.c/h，每个算子：DMA搬入→算子调用→DMA搬出。"""
from __future__ import annotations
import math
import os
from torch2c.common import CodegenError, Node, Tensor, get_logger
from ._helpers import (
    DTYPE_C_ENUM_MAP, FORMAT_MAP, c_header_guard, find_op_sig, load_signatures,
    write_files,
)
from ._naming import (
    _build_c_name_map, _build_groups, _build_sectioned_c_names,
    _collect_used_tensor_ids, _group_name_to_c_func, _partition_tensors,
    _resolve_module_groups,
)
from ._struct_gen import (
    _build_dim_replace_map, _collect_spec_macros, _extract_model_dims,
    _gen_dim_macros, _gen_spec_macro_defs, _gen_tensor_struct_init,
    _gen_tensor_struct_typedef,
)
from ._tiled_emitter import gen_tiled_op_block

logger = get_logger("codegen.c_emitter")
_C_INCLUDES = (
    '#include "model_graph.h"\n#include "model_memory.h"\n'
    '#include "model_weights.h"\n#include "npu_api.h"\n\n\n'
)
_EMPTY_DMA = {"loads": [], "stores": []}

class SourceResolver:
    """将 c_api_signatures 中的 source 规则解析为 C 字符串。"""

    def __init__(self, node: Node, tensors: dict[str, Tensor],
                 c_names: dict[str, str] | None = None,
                 struct_prefix: str = "", dim_replace: dict[int, str] | None = None,
                 l1_layout: dict[str, int] | None = None):
        self._node, self._tensors = node, tensors
        self._c_names = c_names or {}
        self._prefix = struct_prefix
        self._dim_replace = dim_replace or {}
        self._l1_layout = l1_layout  # per-op L1 layout for inline descriptors

    def resolve(self, param: dict) -> str:
        """解析单个参数的 source → C 字符串。"""
        source, ptype = param.get("source", ""), param.get("type", "")
        parts = source.split(".")
        if ptype == "tensor_desc":
            if len(parts) < 2:
                raise CodegenError(f"tensor_desc source 格式错误（需要至少 2 段）: {source}")
            tid = self.find_tensor_id(parts[1])
            if tid is None:
                raise CodegenError(f"张量 {parts[1]} 未找到: node={self._node.id}")
            # per-op eviction: 使用 inline tensor descriptor
            if self._l1_layout and tid in self._l1_layout:
                t = self._tensors.get(tid)
                if t:
                    dt = DTYPE_C_ENUM_MAP.get(t.dtype or "fp16", "NPU_DTYPE_FP16")
                    fmt = FORMAT_MAP.get(t.format or "nd", "NPU_FORMAT_ND")
                    off = self._l1_layout[tid]
                    return f"(npu_tensor_t){{l1 + {off}, {dt}, {fmt}}}"
            return f"{self._prefix}{self._c_names.get(tid, tid)}"
        if parts[0] == "param":
            if len(parts) < 2:
                raise CodegenError(f"param source 格式错误（需要至少 2 段）: {source}")
            return self._resolve_param_ref(parts[1], param)
        if ptype in ("dtype_enum", "format_enum") or parts[0] == "tensor":
            if len(parts) < 3:
                raise CodegenError(f"tensor ref source 格式错误（需要至少 3 段）: {source}")
            return self._resolve_tensor_ref(parts[1], parts[2], parts[3:], param)
        raise CodegenError(f"未知 source 格式: {source}")

    def find_tensor_id(self, key: str) -> str | None:
        """根据 key (input_0, output_0, mask 等) 查找 tensor ID。"""
        if key.startswith(("input_", "output_")):
            try:
                idx = int(key.split("_")[1])
            except (IndexError, ValueError) as exc:
                raise CodegenError(f"无效的 tensor key: {key}") from exc
            if key.startswith("input_"):
                ids = self._node.active_inputs()
            else:
                ids = self._node.outputs
            return ids[idx] if idx < len(ids) else None
        if key == "mask":
            return self._node.absorbed_inputs.get("mask")
        return None

    def find_tensor(self, key: str) -> Tensor | None:
        tid = self.find_tensor_id(key)
        return self._tensors.get(tid) if tid else None

    def _resolve_tensor_ref(self, tensor_key, field, extra, param):
        t = self.find_tensor(tensor_key)
        if t is None:
            if param.get("default") is not None:
                return str(param["default"])
            raise CodegenError(f"张量 {tensor_key} 未找到: node={self._node.id}")
        return self._extract_field(t, field, extra)

    def _resolve_param_ref(self, param_name, param):
        val = self._node.params.get(param_name)
        if val is None:
            if param.get("default") is not None:
                return _format_value(param["default"], param["type"])
            raise CodegenError(f"参数 {param_name} 未找到: node={self._node.id}")
        if param["type"] == "int" and isinstance(val, int) and val in self._dim_replace:
            # 不替换维度索引参数和布尔标志，只替换维度大小
            if param_name not in ("dim0", "dim1", "transpose_b"):
                return self._dim_replace[val]
        return _format_value(val, param["type"])

    def _extract_field(self, t: Tensor, field: str, extra: list[str]) -> str:
        """从 Tensor 提取字段值。"""
        if field in ("l1_offset", "hbm_offset"):
            buf = "l1" if field == "l1_offset" else "hbm"
            offset = getattr(t, field) or 0
            return f"(void*)({buf} + {offset})"
        if field in ("dtype", "dtype_enum"):
            return DTYPE_C_ENUM_MAP.get(t.dtype or "fp16", "NPU_DTYPE_FP16")
        if field in ("format", "format_enum"):
            return FORMAT_MAP.get(t.format or "nd", "NPU_FORMAT_ND")
        if field == "shape":
            shape = t.shape
            if extra:
                idx = int(extra[0])
                if idx < -len(shape) or idx >= len(shape):
                    raise CodegenError(
                        f"shape 索引越界: {t.id}.shape[{idx}], shape={shape}")
                return self._dim_replace.get(shape[idx], str(shape[idx]))
            return str(shape)
        if field == "ndim":
            return str(len(t.shape))
        if field == "elem_count":
            val = math.prod(t.shape or [1])
            return self._dim_replace.get(val, str(val))
        if field == "hbm_size":
            val = t.hbm_size
            if val is None:
                from torch2c.common import DTYPE_INFO
                elem = math.prod(t.shape or [1])
                dtype_sz = DTYPE_INFO.get(t.dtype or "fp16")
                val = elem * (dtype_sz.bytes if dtype_sz else 2)
            return self._dim_replace.get(val, str(val))
        if not hasattr(t, field):
            raise CodegenError(f"未知的 tensor 字段: {field}")
        val = getattr(t, field)
        if val is None:
            raise CodegenError(f"tensor 字段 {field} 为 None: {t.id}")
        return str(val)

def _format_value(val, ptype: str) -> str:
    if ptype == "float":
        return f"{float(val):.6f}f"
    if ptype == "dtype_enum":
        return DTYPE_C_ENUM_MAP.get(str(val), f"NPU_DTYPE_{str(val).upper()}")
    if ptype == "format_enum":
        return FORMAT_MAP.get(str(val), f"NPU_FORMAT_{str(val).upper()}")
    return str(val)

def _build_tid_str(node: Node) -> str:
    """从 Node 构建 TidInfo 复合字面量字符串。"""
    p = node.params
    vals = [node.task_id, p.get("_tid_dep_cube", 0), p.get("_tid_dep_vector", 0),
            p.get("_tid_dep_dma", 0), p.get("_tid_dep_idma", 0)]
    return f"(TidInfo){{{', '.join(str(v) for v in vals)}}}"

def _gen_op_call(npu_op: str, sig: dict, node: Node, tensors: dict[str, Tensor],
                 c_names=None, struct_prefix="", dim_replace=None,
                 l1_layout=None) -> str:
    """生成单个算子的 C 调用语句（自动注入 TidInfo 为第一参数）。"""
    resolver = SourceResolver(node, tensors, c_names, struct_prefix, dim_replace, l1_layout)
    args = [_build_tid_str(node)]
    for p in sig.get("params", []):
        if p["type"] == "int_array":
            parts = p["source"].split(".")
            if len(parts) < 2:
                raise CodegenError(f"int_array source 格式错误: {p['source']}")
            t = resolver.find_tensor(parts[1])
            shape = t.shape if t else []
            args.append(f"(const int[]){{{', '.join(str(s) for s in shape)}}}")
        else:
            args.append(resolver.resolve(p))
    for p in sig.get("optional_params", []):
        args.append(resolver.resolve(p))
    return f"{npu_op}({', '.join(args)})"

def _gen_dma_line(instr: dict) -> str:
    """生成 dma_move 调用（TidInfo 全零，DMA 不参与图调度）。"""
    dt = DTYPE_C_ENUM_MAP.get(instr.get("dtype", "fp16"), "NPU_DTYPE_FP16")
    sf = FORMAT_MAP.get(instr.get("src_format", "nd"), "NPU_FORMAT_ND")
    df = FORMAT_MAP.get(instr.get("dst_format", "nd"), "NPU_FORMAT_ND")
    l1o, ho, sz = instr["l1_offset"], instr["hbm_offset"], instr["size_bytes"]
    if instr["op"] == "load":
        dst, src = f"(npu_tensor_t){{l1 + {l1o}, {dt}, {df}}}", f"(npu_tensor_t){{hbm + {ho}, {dt}, {sf}}}"
    else:
        dst, src = f"(npu_tensor_t){{hbm + {ho}, {dt}, {df}}}", f"(npu_tensor_t){{l1 + {l1o}, {dt}, {sf}}}"
    return f"dma_move((TidInfo){{0}}, {dst}, {src}, {sz});"

def _gen_dma_block(instructions: list[dict], indent: str = "    ") -> str:
    return "\n".join(f"{indent}{_gen_dma_line(i)}" for i in instructions)

def gen_op_block(node: Node, tensors: dict[str, Tensor], dma_plan: dict,
                 signatures: dict, c_names=None, struct_prefix="",
                 dim_replace=None) -> str:
    """为单个算子生成完整的代码块：DMA搬入→算子调用→DMA搬出。"""
    npu_op = node.npu_op or "unknown"
    sig = find_op_sig(signatures, npu_op)
    if sig is None:
        raise CodegenError(f"签名未找到: {npu_op}")

    tile_info = dma_plan.get("tile_info") or node.params.get("_tile_info")
    l1_layout = dma_plan.get("l1_layout")

    if tile_info:
        return gen_tiled_op_block(
            node, tensors, dma_plan, sig, npu_op, tile_info,
            c_names, struct_prefix, dim_replace, l1_layout,
            _gen_dma_line, _gen_op_call,
        )

    op_call = _gen_op_call(npu_op, sig, node, tensors, c_names, struct_prefix, dim_replace,
                           l1_layout)
    loads = _gen_dma_block(dma_plan.get("loads", []))
    stores = _gen_dma_block(dma_plan.get("stores", []))
    lines = [f"    /* === {node.id}: {npu_op} ({node.compute_unit or '?'}) === */"]
    if loads:
        lines.append(loads)
    lines.append(f"    {op_call};")
    if stores:
        lines.append(stores)
    return "\n".join(lines)

def _gen_bulk_dma(dma_plan: dict, label: str) -> str:
    instructions = dma_plan.get("loads", []) + dma_plan.get("stores", [])
    if not instructions:
        return ""
    return "\n".join([f"    /* === {label} === */", _gen_dma_block(instructions)])

def _gen_grouped_body(group_order, group_nids, dma_plans, nodes, tensors,
                      signatures, c_names, dim_replace, sections, spec_macros):
    """生成分组模式的函数定义 + model_run 函数体。返回 (func_sections, main_body)。"""
    prefix, indent = "t->", "    "
    func_sections, func_calls = [], []
    for group in group_order:
        nids = group_nids[group]
        fn = _group_name_to_c_func(group)
        ops = [gen_op_block(nodes[nid], tensors, dma_plans.get(nid, _EMPTY_DMA),
                            signatures, c_names, prefix, dim_replace) for nid in nids]
        func_sections.append(f"static void {fn}(model_tensors_t* t) {{\n"
                             + "\n\n".join(p for p in ops if p) + "\n}")
        func_calls.append(f"{indent}{fn}(&t);")
    # model_run body
    parts = [_gen_tensor_struct_init(sections, tensors, c_names, spec_macros)]
    bl = dma_plans.get("__bulk_load__")
    if bl:
        parts.append(_gen_bulk_dma(bl, "Bulk DMA Load"))
    parts.append("\n".join(func_calls))
    bs = dma_plans.get("__bulk_store__")
    if bs:
        parts.append(_gen_bulk_dma(bs, "Bulk DMA Store"))
    return func_sections, "\n\n".join(p for p in parts if p)

def emit_model_graph_c(plan, signatures: dict) -> str:
    """生成 model_graph.c 完整内容。"""
    nodes, tensors = plan.nodes, plan.tensors
    dma_plans = plan.dma_map
    order = plan.execution_order
    node_group = _resolve_module_groups(nodes, order)
    if not node_group:
        used_tids = _collect_used_tensor_ids(nodes, tensors, order, signatures)
        return _emit_flat(nodes, tensors, order, dma_plans, signatures,
                          _build_c_name_map(used_tids, tensors, nodes))
    group_order, group_nids = _build_groups(nodes, order, node_group)
    used_tids = _collect_used_tensor_ids(nodes, tensors, order, signatures)
    sections = _partition_tensors(
        used_tids, tensors, nodes, signatures, node_group, group_order, group_nids)
    c_names = _build_sectioned_c_names(sections, tensors, nodes)
    spec_macros = _collect_spec_macros(sections, tensors)
    model_dims = _extract_model_dims(tensors)
    dim_replace = _build_dim_replace_map(model_dims)
    struct_typedef = _gen_tensor_struct_typedef(sections, c_names)
    func_secs, main_body = _gen_grouped_body(
        group_order, group_nids, dma_plans, nodes, tensors,
        signatures, c_names, dim_replace, sections, spec_macros)
    all_macros = "\n\n".join(
        m for m in [_gen_dim_macros(model_dims), _gen_spec_macro_defs(spec_macros)] if m)
    global_ptrs = "static unsigned char *hbm, *l1;\n"
    return (f"{_C_INCLUDES}{all_macros}\n\n\n{struct_typedef}\n\n\n"
            + global_ptrs + "\n\n"
            + "\n\n\n".join(func_secs) + "\n\n\n"
            f"void model_run(unsigned char* hbm_, unsigned char* l1_) {{\n"
            f"    hbm = hbm_; l1 = l1_;\n{main_body}\n}}\n")

def _emit_flat(nodes, tensors, order, dma_plans, signatures, c_names):
    """无模块信息时退化为平铺模式。"""
    used_tids = _collect_used_tensor_ids(nodes, tensors, order, signatures)
    sections = [("flat", used_tids)]
    flat_names = {tid: f"flat.{c_names.get(tid, tid)}" for tid in used_tids}
    spec_macros = _collect_spec_macros(sections, tensors)
    struct_td = _gen_tensor_struct_typedef(sections, flat_names)
    struct_init = _gen_tensor_struct_init(sections, tensors, flat_names, spec_macros)
    parts = []
    bl = dma_plans.get("__bulk_load__")
    if bl:
        parts.append(_gen_bulk_dma(bl, "Bulk DMA Load"))
    for nid in order:
        parts.append(gen_op_block(nodes[nid], tensors,
                                  dma_plans.get(nid, _EMPTY_DMA), signatures, flat_names, "t."))
    bs = dma_plans.get("__bulk_store__")
    if bs:
        parts.append(_gen_bulk_dma(bs, "Bulk DMA Store"))
    body = "\n\n".join(p for p in parts if p)
    macros = _gen_spec_macro_defs(spec_macros)
    return (f"{_C_INCLUDES}{macros}\n\n{struct_td}\n\n"
            f"void model_run(unsigned char* hbm, unsigned char* l1) {{\n{struct_init}\n\n"
            f"{body}\n}}\n")

def emit_model_graph_h() -> str:
    return c_header_guard("MODEL_GRAPH_H",
                          "void model_run(unsigned char* hbm, unsigned char* l1);\n")

def emit_model_memory_h(plan) -> str:
    """生成 model_memory.h。"""
    lines = []
    for tid, t in plan.tensors.items():
        s = tid.upper()
        lines += [f"#define {s}_HBM_OFFSET  {t.hbm_offset or 0}",
                  f"#define {s}_HBM_SIZE    {t.hbm_size or 0}",
                  f"#define {s}_L1_OFFSET   {t.l1_offset or 0}"]
    return c_header_guard("MODEL_MEMORY_H", "\n".join(lines) + "\n")

def emit_model_params_h(plan) -> str:
    """生成 model_params.h。"""
    lines = [f"#define {nid}_{k}".upper() + f"  {v}"
             for nid, node in plan.nodes.items()
             for k, v in node.params.items()]
    return c_header_guard("MODEL_PARAMS_H", "\n".join(lines) + "\n")

def run(plan, output_dir: str, config_dir: str | None = None) -> None:
    """生成 model_graph.c/h, model_memory.h, model_params.h。"""
    logger.info("c_emitter: 开始生成 model_graph 文件")
    sigs = load_signatures(config_dir)
    write_files(os.path.join(output_dir, "src"), [
        ("model_graph.c", emit_model_graph_c(plan, sigs)),
        ("model_graph.h", emit_model_graph_h()),
        ("model_memory.h", emit_model_memory_h(plan)),
        ("model_params.h", emit_model_params_h(plan)),
    ])
