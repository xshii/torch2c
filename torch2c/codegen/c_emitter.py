"""c_emitter — 生成 model_graph.c/h，每个算子三段式：DMA搬入→算子调用→DMA搬出。"""

from __future__ import annotations

import math
import os

from torch2c.common import CodegenError, get_logger

from ._helpers import (
    DTYPE_C_ENUM_MAP,
    FORMAT_MAP,
    c_header_guard,
    load_signatures,
    write_files,
)

logger = get_logger("codegen.c_emitter")


# ---- 参数解析 ----


class SourceResolver:
    """将 c_api_signatures 中的 source 规则解析为 C 字符串。"""

    def __init__(
        self, node: dict, tensors: dict,
        c_names: dict[str, str] | None = None,
        struct_prefix: str = "",
        dim_replace: dict[int, str] | None = None,
    ):
        self._node = node
        self._tensors = tensors
        self._c_names = c_names or {}
        self._prefix = struct_prefix  # e.g. "t->" for struct access
        self._dim_replace = dim_replace or {}  # val → macro name

    def resolve(self, param: dict) -> str:
        """解析单个参数的 source → C 字符串。"""
        source = param.get("source", "")
        ptype = param.get("type", "")
        parts = source.split(".")

        # tensor_desc: 返回 tensor 变量名
        if ptype == "tensor_desc":
            if len(parts) < 2:
                raise CodegenError(f"tensor_desc source 格式错误（需要至少 2 段）: {source}")
            return self._resolve_tensor_desc(parts[1], 0)
        # param ref（优先于 dtype_enum 等类型检查，因为 param.compute_dtype 也可以是 dtype_enum 类型）
        if parts[0] == "param":
            if len(parts) < 2:
                raise CodegenError(f"param source 格式错误（需要至少 2 段）: {source}")
            return self._resolve_param_ref(parts[1], param)
        # dtype_enum / format_enum / tensor ref: 从 tensor 取字段
        if ptype in ("dtype_enum", "format_enum") or parts[0] == "tensor":
            if len(parts) < 3:
                raise CodegenError(f"tensor ref source 格式错误（需要至少 3 段）: {source}")
            return self._resolve_tensor_ref(parts[1], parts[2], parts[3:], param)
        raise CodegenError(f"未知 source 格式: {source}")

    def _resolve_tensor_desc(self, tensor_key: str, addr_shift: int) -> str:
        """返回 tensor 变量名（带可选结构体前缀）。"""
        tid = self.find_tensor_id(tensor_key)
        if tid is None:
            raise CodegenError(f"张量 {tensor_key} 未找到: node={self._node['id']}")
        c_name = self._c_names.get(tid, tid)
        return f"{self._prefix}{c_name}"

    def find_tensor_id(self, key: str) -> str | None:
        """根据 key (input_0, output_0, mask 等) 查找 tensor ID。"""
        if key.startswith("input_"):
            try:
                idx = int(key.split("_")[1])
            except (IndexError, ValueError) as exc:
                raise CodegenError(f"无效的 tensor key: {key}") from exc
            inputs = self._node.get("inputs", [])
            absorbed_tids = set(self._node.get("absorbed_inputs", {}).values())
            regular = [tid for tid in inputs if tid not in absorbed_tids]
            return regular[idx] if idx < len(regular) else None
        if key.startswith("output_"):
            try:
                idx = int(key.split("_")[1])
            except (IndexError, ValueError) as exc:
                raise CodegenError(f"无效的 tensor key: {key}") from exc
            outputs = self._node.get("outputs", [])
            return outputs[idx] if idx < len(outputs) else None
        if key == "mask":
            return self._node.get("absorbed_inputs", {}).get("mask")
        return None

    def find_tensor(self, key: str):
        """根据 key (input_0, output_0, mask 等) 查找 tensor 数据。"""
        tid = self.find_tensor_id(key)
        return self._tensors.get(tid) if tid else None

    def _resolve_tensor_ref(self, tensor_key, field, extra, param):
        t = self.find_tensor(tensor_key)
        if t is None:
            default = param.get("default")
            if default is not None:
                return str(default)
            raise CodegenError(f"张量 {tensor_key} 未找到: node={self._node['id']}")
        return self._extract_field(t, field, extra)

    def _resolve_param_ref(self, param_name, param):
        val = self._node.get("params", {}).get(param_name)
        if val is None:
            default = param.get("default")
            if default is not None:
                return _format_value(default, param["type"])
            raise CodegenError(f"参数 {param_name} 未找到: node={self._node['id']}")
        # 整数参数尝试维度宏替换
        if param["type"] == "int" and isinstance(val, int) and val in self._dim_replace:
            return self._dim_replace[val]
        return _format_value(val, param["type"])

    def _extract_field(self, t: dict, field: str, extra: list[str]) -> str:
        """从 tensor dict 提取字段值。"""
        if field in ("l1_offset", "hbm_offset"):
            offset = t.get(field, 0) or 0
            buf = "l1" if field == "l1_offset" else "hbm"
            return f"(void*)({buf} + {offset})"
        if field in ("dtype", "dtype_enum"):
            return DTYPE_C_ENUM_MAP.get(t.get("dtype", "fp16"), "NPU_DTYPE_FP16")
        if field in ("format", "format_enum"):
            return FORMAT_MAP.get(t.get("format", "nd"), "NPU_FORMAT_ND")
        if field == "shape":
            shape = t.get("shape", [])
            if extra:
                val = shape[int(extra[0])]
                return self._dim_replace.get(val, str(val))
            return str(shape)
        if field == "ndim":
            return str(len(t.get("shape", [])))
        if field == "elem_count":
            val = math.prod(t.get("shape", [1]))
            return self._dim_replace.get(val, str(val))
        if field not in t:
            raise CodegenError(f"未知的 tensor 字段: {field}")
        return str(t[field])


def _format_value(val, ptype: str) -> str:
    if ptype == "float":
        return f"{float(val):.6f}f"
    if ptype == "dtype_enum":
        return DTYPE_C_ENUM_MAP.get(str(val), f"NPU_DTYPE_{str(val).upper()}")
    if ptype == "format_enum":
        return FORMAT_MAP.get(str(val), f"NPU_FORMAT_{str(val).upper()}")
    return str(val)


# ---- 代码生成 ----


def _gen_op_call(
    npu_op: str, sig: dict, node: dict, tensors: dict,
    c_names: dict[str, str] | None = None,
    struct_prefix: str = "",
    dim_replace: dict[int, str] | None = None,
) -> str:
    """生成单个算子的 C 调用语句。"""
    resolver = SourceResolver(node, tensors, c_names, struct_prefix, dim_replace)
    args = []
    for p in sig.get("params", []):
        if p["type"] == "int_array":
            parts = p["source"].split(".")
            if len(parts) < 2:
                raise CodegenError(f"int_array source 格式错误: {p['source']}")
            t = resolver.find_tensor(parts[1])
            shape = t.get("shape", []) if t else []
            args.append(f"(const int[]){{{', '.join(str(s) for s in shape)}}}")
        else:
            args.append(resolver.resolve(p))

    for p in sig.get("optional_params", []):
        args.append(resolver.resolve(p))

    return f"{npu_op}({', '.join(args)})"


def _gen_dma_line(instr: dict) -> str:
    """生成单条 DMA 指令 C 代码。"""
    src_fmt = FORMAT_MAP.get(instr.get("src_format", "nd"), "NPU_FORMAT_ND")
    dst_fmt = FORMAT_MAP.get(instr.get("dst_format", "nd"), "NPU_FORMAT_ND")
    if instr["op"] == "load":
        return (
            f"npu_dma_load((void*)(l1 + {instr['l1_offset']}), "
            f"(void*)(hbm + {instr['hbm_offset']}), "
            f"{instr['size_bytes']}, {src_fmt}, {dst_fmt});"
        )
    return (
        f"npu_dma_store((void*)(hbm + {instr['hbm_offset']}), "
        f"(void*)(l1 + {instr['l1_offset']}), "
        f"{instr['size_bytes']}, {src_fmt}, {dst_fmt});"
    )


def _gen_dma_block(instructions: list[dict], indent: str) -> str:
    """生成 DMA load/store 语句列表。"""
    return "\n".join(f"{indent}{_gen_dma_line(i)}" for i in instructions)


def gen_op_block(
    node: dict, tensors: dict, dma_plan: dict, signatures: dict,
    c_names: dict[str, str] | None = None,
    struct_prefix: str = "",
    dim_replace: dict[int, str] | None = None,
) -> str:
    """为单个算子生成完整的三段式代码块。"""
    npu_op = node.get("npu_op", "unknown")
    sig = signatures.get("compute_ops", {}).get(npu_op)
    if sig is None:
        raise CodegenError(f"签名未找到: {npu_op}")

    indent = "    "
    op_call = _gen_op_call(npu_op, sig, node, tensors, c_names, struct_prefix, dim_replace)
    loads = _gen_dma_block(dma_plan.get("loads", []), indent)
    stores = _gen_dma_block(dma_plan.get("stores", []), indent)

    lines = [f"{indent}/* === {node['id']}: {npu_op} ({node.get('compute_unit', '?')}) === */"]
    if loads:
        lines.append(loads)
        lines.append(f"{indent}npu_dma_barrier();")
    lines.append(f"{indent}{op_call};")
    if stores:
        lines.append(stores)
        lines.append(f"{indent}npu_dma_barrier();")
    return "\n".join(lines)


# ---- 文件级生成 ----


def _gen_bulk_dma(dma_plan: dict, label: str) -> str:
    """生成 bulk DMA (load/store) 代码块。"""
    indent = "    "
    loads = dma_plan.get("loads", [])
    stores = dma_plan.get("stores", [])
    instructions = loads + stores
    if not instructions:
        return ""
    lines = [f"{indent}/* === {label} === */"]
    lines.append(_gen_dma_block(instructions, indent))
    lines.append(f"{indent}npu_dma_barrier();")
    return "\n".join(lines)


def _collect_used_tensor_ids(plan: dict, signatures: dict) -> list[str]:
    """收集所有 compute op 引用的 tensor ID（去重、保序）。"""
    nodes = plan["nodes"]
    tensors = plan["tensors"]
    order = plan.get("execution_order", list(nodes.keys()))
    seen: set[str] = set()
    result: list[str] = []
    for nid in order:
        node = nodes[nid]
        npu_op = node.get("npu_op", "unknown")
        sig = signatures.get("compute_ops", {}).get(npu_op)
        if sig is None:
            continue
        resolver = SourceResolver(node, tensors)
        for p in sig.get("params", []) + sig.get("optional_params", []):
            if p["type"] == "tensor_desc":
                parts = p["source"].split(".")
                tid = resolver.find_tensor_id(parts[1])
                if tid and tid not in seen:
                    seen.add(tid)
                    result.append(tid)
    return result


# ---- npu_op → 简短前缀 ----
_OP_SHORT = {
    "cube_matmul": "mm", "cube_matmul_bias": "mm_bias",
    "vector_add": "add", "vector_mul": "mul", "vector_mul_scalar": "mul_s",
    "vector_gelu": "gelu", "vector_softmax": "softmax",
    "vector_layernorm": "layernorm",
    "vector_layernorm_part1": "layernorm_p1", "vector_layernorm_part2": "layernorm_p2",
    "vector_softmax_part1": "softmax_p1", "vector_softmax_part2": "softmax_p2",
    "vector_transpose": "trans", "vector_transpose_2d": "trans2d",
    "scalar_reshape": "reshape", "scalar_broadcast": "bcast", "scalar_copy": "copy",
    "dma_reformat": "reformat",
}


def _make_c_name(tid: str, t: dict, nodes: dict) -> str:
    """根据 tensor 元数据生成有语义的 C 变量名。"""
    # 权重：用 state_dict key 缩写
    if t.get("name"):
        name = t["name"]
        name = name.replace("layers.", "l").replace("self_attn.", "sa_")
        return name.replace(".", "_")
    # 模型输入
    if t.get("is_model_input"):
        return f"in_{tid.split('_', 1)[1]}" if "_" in tid else f"in_{tid}"
    # 模型输出
    if t.get("is_model_output"):
        return f"out_{tid.split('_', 1)[1]}" if "_" in tid else f"out_{tid}"
    # 中间结果：用 producer 算子类型 + 节点编号
    producer_id = t.get("producer_node_id")
    if producer_id and producer_id in nodes:
        node = nodes[producer_id]
        npu_op = node.get("npu_op", "unknown")
        short = _OP_SHORT.get(npu_op, npu_op)
        # 从 producer_id 提取编号（node_2 → 2, reformat_t_34 → t34）
        parts = producer_id.split("_", 1)
        num = parts[1] if len(parts) > 1 else parts[0]
        return f"{short}_{num}"
    return tid


def _build_c_name_map(tensor_ids: list[str], tensors: dict, nodes: dict) -> dict[str, str]:
    """构建 tid → C 变量名 映射（保证唯一性）。"""
    name_map: dict[str, str] = {}
    used: set[str] = set()
    for tid in tensor_ids:
        t = tensors.get(tid, {})
        c_name = _make_c_name(tid, t, nodes)
        # 去重：加后缀
        base = c_name
        suffix = 2
        while c_name in used:
            c_name = f"{base}_{suffix}"
            suffix += 1
        used.add(c_name)
        name_map[tid] = c_name
    return name_map


def _partition_tensors(
    tensor_ids: list[str],
    tensors: dict,
    nodes: dict,
    signatures: dict,
    node_group: dict[str, str],
    group_order: list[str],
    group_nids: dict[str, list[str]],
) -> list[tuple[str, list[str]]]:
    """将 tensor ID 分区为子结构体段：inputs / weights / 各函数组 / outputs。

    中间 tensor 按首次被引用的函数组归类（而非 producer 节点的组）。
    """
    # 构建 tid → 首次引用的函数组
    tid_first_group: dict[str, str] = {}
    for group in group_order:
        nids = group_nids[group]
        used = set(_collect_func_tensor_ids(nids, nodes, tensors, signatures))
        for tid in used:
            if tid not in tid_first_group:
                tid_first_group[tid] = group

    # 固定顺序：inputs → weights → 各函数组(按执行序) → outputs
    buckets: dict[str, list[str]] = {
        "inputs": [], "weights": [],
        **{g: [] for g in group_order},
        "outputs": [],
    }
    for tid in tensor_ids:
        t = tensors.get(tid, {})
        if t.get("is_model_input"):
            buckets["inputs"].append(tid)
        elif t.get("is_weight"):
            buckets["weights"].append(tid)
        elif t.get("is_model_output"):
            buckets["outputs"].append(tid)
        else:
            group = tid_first_group.get(tid, group_order[0] if group_order else "compute")
            buckets[group].append(tid)

    return [(s, tids) for s, tids in buckets.items() if tids]


def _section_to_c_field(section: str) -> str:
    """将 section 名转为 C 结构体字段名。"""
    if section in ("inputs", "weights", "outputs"):
        return section
    return _group_name_to_c_func(section)


def _build_sectioned_c_names(
    sections: list[tuple[str, list[str]]],
    tensors: dict,
    nodes: dict,
) -> dict[str, str]:
    """构建 tid → 'section.field' 访问路径映射。"""
    name_map: dict[str, str] = {}
    for section_name, tids in sections:
        c_section = _section_to_c_field(section_name)
        used: set[str] = set()
        for tid in tids:
            t = tensors.get(tid, {})
            field = _make_c_name(tid, t, nodes)
            base = field
            suffix = 2
            while field in used:
                field = f"{base}_{suffix}"
                suffix += 1
            used.add(field)
            name_map[tid] = f"{c_section}.{field}"
    return name_map


def _gen_tensor_struct_typedef(
    sections: list[tuple[str, list[str]]], c_names: dict[str, str],
) -> str:
    """生成带子结构体的 model_tensors_t 类型定义。"""
    lines = ["typedef struct {"]
    for section_name, tids in sections:
        c_section = _section_to_c_field(section_name)
        lines.append(f"    struct {{")
        for tid in tids:
            # c_names[tid] = "section.field", 取 field 部分
            full = c_names.get(tid, tid)
            field = full.split(".", 1)[1] if "." in full else full
            lines.append(f"        npu_tensor_t {field};")
        lines.append(f"    }} {c_section};")
    lines.append("} model_tensors_t;")
    return "\n".join(lines)


def _collect_spec_macros(
    sections: list[tuple[str, list[str]]], tensors: dict,
) -> dict[tuple[str, str], str]:
    """收集所有 (dtype_enum, format_enum) 组合，生成宏名映射。"""
    specs: set[tuple[str, str]] = set()
    for _, tids in sections:
        for tid in tids:
            t = tensors.get(tid, {})
            dtype_enum = DTYPE_C_ENUM_MAP.get(t.get("dtype", "fp16"), "NPU_DTYPE_FP16")
            fmt_enum = FORMAT_MAP.get(t.get("format", "nd"), "NPU_FORMAT_ND")
            specs.add((dtype_enum, fmt_enum))
    # NPU_DTYPE_FP16 → FP16, NPU_FORMAT_NZ → NZ
    result: dict[tuple[str, str], str] = {}
    for dtype_enum, fmt_enum in sorted(specs):
        d = dtype_enum.replace("NPU_DTYPE_", "")
        f = fmt_enum.replace("NPU_FORMAT_", "")
        result[(dtype_enum, fmt_enum)] = f"T_{d}_{f}"
    return result


def _gen_spec_macro_defs(macros: dict[tuple[str, str], str]) -> str:
    """生成 tensor spec 宏定义。"""
    lines = []
    for (dtype_enum, fmt_enum), macro_name in sorted(macros.items(), key=lambda x: x[1]):
        lines.append(
            f"#define {macro_name}(base, off)  "
            f"{{(base) + (off), {dtype_enum}, {fmt_enum}}}"
        )
    return "\n".join(lines)


def _gen_tensor_struct_init(
    sections: list[tuple[str, list[str]]],
    tensors: dict,
    c_names: dict[str, str],
    spec_macros: dict[tuple[str, str], str],
) -> str:
    """生成 model_tensors_t 的嵌套 designated initializer（使用 spec 宏）。"""
    lines = ["    model_tensors_t t = {"]
    for section_name, tids in sections:
        c_section = _section_to_c_field(section_name)
        lines.append(f"        .{c_section} = {{")
        for tid in tids:
            t = tensors.get(tid, {})
            offset = t.get("l1_offset", 0) or 0
            dtype_enum = DTYPE_C_ENUM_MAP.get(t.get("dtype", "fp16"), "NPU_DTYPE_FP16")
            fmt_enum = FORMAT_MAP.get(t.get("format", "nd"), "NPU_FORMAT_ND")
            full = c_names.get(tid, tid)
            field = full.split(".", 1)[1] if "." in full else full
            macro = spec_macros.get((dtype_enum, fmt_enum))
            if macro:
                lines.append(f"            .{field} = {macro}(l1, {offset}),")
            else:
                lines.append(
                    f"            .{field} = {{l1 + {offset}, {dtype_enum}, {fmt_enum}}},"
                )
        lines.append(f"        }},")
    lines.append("    };")
    return "\n".join(lines)


def _extract_model_dims(tensors: dict) -> dict[str, int]:
    """从 tensor shape 推断模型维度常量。

    返回 {宏名: 值}，如 {"BATCH": 1, "SEQ_LEN": 32, "D_MODEL": 256, "DIM_FF": 512}。
    """
    dims: dict[str, int] = {}
    for t in tensors.values():
        shape = t.get("shape", [])
        if t.get("is_model_input") and not dims.get("BATCH"):
            # 输入形如 [batch, seq_len, d_model]
            if len(shape) == 3:
                dims["BATCH"] = shape[0]
                dims["SEQ_LEN"] = shape[1]
                dims["D_MODEL"] = shape[2]
            elif len(shape) == 2:
                dims["BATCH"] = shape[0]
                dims["SEQ_LEN"] = shape[1]
        if t.get("is_weight") and t.get("name", ""):
            name = t["name"]
            # linear1.weight → [dim_ff, d_model] or NZ equivalent
            if "linear1.weight" in name and len(shape) >= 2:
                # 取最大维度作为 dim_ff
                dim_ff = max(shape)
                if dim_ff != dims.get("D_MODEL"):
                    dims["DIM_FF"] = dim_ff
    return dims


def _gen_dim_macros(dims: dict[str, int]) -> str:
    """生成模型维度宏定义。"""
    if not dims:
        return ""
    lines = ["/* Model dimension constants */"]
    for name, val in dims.items():
        lines.append(f"#define {name}  {val}")
    return "\n".join(lines)


def _build_dim_replace_map(dims: dict[str, int]) -> dict[int, str]:
    """构建 值→宏名 替换表（仅对有歧义的值去重，保留唯一映射）。"""
    val_to_names: dict[int, list[str]] = {}
    for name, val in dims.items():
        val_to_names.setdefault(val, []).append(name)
    # 只保留值唯一对应一个宏名的映射
    result: dict[int, str] = {}
    for val, names in val_to_names.items():
        if len(names) == 1:
            result[val] = names[0]
    return result


def _resolve_module_groups(nodes: dict, order: list[str]) -> dict[str, str]:
    """将每个节点的 module_path 映射到分组名。

    规则：叶子模块（无子模块）合并到父模块，非叶子保留。
    无 module_path 的节点（reformat 等）继承前后邻居。
    """
    # 收集所有 module_path
    all_paths: set[str] = set()
    for nid in order:
        mp = nodes[nid].get("module_path")
        if mp:
            all_paths.add(mp)
    if not all_paths:
        return {}

    # 叶子节点合并到父：如果没有任何其他路径以 path+"." 开头，就是叶子
    path_to_group: dict[str, str] = {}
    for path in all_paths:
        has_children = any(p.startswith(path + ".") for p in all_paths if p != path)
        if has_children:
            path_to_group[path] = path
        else:
            parent = path.rsplit(".", 1)[0] if "." in path else path
            path_to_group[path] = parent

    # 为每个节点分配组
    node_group: dict[str, str] = {}
    for nid in order:
        mp = nodes[nid].get("module_path")
        if mp:
            node_group[nid] = path_to_group.get(mp, mp)

    # 无 module_path 的节点继承最近邻居
    prev_group = None
    for nid in order:
        if nid in node_group:
            prev_group = node_group[nid]
        elif prev_group:
            node_group[nid] = prev_group

    return node_group


def _group_name_to_c_func(group: str) -> str:
    """将 module_path 组名转为 C 函数名。"""
    name = group.replace("layers.", "layer").replace("self_attn", "self_attn")
    return name.replace(".", "_")


def _collect_func_tensor_ids(
    nids: list[str], nodes: dict, tensors: dict, signatures: dict,
) -> list[str]:
    """收集一组节点引用的所有 tensor ID。"""
    seen: set[str] = set()
    result: list[str] = []
    for nid in nids:
        node = nodes[nid]
        npu_op = node.get("npu_op", "unknown")
        sig = signatures.get("compute_ops", {}).get(npu_op)
        if sig is None:
            continue
        resolver = SourceResolver(node, tensors)
        for p in sig.get("params", []) + sig.get("optional_params", []):
            if p["type"] == "tensor_desc":
                parts = p["source"].split(".")
                tid = resolver.find_tensor_id(parts[1])
                if tid and tid not in seen:
                    seen.add(tid)
                    result.append(tid)
    return result


def emit_model_graph_c(plan: dict, signatures: dict) -> str:
    """生成 model_graph.c 完整内容。"""
    nodes = plan["nodes"]
    tensors = plan["tensors"]
    dma_plans = {dp["node_id"]: dp for dp in plan.get("dma_plans", [])}
    order = plan.get("execution_order", list(nodes.keys()))

    # 分组：module_path → 函数
    node_group = _resolve_module_groups(nodes, order)

    # 如果没有模块信息，退化为平铺
    if not node_group:
        used_tids = _collect_used_tensor_ids(plan, signatures)
        c_names = _build_c_name_map(used_tids, tensors, nodes)
        return _emit_flat(plan, signatures, dma_plans, order, nodes, tensors, c_names)

    # 按组收集节点（保持执行顺序）
    group_order: list[str] = []
    group_nids: dict[str, list[str]] = {}
    for nid in order:
        g = node_group.get(nid, "__ungrouped__")
        if g not in group_nids:
            group_order.append(g)
            group_nids[g] = []
        group_nids[g].append(nid)

    # 子结构体分区 + 带路径的命名
    used_tids = _collect_used_tensor_ids(plan, signatures)
    sections = _partition_tensors(
        used_tids, tensors, nodes, signatures, node_group, group_order, group_nids,
    )
    c_names = _build_sectioned_c_names(sections, tensors, nodes)

    # 宏定义：tensor spec + 模型维度
    spec_macros = _collect_spec_macros(sections, tensors)
    model_dims = _extract_model_dims(tensors)
    dim_replace = _build_dim_replace_map(model_dims)

    # 结构体类型定义
    struct_typedef = _gen_tensor_struct_typedef(sections, c_names)

    # 为每组生成 static void 函数（通过结构体指针访问 tensor）
    prefix = "t->"
    func_sections: list[str] = []
    func_calls: list[str] = []
    indent = "    "
    for group in group_order:
        nids = group_nids[group]
        func_name = _group_name_to_c_func(group)
        op_parts = []
        for nid in nids:
            dp = dma_plans.get(nid, {"loads": [], "stores": []})
            op_parts.append(
                gen_op_block(nodes[nid], tensors, dp, signatures, c_names, prefix, dim_replace)
            )
        op_body = "\n\n".join(p for p in op_parts if p)
        func_sections.append(
            f"static void {func_name}(model_tensors_t* t) {{\n"
            f"{op_body}\n"
            f"}}"
        )
        func_calls.append(f"{indent}{func_name}(&t);")

    # model_run: 结构体初始化 + DMA + 函数调用
    main_parts = []
    main_parts.append(_gen_tensor_struct_init(sections, tensors, c_names, spec_macros))
    bulk_load = dma_plans.get("__bulk_load__")
    if bulk_load:
        main_parts.append(_gen_bulk_dma(bulk_load, "Bulk DMA Load"))
    main_parts.append("\n".join(func_calls))
    bulk_store = dma_plans.get("__bulk_store__")
    if bulk_store:
        main_parts.append(_gen_bulk_dma(bulk_store, "Bulk DMA Store"))
    main_body = "\n\n".join(p for p in main_parts if p)

    # 组装文件
    macro_defs = _gen_spec_macro_defs(spec_macros)
    dim_defs = _gen_dim_macros(model_dims)
    all_macros = "\n\n".join(m for m in [dim_defs, macro_defs] if m)

    funcs_code = "\n\n\n".join(func_sections)
    return (
        '#include "model_graph.h"\n'
        '#include "model_memory.h"\n'
        '#include "model_weights.h"\n'
        '#include "npu_mock.h"\n\n\n'
        f"{all_macros}\n\n\n"
        f"{struct_typedef}\n\n\n"
        f"{funcs_code}\n\n\n"
        "void model_run(unsigned char* hbm, unsigned char* l1) {\n"
        f"{main_body}\n"
        "}\n"
    )


def _emit_flat(plan, signatures, dma_plans, order, nodes, tensors, c_names):
    """无模块信息时退化为平铺模式（同样使用结构体）。"""
    used_tids = _collect_used_tensor_ids(plan, signatures)
    sections = [("flat", used_tids)]
    # 为 flat 模式给 c_names 添加 "flat." 前缀
    flat_c_names = {tid: f"flat.{c_names.get(tid, tid)}" for tid in used_tids}
    spec_macros = _collect_spec_macros(sections, tensors)
    struct_typedef = _gen_tensor_struct_typedef(sections, flat_c_names)
    struct_init = _gen_tensor_struct_init(sections, tensors, flat_c_names, spec_macros)
    prefix = "t."
    parts = []
    bulk_load = dma_plans.get("__bulk_load__")
    if bulk_load:
        parts.append(_gen_bulk_dma(bulk_load, "Bulk DMA Load"))
    for nid in order:
        dp = dma_plans.get(nid, {"loads": [], "stores": []})
        parts.append(gen_op_block(nodes[nid], tensors, dp, signatures, flat_c_names, prefix))
    bulk_store = dma_plans.get("__bulk_store__")
    if bulk_store:
        parts.append(_gen_bulk_dma(bulk_store, "Bulk DMA Store"))
    body = "\n\n".join(p for p in parts if p)
    spec_macro_defs = _gen_spec_macro_defs(spec_macros)
    all_macros = spec_macro_defs if spec_macro_defs else ""
    return (
        '#include "model_graph.h"\n'
        '#include "model_memory.h"\n'
        '#include "model_weights.h"\n'
        '#include "npu_mock.h"\n\n\n'
        f"{all_macros}\n\n"
        f"{struct_typedef}\n\n"
        "void model_run(unsigned char* hbm, unsigned char* l1) {\n"
        f"{struct_init}\n\n"
        f"{body}\n"
        "}\n"
    )


def emit_model_graph_h() -> str:
    return c_header_guard(
        "MODEL_GRAPH_H",
        "void model_run(unsigned char* hbm, unsigned char* l1);\n",
    )


def emit_model_memory_h(plan: dict) -> str:
    """生成 model_memory.h — 所有 tensor 的 HBM/L1 偏移宏。"""
    lines = []
    for tid, t in plan["tensors"].items():
        safe = tid.upper()
        hbm = t.get("hbm_offset", 0) or 0
        hbm_sz = t.get("hbm_size", 0) or 0
        l1 = t.get("l1_offset", 0) or 0
        lines.append(f"#define {safe}_HBM_OFFSET  {hbm}")
        lines.append(f"#define {safe}_HBM_SIZE    {hbm_sz}")
        lines.append(f"#define {safe}_L1_OFFSET   {l1}")
    return c_header_guard("MODEL_MEMORY_H", "\n".join(lines) + "\n")


def emit_model_params_h(plan: dict) -> str:
    """生成 model_params.h — 各算子的参数宏。"""
    lines = []
    for nid, node in plan["nodes"].items():
        for k, v in node.get("params", {}).items():
            safe = f"{nid}_{k}".upper()
            lines.append(f"#define {safe}  {v}")
    return c_header_guard("MODEL_PARAMS_H", "\n".join(lines) + "\n")


def run(plan: dict, output_dir: str, config_dir: str | None = None) -> None:
    """生成 model_graph.c/h, model_memory.h, model_params.h。"""
    logger.info("c_emitter: 开始生成 model_graph 文件")
    sigs = load_signatures(config_dir)
    src_dir = os.path.join(output_dir, "src")
    write_files(
        src_dir,
        [
            ("model_graph.c", emit_model_graph_c(plan, sigs)),
            ("model_graph.h", emit_model_graph_h()),
            ("model_memory.h", emit_model_memory_h(plan)),
            ("model_params.h", emit_model_params_h(plan)),
        ],
    )
