"""_naming — Tensor 命名、分区、模块分组。"""

from __future__ import annotations

from ._helpers import DTYPE_C_ENUM_MAP, FORMAT_MAP, find_op_sig  # noqa: F401


# ---- npu_op → 简短前缀 ----
_OP_SHORT = {
    "cube_matmul": "mm", "cube_matmul_bias": "mm_bias",
    "vector_add": "add", "vector_mul": "mul", "vector_mul_scalar": "mul_s",
    "vector_gelu": "gelu", "vector_softmax": "softmax",
    "vector_layernorm": "layernorm",
    "vector_layernorm_part1": "layernorm_p1", "vector_layernorm_part2": "layernorm_p2",
    "vector_softmax_part1": "softmax_p1", "vector_softmax_part2": "softmax_p2",
    "vector_transpose": "trans", "vector_transpose_2d": "trans2d",
    "idma_reshape": "reshape", "idma_broadcast": "bcast", "idma_move": "copy",
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

    # 无 module_path 的节点：按 output 的 consumer 所属组归类
    for nid in order:
        if nid in node_group:
            continue
        node = nodes[nid]
        for out_tid in node.get("outputs", []):
            for other_nid in order:
                if other_nid in node_group and out_tid in nodes[other_nid].get("inputs", []):
                    node_group[nid] = node_group[other_nid]
                    break
            if nid in node_group:
                break
    # 仍无组的节点继承前邻居
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
    from .c_emitter import SourceResolver

    seen: set[str] = set()
    result: list[str] = []
    for nid in nids:
        node = nodes[nid]
        sig = find_op_sig(signatures, node.get("npu_op", "unknown"))
        if sig is None:
            continue
        resolver = SourceResolver(node, tensors)
        for p in sig.get("params", []) + sig.get("optional_params", []):
            if p["type"] == "tensor_desc":
                tid = resolver.find_tensor_id(p["source"].split(".")[1])
                if tid and tid not in seen:
                    seen.add(tid)
                    result.append(tid)
    return result


def _collect_used_tensor_ids(plan: dict, signatures: dict) -> list[str]:
    """收集所有 compute op 引用的 tensor ID（去重、保序）。"""
    nodes = plan["nodes"]
    order = plan.get("execution_order", list(nodes.keys()))
    return _collect_func_tensor_ids(order, nodes, plan["tensors"], signatures)


def _build_groups(nodes, order, node_group):
    """按组收集节点，返回 (group_order, group_nids)。

    group_order 按拓扑序排列：若 A 组的某节点输出被 B 组消费，则 A 排在 B 前。
    """
    group_nids: dict[str, list[str]] = {}
    for nid in order:
        g = node_group.get(nid, "__ungrouped__")
        group_nids.setdefault(g, []).append(nid)

    # 构建 tensor → producer_group 映射
    tid_group: dict[str, str] = {}
    for g, nids in group_nids.items():
        for nid in nids:
            for out_tid in nodes[nid].get("outputs", []):
                tid_group[out_tid] = g

    # 构建组间依赖图
    group_deps: dict[str, set[str]] = {g: set() for g in group_nids}
    for g, nids in group_nids.items():
        for nid in nids:
            for in_tid in nodes[nid].get("inputs", []):
                pg = tid_group.get(in_tid)
                if pg and pg != g:
                    group_deps[g].add(pg)

    # 拓扑排序
    sorted_groups: list[str] = []
    visited: set[str] = set()

    def visit(g: str) -> None:
        if g in visited:
            return
        visited.add(g)
        for dep in group_deps.get(g, set()):
            visit(dep)
        sorted_groups.append(g)

    for g in group_nids:
        visit(g)
    return sorted_groups, group_nids
