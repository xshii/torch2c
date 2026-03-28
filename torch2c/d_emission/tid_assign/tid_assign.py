"""tid_assign — 全局 TID 重分配（含主路径/支路依赖约束）。

约束：
  1. 除 DMA 外，所有计算算子必须有依赖（dep 不全为 0）
  2. 单核提交顺序：TID 严格递增
  3. 最长路径为主路径（critical path）
  4. 非主路径算子只能依赖：当前支路算子 或 主路径算子
  5. DMA 指令关联到其计算算子，融入依赖链

算法：
  1. 从 scheduler 的 dependencies 构建 DAG
  2. DP 求最长路径 → main_path 节点集合
  3. 拓扑遍历，按 main-path 优先顺序分配 TID
  4. 按约束设置依赖
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

logger = get_logger("tid_assign")

_UNIT_DEP_KEY = {
    "cube": "dep_cube_tid",
    "vector": "dep_vector_tid",
    "dma": "dep_dma_tid",
    "idma": "dep_idma_tid",
}

_ZERO_DEPS = {k: 0 for k in _UNIT_DEP_KEY.values()}


# ── 依赖写入 ──

def _assign_dma_tid(instr, tid: int, deps: dict[str, int]) -> None:
    instr.task_id = tid
    instr.dep_cube_tid = deps["dep_cube_tid"]
    instr.dep_vector_tid = deps["dep_vector_tid"]
    instr.dep_dma_tid = deps["dep_dma_tid"]
    instr.dep_idma_tid = deps["dep_idma_tid"]


def _assign_node_tid(node, tid: int, deps: dict[str, int]) -> None:
    node.task_id = tid
    node.params["_tid_dep_cube"] = deps["dep_cube_tid"]
    node.params["_tid_dep_vector"] = deps["dep_vector_tid"]
    node.params["_tid_dep_dma"] = deps["dep_dma_tid"]
    node.params["_tid_dep_idma"] = deps["dep_idma_tid"]


def _make_dep(unit: str, tid: int) -> dict[str, int]:
    """单依赖：对 unit 设置 tid，其余为 0。"""
    deps = dict(_ZERO_DEPS)
    key = _UNIT_DEP_KEY.get(unit)
    if key and tid > 0:
        deps[key] = tid
    return deps


def _merge_deps(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """合并两组依赖：每个 unit 取 max。"""
    return {k: max(a.get(k, 0), b.get(k, 0)) for k in _ZERO_DEPS}


def _node_unit(node) -> str:
    return (node.compute_unit or "vector").lower()


# ── 主路径求解 ──

def _find_critical_path(graph: Graph) -> set[str]:
    """DP 求 DAG 最长路径（节点数），返回主路径节点 id 集合。"""
    order = graph.execution_order
    if not order:
        return set()

    # dist[nid] = 从该节点到终点的最长路径长度
    dist: dict[str, int] = {}
    succ: dict[str, str | None] = {}  # 最长路径上的下一个节点

    # 逆序 DP
    for nid in reversed(order):
        node = graph.nodes[nid]
        # 找所有后继（谁依赖我）
        best_len = 0
        best_next = None
        for other_nid in order:
            other = graph.nodes[other_nid]
            if nid in (other.dependencies or []):
                d = dist.get(other_nid, 1)
                if d > best_len:
                    best_len = d
                    best_next = other_nid
        dist[nid] = best_len + 1
        succ[nid] = best_next

    # 从 dist 最大的节点回溯
    start = max(order, key=lambda n: dist.get(n, 0))
    path = set()
    cur = start
    while cur is not None:
        path.add(cur)
        cur = succ[cur]

    logger.debug("主路径: %d / %d 节点", len(path), len(order))
    return path


def _classify_branches(graph: Graph, main_path: set[str]) -> dict[str, int]:
    """为每个非主路径节点分配支路 ID。

    同一条支路上的连续节点共享 branch_id。
    主路径节点 branch_id = 0。
    """
    branch_map: dict[str, int] = {}
    next_branch = 1

    for nid in graph.execution_order:
        if nid in main_path:
            branch_map[nid] = 0
            continue
        # 看依赖：如果有非主路径的依赖节点，继承其 branch_id
        node = graph.nodes[nid]
        inherited = None
        for dep_id in (node.dependencies or []):
            if dep_id not in main_path and dep_id in branch_map:
                inherited = branch_map[dep_id]
                break
        if inherited is not None:
            branch_map[nid] = inherited
        else:
            branch_map[nid] = next_branch
            next_branch += 1

    return branch_map


# ── TID 分配 ──

def run(graph: Graph, _config: dict | None = None) -> Graph:
    """重分配全局 TID，满足主路径/支路依赖约束。"""
    if not graph.dma_plans:
        logger.info("无 DMA 计划，跳过 TID 分配")
        return graph

    plan_map = {p.node_id: p for p in graph.dma_plans}
    is_bulk = "__bulk_load__" in plan_map

    if is_bulk:
        return _run_bulk(graph, plan_map)
    return _run_perop(graph, plan_map)


def _run_bulk(graph: Graph, plan_map: dict) -> Graph:
    """Bulk 模式：bulk_load → all computes（主路径优先）→ bulk_store。"""
    main_path = _find_critical_path(graph)
    branch_map = _classify_branches(graph, main_path)

    tid = 1
    prev_unit = ""
    prev_tid = 0

    # Bulk load
    bulk_load = plan_map.get("__bulk_load__")
    if bulk_load:
        for instr in bulk_load.loads:
            deps = _make_dep(prev_unit, prev_tid)
            _assign_dma_tid(instr, tid, deps)
            prev_unit, prev_tid = "dma", tid
            tid += 1

    # Compute ops — 用 execution_order（已是拓扑序）
    # 记录每个节点的 TID，供后续依赖查找
    node_tid: dict[str, int] = {}
    node_unit: dict[str, str] = {}

    for nid in graph.execution_order:
        node = graph.nodes[nid]
        deps = _compute_node_deps(
            node, main_path, branch_map, node_tid, node_unit,
            prev_unit, prev_tid, is_first=(tid == 1),
        )
        _assign_node_tid(node, tid, deps)
        node_tid[nid] = tid
        node_unit[nid] = _node_unit(node)
        prev_unit = _node_unit(node)
        prev_tid = tid
        tid += 1

    # Bulk store
    bulk_store = plan_map.get("__bulk_store__")
    if bulk_store:
        for instr in bulk_store.stores:
            deps = _make_dep(prev_unit, prev_tid)
            _assign_dma_tid(instr, tid, deps)
            prev_unit, prev_tid = "dma", tid
            tid += 1

    _log_stats(graph, main_path, branch_map, tid - 1)
    return graph


def _run_perop(graph: Graph, plan_map: dict) -> Graph:
    """Per-op 模式：loads → compute → stores，含主路径/支路约束。"""
    main_path = _find_critical_path(graph)
    branch_map = _classify_branches(graph, main_path)

    tid = 1
    prev_unit = ""
    prev_tid = 0
    node_tid: dict[str, int] = {}
    node_unit: dict[str, str] = {}
    # 记录每个节点最后一条 store 的 (tid, unit)，供后续节点的 load 依赖
    node_last_store: dict[str, tuple[int, str]] = {}

    for nid in graph.execution_order:
        node = graph.nodes[nid]
        plan = plan_map.get(nid)

        # ── DMA loads: 依赖前一条指令（串行提交）──
        if plan:
            for instr in plan.loads:
                deps = _make_dep(prev_unit, prev_tid)
                _assign_dma_tid(instr, tid, deps)
                prev_unit, prev_tid = "dma", tid
                tid += 1

        # ── Compute: 依赖按主路径/支路约束 ──
        is_first = (not node_tid)  # 第一个计算节点
        deps = _compute_node_deps(
            node, main_path, branch_map, node_tid, node_unit,
            prev_unit, prev_tid, is_first=is_first,
        )
        _assign_node_tid(node, tid, deps)
        node_tid[nid] = tid
        node_unit[nid] = _node_unit(node)
        prev_unit = _node_unit(node)
        prev_tid = tid
        tid += 1

        # ── DMA stores: 依赖 compute ──
        if plan:
            for instr in plan.stores:
                deps = _make_dep(prev_unit, prev_tid)
                _assign_dma_tid(instr, tid, deps)
                prev_unit, prev_tid = "dma", tid
                tid += 1
            if plan.stores:
                node_last_store[nid] = (prev_tid, prev_unit)

    _log_stats(graph, main_path, branch_map, tid - 1)
    return graph


def _compute_node_deps(
    node, main_path: set[str], branch_map: dict[str, int],
    node_tid: dict[str, int], node_unit_map: dict[str, str],
    prev_unit: str, prev_tid: int,
    is_first: bool,
) -> dict[str, int]:
    """为计算节点构建依赖，满足约束：

    - 非首节点必须有依赖
    - 主路径节点依赖：前一个主路径节点（如有），以及数据依赖的 DMA
    - 支路节点只能依赖：同支路节点 或 主路径节点
    """
    nid = node.id
    my_branch = branch_map.get(nid, 0)
    is_main = (nid in main_path)

    deps = dict(_ZERO_DEPS)

    # 1. 数据依赖（scheduler 已计算）— 过滤掉跨支路依赖
    for dep_id in (node.dependencies or []):
        dep_branch = branch_map.get(dep_id, 0)
        # 约束：只能依赖主路径(0) 或 同支路
        if dep_branch != 0 and dep_branch != my_branch:
            continue
        if dep_id in node_tid:
            dep_unit = node_unit_map[dep_id]
            deps = _merge_deps(deps, _make_dep(dep_unit, node_tid[dep_id]))

    # 2. 如果有前序 DMA load（prev_unit="dma"），compute 必须依赖它
    if prev_tid > 0 and prev_unit == "dma":
        deps = _merge_deps(deps, _make_dep("dma", prev_tid))

    # 3. 约束：非首节点必须有依赖
    if not is_first and all(v == 0 for v in deps.values()):
        # fallback: 依赖前一个提交的指令
        if prev_tid > 0:
            deps = _merge_deps(deps, _make_dep(prev_unit, prev_tid))

    return deps


def _log_stats(graph: Graph, main_path: set[str], branch_map: dict[str, int], total: int):
    n_branches = len(set(v for v in branch_map.values() if v > 0))
    logger.info(
        "TID 分配完成: %d 条指令, 主路径 %d 节点, %d 条支路",
        total, len(main_path), n_branches,
    )
