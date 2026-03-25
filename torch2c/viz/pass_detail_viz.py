"""pass_detail_viz — 构建单个 Pass 的图数据（4 列 CU 泳道布局）。

数据构建在 Python 侧，渲染在 pipeline_viz.py 的内嵌 JS 中。
"""

from __future__ import annotations

import json

from torch2c.common import get_logger
from torch2c.common.graph_ir import graph_diff

logger = get_logger(__name__)


def build_graph_data(
    snapshot: dict,
    prev_snapshot: dict | None,
) -> dict:
    """构建图数据 + diff 标记，供前端渲染。"""
    nodes_data: list[dict] = []
    edges_data: list[dict] = []
    tensors_data: dict[str, dict] = {}

    diff = graph_diff(prev_snapshot, snapshot) if prev_snapshot else None
    added_n = set(diff["nodes_added"]) if diff else set()
    removed_n = set(diff["nodes_removed"]) if diff else set()
    changed_n = set(diff["nodes_changed"].keys()) if diff else set()

    # Tensor index
    for tid, td in snapshot.get("tensors", {}).items():
        tensors_data[tid] = {
            "id": tid, "shape": td.get("shape", []),
            "dtype": td.get("dtype", ""), "is_weight": td.get("is_weight", False),
        }

    # Nodes
    exec_order = snapshot.get("execution_order", [])
    order_map = {nid: i for i, nid in enumerate(exec_order)}

    for nid, nd in snapshot.get("nodes", {}).items():
        status = "added" if nid in added_n else (
            "changed" if nid in changed_n else "normal")
        opt_log = nd.get("params", {}).get("_opt_log", [])
        nodes_data.append({
            "id": nid,
            "npu_op": nd.get("npu_op", ""),
            "op_type": nd.get("op_type", ""),
            "compute_unit": nd.get("compute_unit", ""),
            "inputs": nd.get("inputs", []),
            "outputs": nd.get("outputs", []),
            "status": status,
            "order": order_map.get(nid, 999),
            "task_id": nd.get("task_id", 0),
            "schedule_order": nd.get("schedule_order"),
            "opt_log": opt_log,
        })

        # Edges
        for tid in nd.get("outputs", []):
            t = snapshot.get("tensors", {}).get(tid, {})
            for cid in t.get("consumer_node_ids", []):
                if cid in snapshot.get("nodes", {}):
                    edges_data.append({
                        "from": nid, "to": cid, "tensor": tid,
                        "shape": t.get("shape", []), "dtype": t.get("dtype", ""),
                    })

    # Removed nodes
    if prev_snapshot:
        for nid in removed_n:
            nd = prev_snapshot["nodes"].get(nid, {})
            nodes_data.append({
                "id": nid,
                "npu_op": nd.get("npu_op", ""),
                "op_type": nd.get("op_type", ""),
                "compute_unit": nd.get("compute_unit", ""),
                "inputs": [], "outputs": [],
                "status": "removed", "order": 9999,
            })

    # Inject DMA nodes from dma_plans (memory_planner 生成的搬运指令)
    # 构建 tensor→consumer 映射，用于 bulk load 连接
    tensor_consumers: dict[str, list[str]] = {}
    for nd in snapshot.get("nodes", {}).values():
        for tid in nd.get("inputs", []):
            tensor_consumers.setdefault(tid, []).append(nd["id"])
    # 构建 tensor→producer 映射，用于 bulk store 连接
    tensor_producers: dict[str, str] = {}
    for tid, td in snapshot.get("tensors", {}).items():
        pid = td.get("producer_node_id")
        if pid:
            tensor_producers[tid] = pid

    dma_plans = snapshot.get("dma_plans", [])
    compute_nodes = {n["id"] for n in nodes_data}
    for plan in dma_plans:
        plan_nid = plan.get("node_id", "")
        is_bulk = plan_nid.startswith("__bulk")
        loads = plan.get("loads", [])
        stores = plan.get("stores", [])

        # 统一处理：每个 load/store 按消费者/生产者位置放置
        for i, load in enumerate(loads):
            tid = load.get("tensor_id", "")
            size = load.get("size_bytes", 0)
            dma_nid = f"dma_ld_{tid}"
            consumers = tensor_consumers.get(tid, [])
            first_consumer = consumers[0] if consumers else None
            # 预取：DMA 和前一个计算节点并行，提前 1 步搬运
            consumer_order = order_map.get(first_consumer, 0) if first_consumer else 0
            dma_order = max(consumer_order - 1.5, -0.5)
            nodes_data.append({
                "id": dma_nid, "npu_op": "dma_load",
                "op_type": f"load ({size}B)", "compute_unit": "dma",
                "inputs": [], "outputs": [tid], "status": "normal",
                "order": dma_order, "task_id": 0, "schedule_order": None,
                "opt_log": [],
            })
            for cid in consumers:
                if cid in compute_nodes:
                    edges_data.append({
                        "from": dma_nid, "to": cid, "tensor": tid,
                        "shape": tensors_data.get(tid, {}).get("shape", []),
                        "dtype": tensors_data.get(tid, {}).get("dtype", ""),
                    })
        for i, store in enumerate(stores):
            tid = store.get("tensor_id", "")
            size = store.get("size_bytes", 0)
            dma_nid = f"dma_st_{tid}"
            producer = tensor_producers.get(tid)
            dma_order = order_map.get(producer, 999) + 0.5 if producer else 9998
            nodes_data.append({
                "id": dma_nid, "npu_op": "dma_store",
                "op_type": f"store ({size}B)", "compute_unit": "dma",
                "inputs": [tid], "outputs": [], "status": "normal",
                "order": dma_order, "task_id": 0, "schedule_order": None,
                "opt_log": [],
            })
            if producer and producer in compute_nodes:
                edges_data.append({
                    "from": producer, "to": dma_nid, "tensor": tid,
                    "shape": tensors_data.get(tid, {}).get("shape", []),
                    "dtype": tensors_data.get(tid, {}).get("dtype", ""),
                })

    nodes_data.sort(key=lambda n: n["order"])
    return {"nodes": nodes_data, "edges": edges_data, "tensors": tensors_data}
