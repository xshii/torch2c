"""tid_assign pass 单元测试。"""

from __future__ import annotations

from torch2c.common.graph_ir import (
    DmaInstruction, DmaPlan, Graph, Node, Tensor,
)
from torch2c.d_emission.tid_assign import run


def _make_instruction(op: str, tid: str, hbm: int = 0, l1: int = 0, size: int = 4096) -> DmaInstruction:
    return DmaInstruction(
        op=op, tensor_id=tid, hbm_offset=hbm, l1_offset=l1,
        size_bytes=size, src_format="nd", dst_format="nd",
    )


def _make_linear_graph():
    """3 节点线性链 + per-op DMA plans。"""
    g = Graph()
    g.add_tensor(Tensor(id="t0", shape=[1, 32, 64], dtype="fp16",
                        is_model_input=True, consumer_node_ids=["n0"]))
    g.add_tensor(Tensor(id="t1", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n0", consumer_node_ids=["n1"]))
    g.add_tensor(Tensor(id="t2", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n1", consumer_node_ids=["n2"]))
    g.add_tensor(Tensor(id="t3", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n2", is_model_output=True))

    for i, (inp, out, unit) in enumerate([
        ("t0", "t1", "cube"),
        ("t1", "t2", "vector"),
        ("t2", "t3", "cube"),
    ]):
        n = Node(id=f"n{i}", op_type="test", inputs=[inp], outputs=[out],
                 compute_unit=unit, npu_op="test_op", is_mapped=True)
        # 设置 scheduler 风格的 dependencies
        n.dependencies = [f"n{i-1}"] if i > 0 else []
        g.add_node(n)

    g.execution_order = ["n0", "n1", "n2"]
    g.dma_plans = [
        DmaPlan(node_id="n0", loads=[_make_instruction("load", "t0")]),
        DmaPlan(node_id="n1", loads=[_make_instruction("load", "t1")],
                stores=[_make_instruction("store", "t1_s")]),
        DmaPlan(node_id="n2", loads=[_make_instruction("load", "t2")],
                stores=[_make_instruction("store", "t3")]),
    ]
    return g


def _make_branching_graph():
    """DAG 带分支:
        n0 (cube) → n1 (vector) → n3 (cube)
                  → n2 (vector) ↗
    n0 的输出 t1 被 n1 和 n2 同时消费，n3 依赖 n1 和 n2。
    主路径应为 n0→n1→n3 或 n0→n2→n3（长度相同，取一条）。
    """
    g = Graph()
    g.add_tensor(Tensor(id="t0", shape=[1, 32, 64], dtype="fp16",
                        is_model_input=True, consumer_node_ids=["n0"]))
    g.add_tensor(Tensor(id="t1", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n0", consumer_node_ids=["n1", "n2"]))
    g.add_tensor(Tensor(id="t2", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n1", consumer_node_ids=["n3"]))
    g.add_tensor(Tensor(id="t3", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n2", consumer_node_ids=["n3"]))
    g.add_tensor(Tensor(id="t4", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n3", is_model_output=True))

    n0 = Node(id="n0", op_type="test", inputs=["t0"], outputs=["t1"],
              compute_unit="cube", npu_op="test", is_mapped=True)
    n0.dependencies = []
    n1 = Node(id="n1", op_type="test", inputs=["t1"], outputs=["t2"],
              compute_unit="vector", npu_op="test", is_mapped=True)
    n1.dependencies = ["n0"]
    n2 = Node(id="n2", op_type="test", inputs=["t1"], outputs=["t3"],
              compute_unit="vector", npu_op="test", is_mapped=True)
    n2.dependencies = ["n0"]
    n3 = Node(id="n3", op_type="test", inputs=["t2", "t3"], outputs=["t4"],
              compute_unit="cube", npu_op="test", is_mapped=True)
    n3.dependencies = ["n1", "n2"]

    for n in [n0, n1, n2, n3]:
        g.add_node(n)
    g.execution_order = ["n0", "n1", "n2", "n3"]

    g.dma_plans = [
        DmaPlan(node_id="n0", loads=[_make_instruction("load", "t0")]),
        DmaPlan(node_id="n1"),
        DmaPlan(node_id="n2"),
        DmaPlan(node_id="n3", stores=[_make_instruction("store", "t4")]),
    ]
    return g


class TestTidMonotonic:
    def test_linear_tids_increase(self):
        """线性链 TID 严格递增。"""
        g = run(_make_linear_graph())
        all_tids = _collect_all_tids(g)
        assert all_tids == sorted(all_tids)
        assert len(all_tids) == len(set(all_tids))  # unique

    def test_branching_tids_increase(self):
        """分支 DAG TID 严格递增。"""
        g = run(_make_branching_graph())
        all_tids = _collect_all_tids(g)
        assert all_tids == sorted(all_tids)
        assert len(all_tids) == len(set(all_tids))


class TestComputeMustHaveDeps:
    def test_linear_all_compute_have_deps(self):
        """线性链：除首节点外所有 compute 有依赖。"""
        g = run(_make_linear_graph())
        for i, nid in enumerate(g.execution_order):
            node = g.nodes[nid]
            deps = [node.params.get(f"_tid_dep_{u}", 0)
                    for u in ("cube", "vector", "dma", "idma")]
            if i == 0 and not any(p.loads for p in g.dma_plans if p.node_id == nid):
                continue  # 首节点无 DMA load 可以无依赖
            assert any(d > 0 for d in deps), f"{nid} 应有依赖，但 deps={deps}"

    def test_branching_all_compute_have_deps(self):
        """分支 DAG：除首节点外所有 compute 有依赖。"""
        g = run(_make_branching_graph())
        for i, nid in enumerate(g.execution_order):
            if i == 0:
                continue
            node = g.nodes[nid]
            deps = [node.params.get(f"_tid_dep_{u}", 0)
                    for u in ("cube", "vector", "dma", "idma")]
            assert any(d > 0 for d in deps), f"{nid} 应有依赖，但 deps={deps}"


class TestBranchDependencyConstraint:
    def test_branch_only_depends_on_own_or_main(self):
        """支路节点只能依赖同支路或主路径节点。"""
        g = run(_make_branching_graph())
        from torch2c.d_emission.tid_assign.tid_assign import _find_critical_path, _classify_branches
        main_path = _find_critical_path(g)
        branch_map = _classify_branches(g, main_path)

        node_tid = {nid: g.nodes[nid].task_id for nid in g.execution_order}

        for nid in g.execution_order:
            if nid in main_path:
                continue
            my_branch = branch_map[nid]
            node = g.nodes[nid]
            # 收集依赖的 TID
            dep_tids = set()
            for u in ("cube", "vector", "dma", "idma"):
                v = node.params.get(f"_tid_dep_{u}", 0)
                if v > 0:
                    dep_tids.add(v)
            # 每个依赖的 TID 必须属于主路径或同支路
            for dep_tid in dep_tids:
                dep_nid = [n for n, t in node_tid.items() if t == dep_tid]
                if dep_nid:
                    dep_branch = branch_map.get(dep_nid[0], -1)
                    assert dep_branch == 0 or dep_branch == my_branch, (
                        f"{nid}(branch={my_branch}) 依赖 {dep_nid[0]}(branch={dep_branch})"
                    )


class TestCriticalPath:
    def test_linear_all_main_path(self):
        """线性链：全部节点都在主路径。"""
        g = _make_linear_graph()
        from torch2c.d_emission.tid_assign.tid_assign import _find_critical_path
        # 设置 dependencies
        for i, nid in enumerate(g.execution_order):
            g.nodes[nid].dependencies = [g.execution_order[i-1]] if i > 0 else []
        path = _find_critical_path(g)
        assert len(path) == 3

    def test_branching_path_length(self):
        """分支 DAG：主路径应为 3 节点（n0→n1→n3 或 n0→n2→n3）。"""
        g = _make_branching_graph()
        from torch2c.d_emission.tid_assign.tid_assign import _find_critical_path
        path = _find_critical_path(g)
        assert len(path) == 3  # 最长路径 3 节点


class TestDmaTid:
    def test_dma_has_nonzero_tid(self):
        """DMA 指令有非零 TID。"""
        g = run(_make_linear_graph())
        for plan in g.dma_plans:
            for instr in plan.loads + plan.stores:
                assert instr.task_id > 0


class TestBulk:
    def test_bulk_chain(self):
        """Bulk 模式正常工作。"""
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1, 4, 4], dtype="fp16",
                            is_model_input=True, consumer_node_ids=["n0"]))
        g.add_tensor(Tensor(id="t1", shape=[1, 4, 4], dtype="fp16",
                            producer_node_id="n0", is_model_output=True))
        n = Node(id="n0", op_type="test", inputs=["t0"], outputs=["t1"],
                 compute_unit="vector", npu_op="test", is_mapped=True)
        n.dependencies = []
        g.add_node(n)
        g.execution_order = ["n0"]
        g.dma_plans = [
            DmaPlan(node_id="__bulk_load__",
                    loads=[_make_instruction("load", "t0")]),
            DmaPlan(node_id="__bulk_store__",
                    stores=[_make_instruction("store", "t1")]),
        ]
        g = run(g)
        plan_map = {p.node_id: p for p in g.dma_plans}
        ld = plan_map["__bulk_load__"].loads[0]
        node = g.nodes["n0"]
        st = plan_map["__bulk_store__"].stores[0]
        assert ld.task_id == 1
        assert node.task_id == 2
        assert node.params["_tid_dep_dma"] == 1
        assert st.task_id == 3


class TestEmpty:
    def test_no_dma_plans(self):
        g = Graph()
        g.dma_plans = []
        g = run(g)
        assert g.dma_plans == []


def _collect_all_tids(g: Graph) -> list[int]:
    """按提交顺序收集所有 TID。"""
    plan_map = {p.node_id: p for p in g.dma_plans}
    tids = []
    for nid in g.execution_order:
        plan = plan_map.get(nid)
        if plan:
            for ld in plan.loads:
                tids.append(ld.task_id)
        tids.append(g.nodes[nid].task_id)
        if plan:
            for st in plan.stores:
                tids.append(st.task_id)
    # Also collect bulk
    for key in ("__bulk_load__", "__bulk_store__"):
        if key in plan_map:
            for instr in plan_map[key].loads + plan_map[key].stores:
                if instr.task_id not in tids:
                    tids.append(instr.task_id)
    return tids
