"""scheduler 单元测试。"""

from torch2c.common import Graph, Node, Tensor
from torch2c.common.testing import make_linear_chain
from torch2c.scheduler.scheduler import post_validate, run

_SCHEDULER_OPS = [
    ("node_0", "cube_matmul", "cube"),
    ("node_1", "vector_add", "vector"),
    ("node_2", "cube_matmul", "cube"),
    ("node_3", "vector_gelu", "vector"),
]


def _make_linear_chain() -> Graph:
    """创建 matmul→add→matmul→gelu 线性链。"""
    return make_linear_chain(ops=_SCHEDULER_OPS)


def _make_parallel_graph() -> Graph:
    """创建两个无数据依赖、不同 compute_unit 的算子。

    input_a → node_a (cube)
    input_b → node_b (vector)
    两者无数据依赖。
    """
    g = Graph()
    g.add_tensor(
        Tensor(
            id="input_a",
            shape=[1, 32, 64],
            dtype="fp16",
            is_model_input=True,
            consumer_node_ids=["node_a"],
        )
    )
    g.add_tensor(
        Tensor(
            id="input_b",
            shape=[1, 32, 64],
            dtype="fp16",
            is_model_input=True,
            consumer_node_ids=["node_b"],
        )
    )
    g.add_tensor(
        Tensor(
            id="out_a",
            shape=[1, 32, 64],
            dtype="fp16",
            producer_node_id="node_a",
            is_model_output=True,
        )
    )
    g.add_tensor(
        Tensor(
            id="out_b",
            shape=[1, 32, 64],
            dtype="fp16",
            producer_node_id="node_b",
            is_model_output=True,
        )
    )
    g.add_node(
        Node(
            id="node_a",
            op_type="cube_matmul",
            inputs=["input_a"],
            outputs=["out_a"],
            compute_unit="cube",
            npu_op="cube_matmul",
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="node_b",
            op_type="vector_add",
            inputs=["input_b"],
            outputs=["out_b"],
            compute_unit="vector",
            npu_op="vector_add",
            is_mapped=True,
        )
    )
    g.execution_order = ["node_a", "node_b"]
    return g


def _make_same_unit_graph() -> Graph:
    """创建两个无数据依赖、相同 compute_unit 的算子。"""
    g = Graph()
    g.add_tensor(
        Tensor(
            id="input_a",
            shape=[1, 32, 64],
            dtype="fp16",
            is_model_input=True,
            consumer_node_ids=["node_a"],
        )
    )
    g.add_tensor(
        Tensor(
            id="input_b",
            shape=[1, 32, 64],
            dtype="fp16",
            is_model_input=True,
            consumer_node_ids=["node_b"],
        )
    )
    g.add_tensor(
        Tensor(
            id="out_a",
            shape=[1, 32, 64],
            dtype="fp16",
            producer_node_id="node_a",
            is_model_output=True,
        )
    )
    g.add_tensor(
        Tensor(
            id="out_b",
            shape=[1, 32, 64],
            dtype="fp16",
            producer_node_id="node_b",
            is_model_output=True,
        )
    )
    g.add_node(
        Node(
            id="node_a",
            op_type="vector_add",
            inputs=["input_a"],
            outputs=["out_a"],
            compute_unit="vector",
            npu_op="vector_add",
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="node_b",
            op_type="vector_gelu",
            inputs=["input_b"],
            outputs=["out_b"],
            compute_unit="vector",
            npu_op="vector_gelu",
            is_mapped=True,
        )
    )
    g.execution_order = ["node_a", "node_b"]
    return g


class TestLinearChain:
    def test_linear_chain(self):
        """线性依赖链的依赖关系正确。"""
        g = _make_linear_chain()
        g = run(g)

        # node_1 依赖 node_0，node_2 依赖 node_1，node_3 依赖 node_2
        assert "node_0" in g.nodes["node_1"].dependencies
        assert "node_1" in g.nodes["node_2"].dependencies
        assert "node_2" in g.nodes["node_3"].dependencies
        # node_0 无依赖
        assert len(g.nodes["node_0"].dependencies) == 0


class TestParallelOpportunity:
    def test_parallel_opportunity(self):
        """两个无依赖的不同 compute_unit 算子不产生依赖。"""
        g = _make_parallel_graph()
        g = run(g)

        # node_b 不依赖 node_a
        assert "node_a" not in g.nodes["node_b"].dependencies


class TestSameUnitNoSharedTensor:
    def test_same_unit_no_shared_parallel(self):
        """同 compute_unit 但无共享 tensor → 可并行（无结构冒险）。"""
        g = _make_same_unit_graph()
        g = run(g)

        # node_b 不依赖 node_a（无共享 tensor）
        assert "node_a" not in g.nodes["node_b"].dependencies

    def test_same_unit_shared_tensor_serialized(self):
        """同 compute_unit + 共享 tensor → 结构冒险，需串行。"""
        g = Graph()
        shared = Tensor(id="shared", shape=[1, 32, 64], dtype="fp16",
                        is_model_input=True, consumer_node_ids=["node_a", "node_b"])
        g.add_tensor(shared)
        g.add_tensor(Tensor(id="out_a", shape=[1, 32, 64], dtype="fp16",
                            producer_node_id="node_a", is_model_output=True))
        g.add_tensor(Tensor(id="out_b", shape=[1, 32, 64], dtype="fp16",
                            producer_node_id="node_b", is_model_output=True))
        g.add_node(Node(id="node_a", op_type="vector_add", inputs=["shared"],
                        outputs=["out_a"], compute_unit="vector",
                        npu_op="vector_add", is_mapped=True))
        g.add_node(Node(id="node_b", op_type="vector_gelu", inputs=["shared"],
                        outputs=["out_b"], compute_unit="vector",
                        npu_op="vector_gelu", is_mapped=True))
        g.execution_order = ["node_a", "node_b"]
        g = run(g)

        # node_b 应依赖 node_a（同 vector 单元 + 共享 shared tensor）
        assert "node_a" in g.nodes["node_b"].dependencies


class TestScheduleOrder:
    def test_schedule_order(self):
        """每个节点的 schedule_order 按拓扑排序递增。"""
        g = _make_linear_chain()
        g = run(g)

        raw_orders = [g.nodes[nid].schedule_order for nid in g.execution_order]
        assert all(o is not None for o in raw_orders)
        orders: list[int] = [o for o in raw_orders if o is not None]
        assert len(orders) == len(raw_orders)
        assert orders == sorted(orders)
        assert orders == list(range(len(orders)))


class TestAdjacentOnlyDeps:
    """回归测试：确保只产生相邻对的依赖，不产生传递依赖。"""

    def test_no_transitive_deps(self):
        """线性链 4 个算子只应有 3 条依赖（仅相邻对）。"""
        g = _make_linear_chain()
        g = run(g)

        total_deps = sum(len(n.dependencies) for n in g.nodes.values())
        assert total_deps == 3, f"期望 3 条依赖，实际 {total_deps}"

        # node_2 不应直接依赖 node_0（只依赖 node_1）
        assert "node_0" not in g.nodes["node_2"].dependencies
        # node_3 不应直接依赖 node_0 或 node_1
        assert "node_0" not in g.nodes["node_3"].dependencies
        assert "node_1" not in g.nodes["node_3"].dependencies


class TestTaskId:
    def test_task_id_assigned(self):
        """每个节点的 task_id = schedule_order + 1。"""
        g = _make_linear_chain()
        g = run(g)
        for nid in g.execution_order:
            node = g.nodes[nid]
            assert node.task_id == node.schedule_order + 1
            assert node.task_id > 0

    def test_task_id_unique(self):
        """task_id 全局唯一。"""
        g = _make_linear_chain()
        g = run(g)
        task_ids = [g.nodes[nid].task_id for nid in g.execution_order]
        assert len(task_ids) == len(set(task_ids))

    def test_tid_deps_in_params(self):
        """per-unit TidInfo deps 写入 params。"""
        g = _make_linear_chain()
        g = run(g)
        # node_1 (vector) depends on node_0 (cube, task_id=1)
        assert g.nodes["node_1"].params["_tid_dep_cube"] == 1
        assert g.nodes["node_1"].params["_tid_dep_vector"] == 0
        # node_0 has no deps
        assert g.nodes["node_0"].params["_tid_dep_cube"] == 0


class TestPostValidate:
    def test_post_validate_after_schedule(self):
        """调度后校验通过。"""
        g = _make_linear_chain()
        g = run(g)
        errors = post_validate(g)
        assert errors == []

    def test_post_validate_missing_schedule_order(self):
        """未调度的节点报错。"""
        g = _make_linear_chain()
        # 不运行 scheduler，直接校验
        errors = post_validate(g)
        assert len(errors) > 0
        assert any("schedule_order" in e for e in errors)
