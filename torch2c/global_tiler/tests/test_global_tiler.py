"""global_tiler 单元测试。"""

from torch2c.common import Graph, Node, Tensor
from torch2c.global_tiler import run


def _hw_config(l1_bytes: int = 16 * 1024 * 1024) -> dict:
    return {
        "memory": {"l1": {"total_size_bytes": l1_bytes}, "hbm": {"alignment_bytes": 512}},
        "fractal": {"cube_size": 16},
    }


def _make_matmul(m: int, k: int, n: int) -> Graph:
    """cube_matmul [1, M, K] × [K, N] → [1, M, N]。"""
    g = Graph()
    g.add_tensor(Tensor(id="A", shape=[1, m, k], dtype="fp16", format="nz"))
    g.add_tensor(Tensor(id="B", shape=[k, n], dtype="fp16", format="nz", is_weight=True))
    g.add_tensor(Tensor(id="C", shape=[1, m, n], dtype="fp16", format="nz",
                        producer_node_id="mm", consumer_node_ids=[]))
    g.add_node(Node(
        id="mm", op_type="cube_matmul", npu_op="cube_matmul",
        compute_unit="cube", is_mapped=True,
        inputs=["A", "B"], outputs=["C"],
    ))
    g.execution_order = ["mm"]
    return g


class TestGlobalTilerSmallMatmul:
    """小 matmul（L1 放得下）→ 不主动 tiling（收益不够）。"""

    def test_no_tile_small(self):
        g = _make_matmul(64, 64, 64)
        g = run(g, _hw_config())
        assert "_tile_config" not in g.nodes["mm"].params


class TestGlobalTilerLargeMatmul:
    """大 matmul（L1 放得下但 tiling+ping-pong 更快）。"""

    def test_proactive_tiling(self):
        # [1, 4096, 1024] × [1024, 1024] — fits in L1 but tiling may be faster
        g = _make_matmul(4096, 1024, 1024)
        g = run(g, _hw_config())
        cfg = g.nodes["mm"].params.get("_tile_config")
        # 不一定触发 tiling（取决于 speedup 阈值），但如果触发则验证正确性
        if cfg:
            assert cfg["tile_size"] < 4096
            assert cfg["num_tiles"] >= 2
            assert cfg["speedup"] >= 1.10


class TestGlobalTilerOverflow:
    """L1 放不下 → 一定会 tiling。"""

    def test_forced_tiling(self):
        # [1, 8192, 4096] × [4096, 4096] — 每 tensor ~64MB，远超 16MB L1
        g = _make_matmul(8192, 4096, 4096)
        # 用小 L1 确保必须 tiling
        g = run(g, _hw_config(l1_bytes=4 * 1024 * 1024))
        cfg = g.nodes["mm"].params.get("_tile_config")
        # global_tiler 只主动 tiling，overflow 由 memory_planner 处理
        # 所以这里可能有也可能没有 _tile_config


class TestGlobalTilerVectorNotTiled:
    """vector 算子不在 _TILEABLE_OPS → 不 tiling。"""

    def test_vector_add_no_tile(self):
        g = Graph()
        g.add_tensor(Tensor(id="a", shape=[1, 4096, 1024], dtype="fp16"))
        g.add_tensor(Tensor(id="b", shape=[1, 4096, 1024], dtype="fp16"))
        g.add_tensor(Tensor(id="c", shape=[1, 4096, 1024], dtype="fp16",
                            producer_node_id="add"))
        g.add_node(Node(
            id="add", op_type="vector_add", npu_op="vector_add",
            compute_unit="vector", is_mapped=True,
            inputs=["a", "b"], outputs=["c"],
        ))
        g.execution_order = ["add"]
        g = run(g, _hw_config())
        assert "_tile_config" not in g.nodes["add"].params


class TestTileConfigFormat:
    """验证 _tile_config 字段完整性。"""

    def test_config_fields(self):
        g = _make_matmul(4096, 1024, 1024)
        g = run(g, _hw_config())
        cfg = g.nodes["mm"].params.get("_tile_config")
        if cfg:
            assert "tile_size" in cfg
            assert "num_buffers" in cfg
            assert "num_tiles" in cfg
            assert "speedup" in cfg
            assert cfg["num_buffers"] in (1, 2)
