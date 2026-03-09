"""共享 pytest fixtures — 自动被 torch2c/ 下所有测试发现。"""

from __future__ import annotations

import pytest

from torch2c.common.paths import INTEGRATION_CONFIG_DIR
from torch2c.common.testing import load_hw_config, make_linear_chain
from torch2c.integration.pipeline import _load_configs


@pytest.fixture
def sample_graph():
    """3 节点线性链：mm -> add -> gelu（ATen 算子）。"""
    return make_linear_chain(n_ops=3, ops=[
        ("n0", "aten.mm.default", None),
        ("n1", "aten.add.Tensor", None),
        ("n2", "aten.gelu.default", None),
    ])


@pytest.fixture
def mapped_graph():
    """已映射的 3 节点链：cube_matmul -> vector_add -> vector_gelu。"""
    return make_linear_chain(n_ops=3, ops=[
        ("n0", "cube_matmul", "cube"),
        ("n1", "vector_add", "vector"),
        ("n2", "vector_gelu", "vector"),
    ])


@pytest.fixture
def hw_config():
    """hardware_config.yaml 配置。"""
    return load_hw_config()


@pytest.fixture
def all_configs():
    """加载 integration/config/ 全部配置。"""
    return _load_configs(str(INTEGRATION_CONFIG_DIR))
