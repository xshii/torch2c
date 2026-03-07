"""torch_debug 维测系统单元测试。"""

from __future__ import annotations

import io

import torch
import torch.nn as nn
import pytest

from torch2c.common import npu, NpuSpec
from torch2c.common.torch_debug import (
    enable_trace,
    disable_trace,
    flush_trace,
    get_records,
    trace_context,
)


# ---- fixture ----

class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = npu(nn.Linear(8, 16), input=NpuSpec("fp32"), output=NpuSpec("fp32"))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = _SimpleModel()
    m.eval()
    return m


@pytest.fixture
def dummy():
    return torch.randn(1, 4, 8)


# ---- tests ----

class TestEnableDisable:
    def test_enable_records_entries(self, model, dummy):
        enable_trace(model)
        with torch.no_grad():
            model(dummy)
        recs = get_records()
        assert len(recs) > 0
        disable_trace(model)

    def test_disable_stops_recording(self, model, dummy):
        enable_trace(model)
        disable_trace(model)
        with torch.no_grad():
            model(dummy)
        recs = get_records()
        assert len(recs) == 0

    def test_flush_clears(self, model, dummy):
        enable_trace(model)
        with torch.no_grad():
            model(dummy)
        buf = io.StringIO()
        flush_trace(file=buf)
        assert len(get_records()) == 0
        assert "TORCH DEBUG TRACE" in buf.getvalue()
        disable_trace(model)


class TestTraceContent:
    def test_records_have_module_paths(self, model, dummy):
        enable_trace(model)
        with torch.no_grad():
            model(dummy)
        recs = get_records()
        paths = {r.module_path for r in recs}
        assert "fc1" in paths
        assert "fc2" in paths
        assert "act" in paths
        disable_trace(model)

    def test_tensor_stats_valid(self, model, dummy):
        enable_trace(model)
        with torch.no_grad():
            model(dummy)
        recs = get_records()
        for r in recs:
            for s in r.inputs + r.outputs:
                assert len(s.shape) > 0
                assert s.numel > 0
                assert s.nan_count == 0
                assert s.inf_count == 0
                assert s.min_val <= s.max_val
                assert s.std_val >= 0
        disable_trace(model)

    def test_input_output_shapes_match_model(self, model, dummy):
        enable_trace(model)
        with torch.no_grad():
            model(dummy)
        recs = get_records()
        fc1_rec = [r for r in recs if r.module_path == "fc1"][0]
        assert fc1_rec.inputs[0].shape == [1, 4, 8]
        assert fc1_rec.outputs[0].shape == [1, 4, 16]
        disable_trace(model)


class TestContextManager:
    def test_context_manager_auto_cleanup(self, model, dummy):
        buf = io.StringIO()
        with trace_context(model, file=buf):
            with torch.no_grad():
                model(dummy)
        # after context: records flushed, hooks removed
        assert len(get_records()) == 0
        assert "TORCH DEBUG TRACE" in buf.getvalue()
        # verify hooks removed: run again, no new records
        with torch.no_grad():
            model(dummy)
        assert len(get_records()) == 0


class TestLeafOnly:
    def test_leaf_only_skips_containers(self, model, dummy):
        enable_trace(model, leaf_only=True)
        with torch.no_grad():
            model(dummy)
        recs = get_records()
        for r in recs:
            mod = dict(model.named_modules())[r.module_path]
            assert len(list(mod.children())) == 0, f"{r.module_path} is not a leaf"
        disable_trace(model)


class TestNanInf:
    def test_detects_nan(self):
        """人为注入 NaN，验证维测能检测到。"""
        class NanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
            def forward(self, x):
                return self.fc(x)

        m = NanModel()
        m.eval()
        x = torch.tensor([[1.0, float('nan'), 3.0, 4.0]])
        enable_trace(m)
        with torch.no_grad():
            m(x)
        recs = get_records()
        fc_rec = [r for r in recs if r.module_path == "fc"][0]
        assert fc_rec.inputs[0].nan_count == 1
        disable_trace(m)
