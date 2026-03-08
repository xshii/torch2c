"""系统测试：混合精度 Encoder 端到端编译 + C golden 比对。"""

from __future__ import annotations

import tempfile

import pytest

from torch2c.integration.demo._runner import compile_and_validate


class TestE2EMixed:
    """混合精度（fp32 + fp16）全链路系统测试。"""

    def test_compile_and_golden_pass(self):
        """编译 mixed 模型，C golden 比对通过。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            r = compile_and_validate("mixed", tmpdir)
            assert r["passed"], f"C golden 比对失败: {r['stdout']}"

    def test_max_abs_within_tolerance(self):
        """mixed 模式 max_abs_diff < 0.02。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            r = compile_and_validate("mixed", tmpdir)
            assert r["max_abs"] < 0.02, f"max_abs={r['max_abs']} 超过容差"

    def test_cosine_similarity(self):
        """mixed 模式 cosine > 0.999。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            r = compile_and_validate("mixed", tmpdir)
            assert r["cosine"] > 0.999, f"cosine={r['cosine']} 低于阈值"
