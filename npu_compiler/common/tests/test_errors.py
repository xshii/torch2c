"""DiagnosticCollector 单元测试。"""

from npu_compiler.common.errors import (
    CompileDiagnostic,
    DiagnosticCollector,
    Severity,
)


class TestDiagnosticCollector:
    def test_warn_and_error(self):
        c = DiagnosticCollector()
        c.warn("graph_capture", "权重缺少 name")
        c.error("op_mapping", "未知算子")
        assert len(c.items) == 2
        assert c.items[0].severity == Severity.WARNING
        assert c.items[1].severity == Severity.ERROR

    def test_has_errors(self):
        c = DiagnosticCollector()
        assert not c.has_errors()
        c.warn("p1", "w")
        assert not c.has_errors()
        c.error("p2", "e")
        assert c.has_errors()

    def test_summary(self):
        c = DiagnosticCollector()
        c.warn("a", "w1")
        c.warn("a", "w2")
        c.error("b", "e1")
        s = c.summary()
        assert "1 error" in s
        assert "2 warning" in s
