"""config_loader 单元测试。"""

import pytest

from torch2c.common.config_loader import load_config
from torch2c.common.errors import ConfigError


@pytest.fixture
def valid_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("target: npu\nbatch_size: 4\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def empty_yaml(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    return str(p)


class TestLoadValid:
    def test_load_valid(self, valid_yaml):
        cfg = load_config(valid_yaml)
        assert cfg["target"] == "npu"
        assert cfg["batch_size"] == 4

    def test_load_with_required_keys(self, valid_yaml):
        cfg = load_config(valid_yaml, required_keys=["target"])
        assert "target" in cfg

    def test_load_empty_yaml(self, empty_yaml):
        cfg = load_config(empty_yaml)
        assert cfg == {}


class TestMissingKey:
    def test_missing_key(self, valid_yaml):
        with pytest.raises(ConfigError, match="缺少必填字段"):
            load_config(valid_yaml, required_keys=["nonexistent"])


class TestFileNotFound:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/no/such/file.yaml")
