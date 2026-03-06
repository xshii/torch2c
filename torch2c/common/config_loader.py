"""YAML 配置加载与 schema 校验。"""

from __future__ import annotations

import yaml

from .errors import ConfigError


def load_config(path: str, required_keys: list[str] | None = None) -> dict:
    """加载 YAML 配置文件并校验必填字段。

    Args:
        path: YAML 文件路径。
        required_keys: 需要存在的顶层字段列表。

    Returns:
        解析后的配置字典。

    Raises:
        FileNotFoundError: 文件不存在。
        ConfigError: 缺少必填字段或 YAML 解析失败。
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML 解析失败: {e}") from e

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射，实际为 {type(data).__name__}")

    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ConfigError(f"缺少必填字段: {', '.join(missing)}")

    return data
