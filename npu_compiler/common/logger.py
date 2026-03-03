"""统一日志系统。

日志格式: [时间] [级别] [模块名] 消息
通过环境变量 NPU_LOG_LEVEL 设置级别（默认 INFO）。
"""

import logging
import os

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(level: str = None, log_file: str = None) -> None:
    """配置全局日志。

    Args:
        level: 日志级别字符串，默认从 NPU_LOG_LEVEL 环境变量读取，否则 INFO。
        log_file: 可选的日志文件路径。
    """
    global _initialized
    if level is None:
        level = os.environ.get("NPU_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取指定模块的 logger。

    Args:
        name: 模块名称。

    Returns:
        配置好的 Logger 实例。
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
