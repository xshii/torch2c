"""demo 公共编译+验证逻辑。"""

from __future__ import annotations

import os

import torch

from torch2c.common import INTEGRATION_CONFIG_DIR, get_logger
from torch2c.integration.demo.encoder_model import EncoderModel
from torch2c.integration.demo.validate_c_output import validate_c
from torch2c.integration.demo.validate_output import validate
from torch2c.integration.pipeline import compile

logger = get_logger(__name__)

_CONFIG_DIR = str(INTEGRATION_CONFIG_DIR)

# 精度模式 → 中文标签
PRECISION_LABEL = {"mixed": "混合精度", "fp16": "全 FP16"}


def compile_and_validate(
    precision: str,
    output_dir: str,
    *,
    d_model: int = 192,
    dim_ff: int = 384,
    num_layers: int = 3,
    num_heads: int = 3,
    batch: int = 1,
    seq_len: int = 32,
) -> dict:
    """编译指定精度模式的 Encoder 模型，返回验证结果。

    Returns:
        {"passed": bool, "output_dir": str, "precision": str,
         "max_abs": float, "cosine": float, "stdout": str}
    """
    model = EncoderModel(
        d_model=d_model, dim_ff=dim_ff, num_layers=num_layers,
        num_heads=num_heads, precision=precision,
    )
    model.eval()

    dummy_input = torch.randn(batch, seq_len, d_model)
    mask = torch.zeros(batch, seq_len, seq_len)

    out = compile(
        model=model,
        dummy_input=dummy_input,
        config_dir=_CONFIG_DIR,
        output_dir=output_dir,
        mask=mask,
    )

    validate(out)
    c_result = validate_c(out)

    # 从 stdout 提取精度指标
    max_abs, cosine = _parse_metrics(c_result["stdout"])

    return {
        "passed": c_result["passed"],
        "output_dir": os.path.abspath(out),
        "precision": precision,
        "max_abs": max_abs,
        "cosine": cosine,
        "stdout": c_result["stdout"],
    }


def _parse_metrics(stdout: str) -> tuple[float, float]:
    """从 C 运行输出解析 max_abs 和 cosine 值。"""
    import re
    m = re.search(r"max_abs=([0-9.]+).*?cosine=([0-9.]+)", stdout)
    if m:
        return float(m.group(1)), float(m.group(2))
    return -1.0, -1.0
