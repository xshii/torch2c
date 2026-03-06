"""端到端 demo：编译 2 层 Encoder 模型并生成完整 C 工程。

用法：
    python -m torch2c.integration.demo.run_full_demo
"""

from __future__ import annotations

import os

import torch

from torch2c.common import DEFAULT_OUTPUT_DIR, INTEGRATION_CONFIG_DIR, setup_logging
from torch2c.integration.demo.encoder_model import EncoderModel
from torch2c.integration.pipeline import compile

_CONFIG_DIR = str(INTEGRATION_CONFIG_DIR)
_OUTPUT_DIR = str(DEFAULT_OUTPUT_DIR)


def main() -> None:
    setup_logging("INFO")

    model = EncoderModel(d_model=256, dim_ff=512, num_layers=2)
    model.eval()

    batch, seq, d_model = 1, 32, 256
    dummy_input = torch.randn(batch, seq, d_model)
    mask = torch.zeros(batch, seq, seq)

    output_dir = compile(
        model=model,
        dummy_input=dummy_input,
        config_dir=_CONFIG_DIR,
        output_dir=_OUTPUT_DIR,
        mask=mask,
    )

    print(f"\n编译完成！输出目录: {os.path.abspath(output_dir)}")

    # 验证关键文件存在
    from torch2c.integration.demo.validate_output import validate

    validate(output_dir)

    # C 编译 + 运行 + golden 比对
    from torch2c.integration.demo.validate_c_output import validate_c

    result = validate_c(output_dir)
    if result["passed"]:
        print("\nC golden 比对通过!")
    else:
        print(f"\nC golden 比对失败! (exit {result['returncode']})")
        print(result["stdout"])


if __name__ == "__main__":
    main()
