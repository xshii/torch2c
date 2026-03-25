"""sync_api_header — 从 c_api_signatures.yaml 同步 npu_api.h 的 compute op 声明。

用法:
    python -m torch2c.d_emission.codegen.sync_api_header

会就地更新 npu_cpu_mock/include/npu_api.h 中 BEGIN/END AUTO-GENERATED 标记之间的内容。
"""

from __future__ import annotations

import re
from pathlib import Path

from ._helpers import gen_compute_declarations, load_signatures

_BEGIN = "/* BEGIN AUTO-GENERATED COMPUTE OPS"
_END = "/* END AUTO-GENERATED COMPUTE OPS */"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HEADER_PATH = _REPO_ROOT / "npu_cpu_mock" / "include" / "npu_api.h"


def sync(header_path: Path | None = None) -> str:
    """读取 header，替换 compute op 声明区域，返回新内容。"""
    path = header_path or _HEADER_PATH
    text = path.read_text()

    sigs = load_signatures()
    decls = gen_compute_declarations(sigs)

    new_block = (
        f"{_BEGIN} — do not edit manually.\n"
        "   Source: torch2c/integration/config/c_api_signatures.yaml\n"
        "   Sync:   python -m torch2c.d_emission.codegen.sync_api_header */\n"
        f"{decls}\n"
        f"{_END}"
    )

    pattern = re.compile(
        re.escape(_BEGIN) + r".*?" + re.escape(_END),
        re.DOTALL,
    )
    new_text, count = pattern.subn(new_block, text)
    if count == 0:
        raise RuntimeError(f"未找到 AUTO-GENERATED 标记: {path}")

    return new_text


def main() -> None:
    path = _HEADER_PATH
    new_text = sync(path)
    path.write_text(new_text)
    print(f"已同步 {path}")


if __name__ == "__main__":
    main()
