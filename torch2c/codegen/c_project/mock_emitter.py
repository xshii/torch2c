"""mock_emitter — 根据 c_api_signatures.yaml 生成 npu_mock.h。"""

from __future__ import annotations

import os

from torch2c.common import get_logger

from .._helpers import gen_compute_decl, load_signatures, write_file

logger = get_logger("codegen.mock_emitter")


def emit_mock_h(signatures: dict) -> str:
    """生成 npu_mock.h 内容。"""
    lines = [
        "/* 自动生成 — 仅用于编译验证，不含实现 */",
        "#ifndef NPU_MOCK_H",
        "#define NPU_MOCK_H",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "typedef enum { NPU_DTYPE_FP16=0, NPU_DTYPE_FP32, NPU_DTYPE_BF16, "
        "NPU_DTYPE_INT8, NPU_DTYPE_INT32, NPU_DTYPE_INT16 } npu_dtype_t;",
        "typedef enum { NPU_FORMAT_ND=0, NPU_FORMAT_NZ, NPU_FORMAT_NC1HWC0 } npu_format_t;",
        "",
        "typedef struct { void* ptr; npu_dtype_t dtype; npu_format_t format; } npu_tensor_t;",
        "",
        "typedef struct { int task_id; int dep_cube_tid; int dep_vector_tid; "
        "int dep_dma_tid; int dep_idma_tid; } TidInfo;",
        "",
        "static inline void* npu_t_ptr(npu_tensor_t t) {",
        "    return t.ptr;",
        "}",
        "",
    ]
    for section in ["compute_ops", "dma_ops", "idma_ops"]:
        ops = signatures.get(section, {})
        if ops:
            lines.append(f"/* {section} */")
            for name, sig in ops.items():
                lines.append(gen_compute_decl(name, sig, include_optional=True))
            lines.append("")

    lines += [
        "#ifdef NPU_DEBUG_DUMP",
        "#include <stdio.h>",
        '#define NPU_LOG(fmt, ...) printf("[NPU] " fmt "\\n", ##__VA_ARGS__)',
        "#else",
        "#define NPU_LOG(fmt, ...) ((void)0)",
        "#endif",
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: str, config_dir: str | None = None) -> None:
    """生成 npu_mock.h 到 output_dir。"""
    logger.info("mock_emitter: 开始生成 npu_mock.h")
    sigs = load_signatures(config_dir)
    write_file(os.path.join(output_dir, "npu_mock.h"), emit_mock_h(sigs))
