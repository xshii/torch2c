"""utils_emitter — 生成辅助工具 C 代码：data_loader, data_dumper, comparator。"""

from __future__ import annotations

import os
from pathlib import Path

from npu_compiler.common import get_logger

from .._helpers import c_header_guard, write_files

logger = get_logger("codegen.utils_emitter")

_TMPL_DIR = Path(__file__).parent.parent / "templates"


def _load_template(name: str) -> str:
    with open(_TMPL_DIR / name, "r", encoding="utf-8") as f:
        return f.read()


# ---- data_loader ----


def emit_data_loader_c() -> str:
    return _load_template("data_loader.c.tmpl").format()


_MAX_TENSOR_NDIM = 8
_NAME_BUF_LEN = 16  # dtype / format 字段的 C 缓冲区长度


def emit_data_loader_h() -> str:
    body = (
        "#include <stddef.h>\n\n"
        "typedef struct {\n"
        f"    int shape[{_MAX_TENSOR_NDIM}];\n"
        "    int ndim;\n"
        f"    char dtype[{_NAME_BUF_LEN}];\n"
        f"    char format[{_NAME_BUF_LEN}];\n"
        "    size_t total_bytes;\n"
        "} tensor_desc_t;\n\n"
        "int parse_desc(const char* desc_path, tensor_desc_t* desc);\n"
        "int load_tensor(void* hbm_base, size_t offset,\n"
        "                const char* bin_path, const char* desc_path);\n"
    )
    return c_header_guard("DATA_LOADER_H", body)


# ---- comparator ----


def emit_comparator_c() -> str:
    return _load_template("comparator.c.tmpl").format()


def emit_comparator_h() -> str:
    body = (
        '#include "data_loader.h"\n\n'
        "typedef struct {\n"
        "    float max_abs_diff;\n"
        "    float max_rel_diff;\n"
        "    float cosine_similarity;\n"
        "    float mse;\n"
        "    int mismatch_count;\n"
        "    int total_elements;\n"
        "    int first_mismatch_index;\n"
        "} compare_result_t;\n\n"
        "int compare_tensors(const char* actual_path, const char* golden_path,\n"
        "                    const char* desc_path, float abs_tol, float cos_tol,\n"
        "                    compare_result_t* result);\n"
    )
    return c_header_guard("COMPARATOR_H", body)


# ---- data_dumper ----


def emit_data_dumper_c() -> str:
    return (
        '#include "data_dumper.h"\n'
        "#include <stdio.h>\n\n"
        "int dump_tensor(const void* hbm_base, size_t offset,\n"
        "                size_t size, const char* path) {\n"
        '    FILE* f = fopen(path, "wb");\n'
        "    if (!f) return -1;\n"
        "    fwrite((const unsigned char*)hbm_base + offset, 1, size, f);\n"
        "    fclose(f);\n"
        "    return 0;\n"
        "}\n"
    )


def emit_data_dumper_h() -> str:
    body = (
        "#include <stddef.h>\n\n"
        "int dump_tensor(const void* hbm_base, size_t offset,\n"
        "                size_t size, const char* path);\n"
    )
    return c_header_guard("DATA_DUMPER_H", body)


# ---- tensor_utils ----


def emit_tensor_utils_h() -> str:
    body = (
        "#include <stddef.h>\n"
        "#include <string.h>\n\n"
        "static inline size_t dtype_size(const char* dtype) {\n"
        '    if (strncmp(dtype, "fp32", 4) == 0 || strncmp(dtype, "int32", 5) == 0) return 4;\n'
        "    return 2; /* fp16, bf16, int8, int16 default */\n"
        "}\n\n"
        "static inline size_t elem_count(const int* shape, int ndim) {\n"
        "    size_t n = 1;\n"
        "    for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];\n"
        "    return n;\n"
        "}\n"
    )
    return c_header_guard("TENSOR_UTILS_H", body)


# ---- 批量生成 ----


def run(output_dir: str) -> None:
    logger.info("utils_emitter: 开始生成辅助工具代码")
    write_files(
        os.path.join(output_dir, "utils"),
        [
            ("data_loader.c", emit_data_loader_c()),
            ("data_loader.h", emit_data_loader_h()),
            ("comparator.c", emit_comparator_c()),
            ("comparator.h", emit_comparator_h()),
            ("data_dumper.c", emit_data_dumper_c()),
            ("data_dumper.h", emit_data_dumper_h()),
            ("tensor_utils.h", emit_tensor_utils_h()),
        ],
    )
