"""utils_emitter — 生成辅助工具 C 代码：data_loader, data_dumper, comparator。"""

from __future__ import annotations

import os
from pathlib import Path

from npu_compiler.common import get_logger

logger = get_logger("codegen.utils_emitter")

_TMPL_DIR = Path(__file__).parent.parent / "templates"


def _load_template(name: str) -> str:
    path = _TMPL_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def emit_data_loader_c() -> str:
    return _load_template("data_loader.c.tmpl").format()


def emit_data_loader_h() -> str:
    return (
        "#ifndef DATA_LOADER_H\n#define DATA_LOADER_H\n\n"
        "#include <stddef.h>\n\n"
        "typedef struct {\n"
        "    int shape[8];\n"
        "    int ndim;\n"
        "    char dtype[16];\n"
        "    char format[16];\n"
        "    size_t total_bytes;\n"
        "} tensor_desc_t;\n\n"
        "int parse_desc(const char* desc_path, tensor_desc_t* desc);\n"
        "int load_tensor(void* hbm_base, size_t offset,\n"
        "                const char* bin_path, const char* desc_path);\n\n"
        "#endif\n"
    )


def emit_comparator_c() -> str:
    return _load_template("comparator.c.tmpl").format()


def emit_comparator_h() -> str:
    return (
        "#ifndef COMPARATOR_H\n#define COMPARATOR_H\n\n"
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
        "                    compare_result_t* result);\n\n"
        "#endif\n"
    )


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
    return (
        "#ifndef DATA_DUMPER_H\n#define DATA_DUMPER_H\n\n"
        "#include <stddef.h>\n\n"
        "int dump_tensor(const void* hbm_base, size_t offset,\n"
        "                size_t size, const char* path);\n\n"
        "#endif\n"
    )


def emit_tensor_utils_h() -> str:
    return (
        "#ifndef TENSOR_UTILS_H\n#define TENSOR_UTILS_H\n\n"
        "#include <stddef.h>\n\n"
        "static inline size_t dtype_size(const char* dtype) {\n"
        '    if (dtype[0] == \'f\' && dtype[1] == \'p\' && dtype[2] == \'3\') return 4;\n'
        "    return 2; /* fp16, bf16, int16 default */\n"
        "}\n\n"
        "static inline size_t elem_count(const int* shape, int ndim) {\n"
        "    size_t n = 1;\n"
        "    for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];\n"
        "    return n;\n"
        "}\n\n"
        "#endif\n"
    )


def run(output_dir: str) -> None:
    logger.info("utils_emitter: 开始生成辅助工具代码")
    utils_dir = os.path.join(output_dir, "utils")
    os.makedirs(utils_dir, exist_ok=True)

    files = [
        ("data_loader.c", emit_data_loader_c()),
        ("data_loader.h", emit_data_loader_h()),
        ("comparator.c", emit_comparator_c()),
        ("comparator.h", emit_comparator_h()),
        ("data_dumper.c", emit_data_dumper_c()),
        ("data_dumper.h", emit_data_dumper_h()),
        ("tensor_utils.h", emit_tensor_utils_h()),
    ]
    for name, content in files:
        path = os.path.join(utils_dir, name)
        with open(path, "w") as f:
            f.write(content)
        logger.info("utils_emitter: 已生成 %s", path)
