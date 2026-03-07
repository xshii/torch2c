"""_struct_gen — 结构体/宏定义生成。"""

from __future__ import annotations

from ._helpers import DTYPE_C_ENUM_MAP, FORMAT_MAP
from ._naming import _section_to_c_field


def _gen_tensor_struct_typedef(
    sections: list[tuple[str, list[str]]], c_names: dict[str, str],
) -> str:
    """生成带子结构体的 model_tensors_t 类型定义。"""
    lines = ["typedef struct {"]
    for section_name, tids in sections:
        c_section = _section_to_c_field(section_name)
        lines.append(f"    struct {{")
        for tid in tids:
            # c_names[tid] = "section.field", 取 field 部分
            full = c_names.get(tid, tid)
            field = full.split(".", 1)[1] if "." in full else full
            lines.append(f"        npu_tensor_t {field};")
        lines.append(f"    }} {c_section};")
    lines.append("} model_tensors_t;")
    return "\n".join(lines)


def _collect_spec_macros(
    sections: list[tuple[str, list[str]]], tensors: dict,
) -> dict[tuple[str, str], str]:
    """收集所有 (dtype_enum, format_enum) 组合，生成宏名映射。"""
    specs: set[tuple[str, str]] = set()
    for _, tids in sections:
        for tid in tids:
            t = tensors.get(tid, {})
            dtype_enum = DTYPE_C_ENUM_MAP.get(t.get("dtype", "fp16"), "NPU_DTYPE_FP16")
            fmt_enum = FORMAT_MAP.get(t.get("format", "nd"), "NPU_FORMAT_ND")
            specs.add((dtype_enum, fmt_enum))
    # NPU_DTYPE_FP16 → FP16, NPU_FORMAT_NZ → NZ
    result: dict[tuple[str, str], str] = {}
    for dtype_enum, fmt_enum in sorted(specs):
        d = dtype_enum.replace("NPU_DTYPE_", "")
        f = fmt_enum.replace("NPU_FORMAT_", "")
        result[(dtype_enum, fmt_enum)] = f"T_{d}_{f}"
    return result


def _gen_spec_macro_defs(macros: dict[tuple[str, str], str]) -> str:
    """生成 tensor spec 宏定义。"""
    lines = []
    for (dtype_enum, fmt_enum), macro_name in sorted(macros.items(), key=lambda x: x[1]):
        lines.append(
            f"#define {macro_name}(base, off)  "
            f"{{(base) + (off), {dtype_enum}, {fmt_enum}}}"
        )
    return "\n".join(lines)


def _gen_tensor_struct_init(
    sections: list[tuple[str, list[str]]],
    tensors: dict,
    c_names: dict[str, str],
    spec_macros: dict[tuple[str, str], str],
) -> str:
    """生成 model_tensors_t 的嵌套 designated initializer（使用 spec 宏）。"""
    lines = ["    model_tensors_t t = {"]
    for section_name, tids in sections:
        c_section = _section_to_c_field(section_name)
        lines.append(f"        .{c_section} = {{")
        for tid in tids:
            t = tensors.get(tid, {})
            offset = t.get("l1_offset", 0) or 0
            dtype_enum = DTYPE_C_ENUM_MAP.get(t.get("dtype", "fp16"), "NPU_DTYPE_FP16")
            fmt_enum = FORMAT_MAP.get(t.get("format", "nd"), "NPU_FORMAT_ND")
            full = c_names.get(tid, tid)
            field = full.split(".", 1)[1] if "." in full else full
            macro = spec_macros.get((dtype_enum, fmt_enum))
            if macro:
                lines.append(f"            .{field} = {macro}(l1, {offset}),")
            else:
                lines.append(
                    f"            .{field} = {{l1 + {offset}, {dtype_enum}, {fmt_enum}}},"
                )
        lines.append(f"        }},")
    lines.append("    };")
    return "\n".join(lines)


def _extract_model_dims(tensors: dict) -> dict[str, int]:
    """从 tensor shape 推断模型维度常量。

    返回 {宏名: 值}，如 {"BATCH": 1, "SEQ_LEN": 32, "D_MODEL": 256, "DIM_FF": 512}。
    """
    dims: dict[str, int] = {}
    for t in tensors.values():
        shape = t.get("shape", [])
        if t.get("is_model_input") and not dims.get("BATCH"):
            # 输入形如 [batch, seq_len, d_model]
            if len(shape) == 3:
                dims["BATCH"] = shape[0]
                dims["SEQ_LEN"] = shape[1]
                dims["D_MODEL"] = shape[2]
            elif len(shape) == 2:
                dims["BATCH"] = shape[0]
                dims["SEQ_LEN"] = shape[1]
        if t.get("is_weight") and t.get("name", ""):
            name = t["name"]
            # linear1.weight → [dim_ff, d_model] or NZ equivalent
            if "linear1.weight" in name and len(shape) >= 2:
                # 取最大维度作为 dim_ff
                dim_ff = max(shape)
                if dim_ff != dims.get("D_MODEL"):
                    dims["DIM_FF"] = dim_ff
    return dims


def _gen_dim_macros(dims: dict[str, int]) -> str:
    """生成模型维度宏定义。"""
    if not dims:
        return ""
    lines = ["/* Model dimension constants */"]
    for name, val in dims.items():
        lines.append(f"#define {name}  {val}")
    return "\n".join(lines)


def _build_dim_replace_map(dims: dict[str, int]) -> dict[int, str]:
    """构建 值→宏名 替换表（仅对有歧义的值去重，保留唯一映射）。"""
    val_to_names: dict[int, list[str]] = {}
    for name, val in dims.items():
        val_to_names.setdefault(val, []).append(name)
    # 只保留值唯一对应一个宏名的映射
    result: dict[int, str] = {}
    for val, names in val_to_names.items():
        if len(names) == 1:
            result[val] = names[0]
    return result
