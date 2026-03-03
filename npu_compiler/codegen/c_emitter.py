"""c_emitter — 生成 model_graph.c/h，每个算子三段式：DMA搬入→算子调用→DMA搬出。"""

from __future__ import annotations

import math
import os
from pathlib import Path

from npu_compiler.common import CodegenError, get_logger, load_config

logger = get_logger("codegen.c_emitter")

_DTYPE_MAP = {"fp16": "NPU_DTYPE_FP16", "fp32": "NPU_DTYPE_FP32",
              "bf16": "NPU_DTYPE_BF16", "int8": "NPU_DTYPE_INT8",
              "int32": "NPU_DTYPE_INT32"}
_FORMAT_MAP = {"nd": "NPU_FORMAT_ND", "nz": "NPU_FORMAT_NZ",
               "nc1hwc0": "NPU_FORMAT_NC1HWC0"}


def _resolve_param(param: dict, node: dict, tensors: dict) -> str:
    """根据 source 规则解析参数值为 C 字符串。"""
    source = param.get("source", "")
    parts = source.split(".")

    if parts[0] == "tensor":
        tensor_key, field = parts[1], parts[2]
        t = _find_tensor(tensor_key, node, tensors)
        if t is None:
            default = param.get("default")
            if default is not None:
                return str(default)
            raise CodegenError(f"张量 {tensor_key} 未找到: node={node['id']}")
        return _extract_tensor_field(t, field, parts[3:])

    if parts[0] == "param":
        val = node.get("params", {}).get(parts[1])
        if val is None:
            default = param.get("default")
            if default is not None:
                return str(default)
            raise CodegenError(f"参数 {parts[1]} 未找到: node={node['id']}")
        return _format_value(val, param["type"])

    raise CodegenError(f"未知 source 格式: {source}")


def _find_tensor(key: str, node: dict, tensors: dict):
    """根据 key (input_0, output_0, mask 等) 查找 tensor。"""
    if key.startswith("input_"):
        idx = int(key.split("_")[1])
        inputs = node.get("inputs", [])
        # absorbed_inputs 中的 tensor 已被吸收为参数，不计入常规输入
        absorbed_tids = set(node.get("absorbed_inputs", {}).values())
        regular = [tid for tid in inputs if tid not in absorbed_tids]
        if idx < len(regular):
            return tensors.get(regular[idx])
        return None
    if key.startswith("output_"):
        idx = int(key.split("_")[1])
        outputs = node.get("outputs", [])
        return tensors.get(outputs[idx]) if idx < len(outputs) else None
    if key == "mask":
        absorbed = node.get("absorbed_inputs", {})
        mask_tid = absorbed.get("mask")
        return tensors.get(mask_tid) if mask_tid else None
    return None


def _extract_tensor_field(t: dict, field: str, extra: list[str]) -> str:
    """从 tensor dict 提取字段值。"""
    if field == "l1_offset":
        offset = t.get("l1_offset", 0) or 0
        return f"(void*)(l1 + {offset})"
    if field == "hbm_offset":
        offset = t.get("hbm_offset", 0) or 0
        return f"(void*)(hbm + {offset})"
    if field == "dtype":
        return _DTYPE_MAP.get(t.get("dtype", "fp16"), "NPU_DTYPE_FP16")
    if field == "format":
        return _FORMAT_MAP.get(t.get("format", "nd"), "NPU_FORMAT_ND")
    if field == "shape":
        shape = t.get("shape", [])
        if extra:
            idx = int(extra[0])
            return str(shape[idx])
        return str(shape)
    if field == "ndim":
        return str(len(t.get("shape", [])))
    if field == "elem_count":
        return str(math.prod(t.get("shape", [1])))
    return str(t.get(field, 0))


def _format_value(val, ptype: str) -> str:
    if ptype == "float":
        return f"{float(val):.6f}f"
    return str(val)


def _gen_op_call(npu_op: str, sig: dict, node: dict, tensors: dict) -> str:
    """生成单个算子的 C 调用语句。"""
    args = []
    for p in sig.get("params", []):
        if p["type"] == "int_array":
            shape = _get_shape_for_int_array(p, node, tensors)
            args.append(f"(const int[]){{{', '.join(str(s) for s in shape)}}}")
        else:
            args.append(_resolve_param(p, node, tensors))

    for p in sig.get("optional_params", []):
        args.append(_resolve_param(p, node, tensors))

    return f"{npu_op}({', '.join(args)})"


def _get_shape_for_int_array(param: dict, node: dict, tensors: dict) -> list:
    parts = param["source"].split(".")
    t = _find_tensor(parts[1], node, tensors)
    return t.get("shape", []) if t else []


def _gen_dma_block(instructions: list[dict], indent: str) -> str:
    """生成 DMA load/store 语句列表。"""
    lines = []
    for instr in instructions:
        op = instr["op"]
        src_fmt = _FORMAT_MAP.get(instr.get("src_format", "nd"), "NPU_FORMAT_ND")
        dst_fmt = _FORMAT_MAP.get(instr.get("dst_format", "nd"), "NPU_FORMAT_ND")
        if op == "load":
            line = (f"npu_dma_load((void*)(l1 + {instr['l1_offset']}), "
                    f"(void*)(hbm + {instr['hbm_offset']}), "
                    f"{instr['size_bytes']}, {src_fmt}, {dst_fmt});")
        else:
            line = (f"npu_dma_store((void*)(hbm + {instr['hbm_offset']}), "
                    f"(void*)(l1 + {instr['l1_offset']}), "
                    f"{instr['size_bytes']}, {src_fmt}, {dst_fmt});")
        lines.append(f"{indent}{line}")
    return "\n".join(lines)


def gen_op_block(node: dict, tensors: dict, dma_plan: dict,
                 signatures: dict) -> str:
    """为单个算子生成完整的三段式代码块。"""
    npu_op = node.get("npu_op", "unknown")
    sig = signatures.get("compute_ops", {}).get(npu_op)
    if sig is None:
        raise CodegenError(f"签名未找到: {npu_op}")

    indent = "    "
    op_call = _gen_op_call(npu_op, sig, node, tensors)
    loads = _gen_dma_block(dma_plan.get("loads", []), indent)
    stores = _gen_dma_block(dma_plan.get("stores", []), indent)

    lines = [f"    /* === {node['id']}: {npu_op} ({node.get('compute_unit', '?')}) === */"]
    if loads:
        lines.append(loads)
        lines.append(f"{indent}npu_dma_barrier();")
    lines.append(f"{indent}{op_call};")
    if stores:
        lines.append(stores)
        lines.append(f"{indent}npu_dma_barrier();")
    return "\n".join(lines)


def emit_model_graph_c(plan: dict, signatures: dict) -> str:
    """生成 model_graph.c 完整内容。"""
    nodes = plan["nodes"]
    tensors = plan["tensors"]
    dma_plans = {dp["node_id"]: dp for dp in plan.get("dma_plans", [])}
    order = plan.get("execution_order", list(nodes.keys()))

    blocks = []
    for nid in order:
        node = nodes[nid]
        dp = dma_plans.get(nid, {"loads": [], "stores": []})
        blocks.append(gen_op_block(node, tensors, dp, signatures))

    body = "\n\n".join(blocks)
    return (
        '#include "model_graph.h"\n'
        '#include "model_memory.h"\n'
        '#include "model_weights.h"\n'
        '#include "npu_mock.h"\n\n'
        "void model_run(unsigned char* hbm, unsigned char* l1) {\n"
        f"{body}\n"
        "}\n"
    )


def emit_model_graph_h() -> str:
    return (
        "#ifndef MODEL_GRAPH_H\n#define MODEL_GRAPH_H\n\n"
        "void model_run(unsigned char* hbm, unsigned char* l1);\n\n"
        "#endif\n"
    )


def emit_model_memory_h(plan: dict) -> str:
    """生成 model_memory.h — 所有 tensor 的 HBM/L1 偏移宏。"""
    lines = ["#ifndef MODEL_MEMORY_H", "#define MODEL_MEMORY_H", ""]
    for tid, t in plan["tensors"].items():
        safe = tid.upper()
        hbm = t.get("hbm_offset", 0) or 0
        hbm_sz = t.get("hbm_size", 0) or 0
        l1 = t.get("l1_offset", 0) or 0
        lines.append(f"#define {safe}_HBM_OFFSET  {hbm}")
        lines.append(f"#define {safe}_HBM_SIZE    {hbm_sz}")
        lines.append(f"#define {safe}_L1_OFFSET   {l1}")
    lines += ["", "#endif", ""]
    return "\n".join(lines)


def emit_model_params_h(plan: dict) -> str:
    """生成 model_params.h — 各算子的参数宏。"""
    lines = ["#ifndef MODEL_PARAMS_H", "#define MODEL_PARAMS_H", ""]
    for nid, node in plan["nodes"].items():
        for k, v in node.get("params", {}).items():
            safe = f"{nid}_{k}".upper()
            lines.append(f"#define {safe}  {v}")
    lines += ["", "#endif", ""]
    return "\n".join(lines)


def run(plan: dict, output_dir: str, config_dir: str | None = None) -> None:
    """生成 model_graph.c/h, model_memory.h, model_params.h。"""
    logger.info("c_emitter: 开始生成 model_graph 文件")
    if config_dir is None:
        config_dir = str(Path(__file__).parent / "config")
    sigs = load_config(
        os.path.join(config_dir, "c_api_signatures.yaml"),
        required_keys=["compute_ops"],
    )

    src_dir = os.path.join(output_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    for name, content in [
        ("model_graph.c", emit_model_graph_c(plan, sigs)),
        ("model_graph.h", emit_model_graph_h()),
        ("model_memory.h", emit_model_memory_h(plan)),
        ("model_params.h", emit_model_params_h(plan)),
    ]:
        path = os.path.join(src_dir, name)
        with open(path, "w") as f:
            f.write(content)
        logger.info("c_emitter: 已生成 %s", path)
