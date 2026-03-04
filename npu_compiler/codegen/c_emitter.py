"""c_emitter — 生成 model_graph.c/h，每个算子三段式：DMA搬入→算子调用→DMA搬出。"""

from __future__ import annotations

import math
import os

from npu_compiler.common import CodegenError, get_logger

from ._helpers import (
    DTYPE_MAP,
    FORMAT_MAP,
    c_header_guard,
    load_signatures,
    write_files,
)

logger = get_logger("codegen.c_emitter")


# ---- 参数解析 ----


class SourceResolver:
    """将 c_api_signatures 中的 source 规则解析为 C 字符串。"""

    def __init__(self, node: dict, tensors: dict):
        self._node = node
        self._tensors = tensors

    def resolve(self, param: dict) -> str:
        """解析单个参数的 source → C 字符串。"""
        source = param.get("source", "")
        parts = source.split(".")

        if parts[0] == "tensor":
            return self._resolve_tensor_ref(parts[1], parts[2], parts[3:], param)
        if parts[0] == "param":
            return self._resolve_param_ref(parts[1], param)
        raise CodegenError(f"未知 source 格式: {source}")

    def find_tensor(self, key: str):
        """根据 key (input_0, output_0, mask 等) 查找 tensor。"""
        if key.startswith("input_"):
            idx = int(key.split("_")[1])
            inputs = self._node.get("inputs", [])
            absorbed_tids = set(self._node.get("absorbed_inputs", {}).values())
            regular = [tid for tid in inputs if tid not in absorbed_tids]
            return self._tensors.get(regular[idx]) if idx < len(regular) else None
        if key.startswith("output_"):
            idx = int(key.split("_")[1])
            outputs = self._node.get("outputs", [])
            return self._tensors.get(outputs[idx]) if idx < len(outputs) else None
        if key == "mask":
            mask_tid = self._node.get("absorbed_inputs", {}).get("mask")
            return self._tensors.get(mask_tid) if mask_tid else None
        return None

    def _resolve_tensor_ref(self, tensor_key, field, extra, param):
        t = self.find_tensor(tensor_key)
        if t is None:
            default = param.get("default")
            if default is not None:
                return str(default)
            raise CodegenError(f"张量 {tensor_key} 未找到: node={self._node['id']}")
        return self._extract_field(t, field, extra)

    def _resolve_param_ref(self, param_name, param):
        val = self._node.get("params", {}).get(param_name)
        if val is None:
            default = param.get("default")
            if default is not None:
                return str(default)
            raise CodegenError(f"参数 {param_name} 未找到: node={self._node['id']}")
        return _format_value(val, param["type"])

    @staticmethod
    def _extract_field(t: dict, field: str, extra: list[str]) -> str:
        """从 tensor dict 提取字段值。"""
        if field in ("l1_offset", "hbm_offset"):
            offset = t.get(field, 0) or 0
            buf = "l1" if field == "l1_offset" else "hbm"
            return f"(void*)({buf} + {offset})"
        if field == "dtype":
            return DTYPE_MAP.get(t.get("dtype", "fp16"), "NPU_DTYPE_FP16")
        if field == "format":
            return FORMAT_MAP.get(t.get("format", "nd"), "NPU_FORMAT_ND")
        if field == "shape":
            shape = t.get("shape", [])
            return str(shape[int(extra[0])]) if extra else str(shape)
        if field == "ndim":
            return str(len(t.get("shape", [])))
        if field == "elem_count":
            return str(math.prod(t.get("shape", [1])))
        return str(t.get(field, 0))


def _format_value(val, ptype: str) -> str:
    if ptype == "float":
        return f"{float(val):.6f}f"
    return str(val)


# ---- 代码生成 ----


def _gen_op_call(npu_op: str, sig: dict, node: dict, tensors: dict) -> str:
    """生成单个算子的 C 调用语句。"""
    resolver = SourceResolver(node, tensors)
    args = []
    for p in sig.get("params", []):
        if p["type"] == "int_array":
            parts = p["source"].split(".")
            t = resolver.find_tensor(parts[1])
            shape = t.get("shape", []) if t else []
            args.append(f"(const int[]){{{', '.join(str(s) for s in shape)}}}")
        else:
            args.append(resolver.resolve(p))

    for p in sig.get("optional_params", []):
        args.append(resolver.resolve(p))

    return f"{npu_op}({', '.join(args)})"


def _gen_dma_line(instr: dict) -> str:
    """生成单条 DMA 指令 C 代码。"""
    src_fmt = FORMAT_MAP.get(instr.get("src_format", "nd"), "NPU_FORMAT_ND")
    dst_fmt = FORMAT_MAP.get(instr.get("dst_format", "nd"), "NPU_FORMAT_ND")
    if instr["op"] == "load":
        return (
            f"npu_dma_load((void*)(l1 + {instr['l1_offset']}), "
            f"(void*)(hbm + {instr['hbm_offset']}), "
            f"{instr['size_bytes']}, {src_fmt}, {dst_fmt});"
        )
    return (
        f"npu_dma_store((void*)(hbm + {instr['hbm_offset']}), "
        f"(void*)(l1 + {instr['l1_offset']}), "
        f"{instr['size_bytes']}, {src_fmt}, {dst_fmt});"
    )


def _gen_dma_block(instructions: list[dict], indent: str) -> str:
    """生成 DMA load/store 语句列表。"""
    return "\n".join(f"{indent}{_gen_dma_line(i)}" for i in instructions)


def gen_op_block(node: dict, tensors: dict, dma_plan: dict, signatures: dict) -> str:
    """为单个算子生成完整的三段式代码块。"""
    npu_op = node.get("npu_op", "unknown")
    sig = signatures.get("compute_ops", {}).get(npu_op)
    if sig is None:
        raise CodegenError(f"签名未找到: {npu_op}")

    indent = "    "
    op_call = _gen_op_call(npu_op, sig, node, tensors)
    loads = _gen_dma_block(dma_plan.get("loads", []), indent)
    stores = _gen_dma_block(dma_plan.get("stores", []), indent)

    lines = [f"{indent}/* === {node['id']}: {npu_op} ({node.get('compute_unit', '?')}) === */"]
    if loads:
        lines.append(loads)
        lines.append(f"{indent}npu_dma_barrier();")
    lines.append(f"{indent}{op_call};")
    if stores:
        lines.append(stores)
        lines.append(f"{indent}npu_dma_barrier();")
    return "\n".join(lines)


# ---- 文件级生成 ----


def _gen_bulk_dma(dma_plan: dict, label: str) -> str:
    """生成 bulk DMA (load/store) 代码块。"""
    indent = "    "
    loads = dma_plan.get("loads", [])
    stores = dma_plan.get("stores", [])
    instructions = loads + stores
    if not instructions:
        return ""
    lines = [f"{indent}/* === {label} === */"]
    lines.append(_gen_dma_block(instructions, indent))
    lines.append(f"{indent}npu_dma_barrier();")
    return "\n".join(lines)


def emit_model_graph_c(plan: dict, signatures: dict) -> str:
    """生成 model_graph.c 完整内容。"""
    nodes = plan["nodes"]
    tensors = plan["tensors"]
    dma_plans = {dp["node_id"]: dp for dp in plan.get("dma_plans", [])}
    order = plan.get("execution_order", list(nodes.keys()))

    parts = []

    # bulk load（全局 L1 布局时）
    bulk_load = dma_plans.get("__bulk_load__")
    if bulk_load:
        parts.append(_gen_bulk_dma(bulk_load, "Bulk DMA Load"))

    for nid in order:
        dp = dma_plans.get(nid, {"loads": [], "stores": []})
        parts.append(gen_op_block(nodes[nid], tensors, dp, signatures))

    # bulk store（全局 L1 布局时）
    bulk_store = dma_plans.get("__bulk_store__")
    if bulk_store:
        parts.append(_gen_bulk_dma(bulk_store, "Bulk DMA Store"))

    body = "\n\n".join(p for p in parts if p)
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
    return c_header_guard(
        "MODEL_GRAPH_H",
        "void model_run(unsigned char* hbm, unsigned char* l1);\n",
    )


def emit_model_memory_h(plan: dict) -> str:
    """生成 model_memory.h — 所有 tensor 的 HBM/L1 偏移宏。"""
    lines = []
    for tid, t in plan["tensors"].items():
        safe = tid.upper()
        hbm = t.get("hbm_offset", 0) or 0
        hbm_sz = t.get("hbm_size", 0) or 0
        l1 = t.get("l1_offset", 0) or 0
        lines.append(f"#define {safe}_HBM_OFFSET  {hbm}")
        lines.append(f"#define {safe}_HBM_SIZE    {hbm_sz}")
        lines.append(f"#define {safe}_L1_OFFSET   {l1}")
    return c_header_guard("MODEL_MEMORY_H", "\n".join(lines) + "\n")


def emit_model_params_h(plan: dict) -> str:
    """生成 model_params.h — 各算子的参数宏。"""
    lines = []
    for nid, node in plan["nodes"].items():
        for k, v in node.get("params", {}).items():
            safe = f"{nid}_{k}".upper()
            lines.append(f"#define {safe}  {v}")
    return c_header_guard("MODEL_PARAMS_H", "\n".join(lines) + "\n")


def run(plan: dict, output_dir: str, config_dir: str | None = None) -> None:
    """生成 model_graph.c/h, model_memory.h, model_params.h。"""
    logger.info("c_emitter: 开始生成 model_graph 文件")
    sigs = load_signatures(config_dir)
    src_dir = os.path.join(output_dir, "src")
    write_files(
        src_dir,
        [
            ("model_graph.c", emit_model_graph_c(plan, sigs)),
            ("model_graph.h", emit_model_graph_h()),
            ("model_memory.h", emit_model_memory_h(plan)),
            ("model_params.h", emit_model_params_h(plan)),
        ],
    )
