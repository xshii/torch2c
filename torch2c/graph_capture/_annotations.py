"""NPU 标注传播：将 npu() / npu_input() 标注写入 Graph IR。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..common.graph_ir import Graph
from ..common.logger import get_logger
from ..common.npu_annotate import NpuSpec, get_input_annotation, get_npu_annotations
from ._constants import DIM_TO_SIZE_OPS

logger = get_logger(__name__)


def _apply_module_annotations(
    graph: Graph,
    annotations: dict[str, dict],
    ep: Any,
    fx_to_nid: dict[str, str],
) -> None:
    """通过 nn_module_stack 匹配 FX 节点到模块路径，写入标注。"""
    if not annotations:
        return
    for fx_node in ep.graph_module.graph.nodes:
        if fx_node.op != "call_function" or fx_node.name not in fx_to_nid:
            continue
        nn_stack = fx_node.meta.get("nn_module_stack")
        if not nn_stack:
            continue
        innermost_path = list(nn_stack.values())[-1][0]
        if innermost_path not in annotations:
            continue
        ann = annotations[innermost_path]
        nid = fx_to_nid[fx_node.name]
        node = graph.get_node(nid)
        if node is None:
            continue
        node.params["_npu"] = ann
        if "compute_dtype" in ann and "compute_dtype" not in node.params:
            node.params["compute_dtype"] = ann["compute_dtype"]
        logger.debug("模块标注 %s -> node %s: %s", innermost_path, nid, ann)


def _apply_weight_annotations(
    graph: Graph,
    annotations: dict[str, dict],
    model: nn.Module,
) -> None:
    """将模块标注传播到 weight Tensor 的 dtype/format。"""
    if not annotations:
        return
    weight_to_module: dict[str, str] = {}
    for name in model.state_dict():
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            weight_to_module[name] = parts[0]
    for t in graph.tensors.values():
        if not (t.is_weight and t.name):
            continue
        mod_path = weight_to_module.get(t.name)
        if mod_path is None or mod_path not in annotations:
            continue
        ann = annotations[mod_path]
        spec: NpuSpec | None = ann.get("weight")
        if spec is not None:
            t.src_dtype = t.dtype
            t.dtype = spec.dtype
            t.format = spec.format
        logger.debug("权重标注 %s: %s -> %s", t.name, t.src_dtype, spec)


def _apply_input_annotations(
    graph: Graph,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    """将 npu_input() 标注写入 model_input Tensor。"""
    input_tensors = [t for t in graph.tensors.values() if t.is_model_input]
    input_objects = [dummy_input] + ([mask] if mask is not None else [])
    for tensor_ir, input_obj in zip(input_tensors, input_objects):
        spec = get_input_annotation(input_obj)
        if spec is None:
            continue
        tensor_ir.src_dtype = tensor_ir.dtype
        tensor_ir.dtype = spec.dtype
        tensor_ir.format = spec.format
        logger.debug("输入标注 %s: %s -> %s", tensor_ir.id, tensor_ir.src_dtype, spec)


def _apply_npu_annotations(
    graph: Graph,
    model: nn.Module,
    dummy_input: torch.Tensor,
    mask: torch.Tensor | None,
    ep: Any,
    fx_to_nid: dict[str, str],
) -> None:
    """将 npu() / npu_input() 标注传播到 Graph IR。

    模块标注：通过 nn_module_stack 匹配 FX 节点到模块路径，
    将完整标注写入 Node.params["_npu"]（供 format_annotator 等下游读取），
    compute_dtype 同时写入 Node.params["compute_dtype"]。
    输入标注：按 USER_INPUT placeholder 顺序匹配。
    """
    annotations = get_npu_annotations(model)
    _apply_module_annotations(graph, annotations, ep, fx_to_nid)
    _apply_weight_annotations(graph, annotations, model)
    _apply_input_annotations(graph, dummy_input, mask)
    _resolve_negative_dims(graph)


def _resolve_negative_dims(graph: Graph) -> None:
    """将 dim 参数的负索引转正，并将 softmax 的 dim 从索引转为维度大小。"""
    for node in graph.nodes.values():
        dim_val = node.params.get("dim")
        if dim_val is None or not isinstance(dim_val, int):
            continue
        if not node.inputs:
            continue
        t = graph.get_tensor(node.inputs[0])
        if t is None or not t.shape:
            continue
        ndim = len(t.shape)
        if dim_val < 0:
            dim_val = dim_val + ndim
        if node.op_type in DIM_TO_SIZE_OPS:
            node.params["dim"] = t.shape[dim_val]
        else:
            node.params["dim"] = dim_val
