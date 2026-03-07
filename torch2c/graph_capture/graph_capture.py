"""图捕获：将 PyTorch 模型（通过 torch.export）导出为 Graph IR。"""

from __future__ import annotations

import operator
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch.export import export

from ..common.graph_ir import Graph, Node, Tensor
from ..common.logger import get_logger
from ..common.npu_annotate import NpuSpec, get_input_annotation, get_npu_annotations
from ._constants import (
    DIM_TO_SIZE_OPS,
    INPUT_REORDER,
    PARAM_DEFAULTS,
    PARAM_RENAMES,
    WEIGHT_TRANSPOSE_INPUT,
    dtype_str,
    is_tensor_overload,
    op_name,
)

logger = get_logger(__name__)

_FxMap: TypeAlias = "dict[str, str | list[str]]"


class _IdGen:
    """顺序 ID 生成器。"""

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._count = 0

    def next(self) -> str:
        tid = f"{self._prefix}_{self._count}"
        self._count += 1
        return tid


def _make_tensor(tid: str, val: Any, **kwargs) -> Tensor:
    """根据 FakeTensor 元信息创建 Tensor 对象。"""
    if isinstance(val, torch.Tensor):
        return Tensor(id=tid, shape=list(val.shape), dtype=dtype_str(val.dtype), **kwargs)
    return Tensor(id=tid, shape=[1], dtype="fp32", **kwargs)


def capture(model: nn.Module, dummy_input: torch.Tensor, mask: torch.Tensor | None = None) -> Graph:
    """将 PyTorch 模型导出为 Graph IR。"""
    graph = Graph()
    args = (dummy_input, mask) if mask is not None else (dummy_input,)

    ep = export(model, args)

    # 区分参数/权重 placeholder 和用户输入 placeholder
    # 映射 FX placeholder name → state_dict key (spec.target)
    param_names: dict[str, str] = {}
    for spec in ep.graph_signature.input_specs:
        if spec.kind.name != "USER_INPUT":
            param_names[spec.arg.name] = spec.target

    fx_map: _FxMap = {}
    fx_to_nid: dict[str, str] = {}  # fx_node.name → graph node id
    tgen = _IdGen("t")
    ngen = _IdGen("node")

    for fx_node in ep.graph_module.graph.nodes:
        if fx_node.op == "placeholder":
            _handle_placeholder(graph, fx_node, fx_map, tgen, param_names)
        elif fx_node.op == "call_function":
            if fx_node.target is operator.getitem:
                _handle_getitem(fx_node, fx_map)
            else:
                _handle_call(graph, fx_node, fx_map, tgen, ngen, fx_to_nid)
        elif fx_node.op == "output":
            _handle_output(graph, fx_node, fx_map)

    # npu() 标注传播：通过 nn_module_stack 匹配模块，写入 Node.params["_npu"]
    _apply_npu_annotations(graph, model, dummy_input, mask, ep, fx_to_nid)

    # 负索引 dim 解析 & softmax dim → size 转换
    _resolve_negative_dims(graph)

    # 校验图完整性
    errors = graph.validate()
    if errors:
        for err in errors:
            logger.warning("图校验警告: %s", err)

    weight_count = sum(1 for t in graph.tensors.values() if t.is_weight)
    logger.info(
        "图捕获完成，节点数: %d, tensor数: %d, 权重tensor数: %d",
        len(graph.nodes),
        len(graph.tensors),
        weight_count,
    )
    return graph


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

    # ---- 模块标注 → 通过 nn_module_stack 匹配节点 ----
    if annotations:
        for fx_node in ep.graph_module.graph.nodes:
            if fx_node.op != "call_function" or fx_node.name not in fx_to_nid:
                continue
            nn_stack = fx_node.meta.get("nn_module_stack")
            if not nn_stack:
                continue
            # 取最内层模块路径 (value[0] 是 module path 字符串)
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
            logger.debug("模块标注 %s → node %s: %s", innermost_path, nid, ann)

    # ---- 权重标注 → weight Tensor 的 dtype/format ----
    if annotations:
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
                # 保留原始 dtype（来自 .pth / torch.export）
                # format 无需保留：PyTorch 没有 format 概念，源端始终是 nd
                t.src_dtype = t.dtype
                # 目标 dtype/format（NPU 上的存储）
                t.dtype = spec.dtype
                t.format = spec.format
            logger.debug("权重标注 %s: %s → %s", t.name, t.src_dtype, spec)

    # ---- 输入标注 → model_input Tensor ----
    input_tensors = [t for t in graph.tensors.values() if t.is_model_input]
    input_objects = [dummy_input] + ([mask] if mask is not None else [])
    for tensor_ir, input_obj in zip(input_tensors, input_objects):
        spec = get_input_annotation(input_obj)
        if spec is None:
            continue
        tensor_ir.src_dtype = tensor_ir.dtype
        tensor_ir.dtype = spec.dtype
        tensor_ir.format = spec.format
        logger.debug("输入标注 %s: %s → %s", tensor_ir.id, tensor_ir.src_dtype, spec)


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


def _handle_placeholder(
    graph: Graph, fx_node: torch.fx.Node, fx_map: _FxMap, tgen: _IdGen, param_names: dict[str, str],
) -> None:
    """处理 placeholder 节点（模型输入或参数）。"""
    tid = tgen.next()
    val = fx_node.meta.get("val")
    is_weight = fx_node.name in param_names
    t = _make_tensor(tid, val, is_weight=is_weight, is_model_input=not is_weight)
    if is_weight:
        t.name = param_names[fx_node.name]
    graph.add_tensor(t)
    fx_map[fx_node.name] = tid


def _handle_getitem(fx_node: torch.fx.Node, fx_map: _FxMap) -> None:
    """处理 getitem 节点（多输出算子的索引访问）。"""
    source = fx_node.args[0]
    index = fx_node.args[1]
    source_tids = fx_map.get(source.name)
    if isinstance(source_tids, list) and index < len(source_tids):
        fx_map[fx_node.name] = source_tids[index]


def _parse_call_args(
    graph: Graph,
    fx_node,
    fx_map: dict,
    tgen: _IdGen,
    nid: str,
    op: str,
    tensor_overload: bool,
) -> tuple[list[str], dict]:
    """解析 FX call 的 args/kwargs，返回 (input_tids, params)。"""
    input_tids: list[str] = []
    params: dict = {}
    pi = 0

    for i, arg in enumerate(fx_node.args):
        if isinstance(arg, torch.fx.Node):
            resolved = fx_map.get(arg.name)
            if isinstance(resolved, str):
                input_tids.append(resolved)
            elif isinstance(resolved, list):
                input_tids.append(resolved[0])
        elif isinstance(arg, (int, float)) and tensor_overload and i > 0:
            scalar_tid = tgen.next()
            scalar_t = Tensor(id=scalar_tid, shape=[1], dtype="fp32", is_weight=True)
            scalar_t.consumer_node_ids.append(nid)
            graph.add_tensor(scalar_t)
            input_tids.append(scalar_tid)
        elif arg is None:
            continue
        else:
            if isinstance(arg, (int, float, bool, str)):
                params[f"p{pi}"] = arg
            elif isinstance(arg, (list, tuple)):
                params[f"p{pi}"] = list(arg)
            pi += 1

    for k, v in fx_node.kwargs.items():
        if isinstance(v, (int, float, bool, str)):
            params[k] = v
        elif isinstance(v, (list, tuple)):
            params[k] = list(v)

    return input_tids, params


def _normalize_op_inputs(
    op: str,
    input_tids: list[str],
    params: dict,
) -> tuple[list[str], dict]:
    """输入重排 + 参数重命名（规则来自 capture_rules.yaml）。"""
    reorder = INPUT_REORDER.get(op)
    if reorder and len(input_tids) >= len(reorder):
        input_tids = [input_tids[i] for i in reorder]

    renames = PARAM_RENAMES.get(op)
    if renames:
        for old_key, new_key in renames.items():
            if old_key in params and new_key not in params:
                params[new_key] = params.pop(old_key)

    # 缺失参数补充默认值
    defaults = PARAM_DEFAULTS.get(op)
    if defaults:
        for key, val in defaults.items():
            if key not in params:
                params[key] = val

    return input_tids, params


def _create_call_outputs(
    graph: Graph,
    fx_node,
    fx_map: dict,
    tgen: _IdGen,
    nid: str,
) -> list[str]:
    """创建输出 tensor 并更新 fx_map，返回 output_tids。"""
    val = fx_node.meta.get("val")
    output_tids: list[str] = []

    if isinstance(val, (tuple, list)):
        tid_list: list[str] = []
        for v in val:
            out_tid = tgen.next()
            out_t = _make_tensor(out_tid, v, producer_node_id=nid)
            graph.add_tensor(out_t)
            output_tids.append(out_tid)
            tid_list.append(out_tid)
        fx_map[fx_node.name] = tid_list
    else:
        out_tid = tgen.next()
        out_t = _make_tensor(out_tid, val, producer_node_id=nid)
        graph.add_tensor(out_t)
        output_tids.append(out_tid)
        fx_map[fx_node.name] = out_tid

    return output_tids


def _insert_weight_transpose(
    graph: Graph, input_tids: list[str], weight_idx: int, tgen: _IdGen, ngen: _IdGen,
) -> list[str]:
    """为 linear 等算子的 weight 输入插入 transpose_2d 节点。

    PyTorch nn.Linear weight shape = [out, in]，需转置为 [in, out]
    才能用 npu_matmul_bias(a @ b + bias) 正确计算。
    """
    weight_tid = input_tids[weight_idx]
    weight_t = graph.get_tensor(weight_tid)
    if weight_t is None or len(weight_t.shape) != 2:
        return input_tids

    tr_nid = ngen.next()
    tr_out_tid = tgen.next()
    transposed_shape = [weight_t.shape[1], weight_t.shape[0]]
    tr_out = Tensor(
        id=tr_out_tid, shape=transposed_shape, dtype=weight_t.dtype,
        producer_node_id=tr_nid,
    )
    graph.add_tensor(tr_out)

    weight_t.consumer_node_ids.append(tr_nid)
    tr_node = Node(
        id=tr_nid, op_type="aten.t.default",
        inputs=[weight_tid], outputs=[tr_out_tid], params={},
    )
    graph.add_node(tr_node)

    new_tids = list(input_tids)
    new_tids[weight_idx] = tr_out_tid
    return new_tids


def _handle_call(
    graph: Graph, fx_node: torch.fx.Node, fx_map: _FxMap, tgen: _IdGen, ngen: _IdGen,
    fx_to_nid: dict[str, str] | None = None,
) -> None:
    """处理 call_function 节点（ATen 算子调用）。"""
    op = op_name(fx_node.target)
    nid = ngen.next()
    if fx_to_nid is not None:
        fx_to_nid[fx_node.name] = nid
    tensor_overload = is_tensor_overload(op)

    input_tids, params = _parse_call_args(
        graph,
        fx_node,
        fx_map,
        tgen,
        nid,
        op,
        tensor_overload,
    )
    input_tids, params = _normalize_op_inputs(op, input_tids, params)

    # linear 等算子需要对 weight 做转置
    wt_idx = WEIGHT_TRANSPOSE_INPUT.get(op)
    if wt_idx is not None and wt_idx < len(input_tids):
        input_tids = _insert_weight_transpose(graph, input_tids, wt_idx, tgen, ngen)

    output_tids = _create_call_outputs(graph, fx_node, fx_map, tgen, nid)

    for itid in input_tids:
        t = graph.get_tensor(itid)
        if t and nid not in t.consumer_node_ids:
            t.consumer_node_ids.append(nid)

    # 从 nn_module_stack 提取模块路径
    module_path = None
    nn_stack = fx_node.meta.get("nn_module_stack")
    if nn_stack:
        entries = list(nn_stack.values())
        if len(entries) > 1:
            module_path = entries[-1][0]  # 最内层模块路径

    node = Node(id=nid, op_type=op, inputs=input_tids, outputs=output_tids,
                params=params, module_path=module_path)
    graph.add_node(node)
    logger.debug("节点 %s: op=%s, module=%s, inputs=%s, outputs=%s",
                 nid, op, module_path, input_tids, output_tids)


def post_validate(graph: Graph) -> list[str]:
    """graph_capture 后的校验：权重需要 name，输入需要 dtype/shape。"""
    errors: list[str] = []
    for t in graph.tensors.values():
        if t.is_weight and not t.name:
            errors.append(f"权重 tensor {t.id} 缺少 name 字段")
        if t.is_model_input and not t.dtype:
            errors.append(f"模型输入 tensor {t.id} 缺少 dtype")
        if t.is_model_input and not t.shape:
            errors.append(f"模型输入 tensor {t.id} 缺少 shape")
    for n in graph.nodes.values():
        if not n.op_type:
            errors.append(f"节点 {n.id} 缺少 op_type")
    return errors


def _handle_output(graph: Graph, fx_node: torch.fx.Node, fx_map: _FxMap) -> None:
    """处理 output 节点，标记模型输出张量。"""
    output_args = fx_node.args[0] if fx_node.args else ()
    if isinstance(output_args, torch.fx.Node):
        output_args = (output_args,)
    if not isinstance(output_args, (tuple, list)):
        return
    for arg in output_args:
        if isinstance(arg, torch.fx.Node):
            tid = fx_map.get(arg.name)
            if isinstance(tid, str):
                t = graph.get_tensor(tid)
                if t:
                    t.is_model_output = True
