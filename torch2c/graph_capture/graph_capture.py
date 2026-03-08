"""图捕获：将 PyTorch 模型（通过 torch.export）导出为 Graph IR。"""

from __future__ import annotations

import operator
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch.export import export

from ..common.graph_ir import Graph, Node, Tensor
from ..common.logger import get_logger
from ._annotations import _apply_npu_annotations
from ._constants import (
    WEIGHT_TRANSPOSE_INPUT,
    dtype_str,
    is_tensor_overload,
    normalize_op_inputs,
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

    # 映射 FX placeholder name → state_dict key (区分权重/用户输入)
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

    # npu() 标注传播 + 负索引 dim 解析
    _apply_npu_annotations(graph, model, dummy_input, mask, ep, fx_to_nid)

    # 校验图完整性
    errors = graph.validate()
    if errors:
        for err in errors:
            logger.warning("图校验警告: %s", err)

    n_weights = sum(1 for t in graph.tensors.values() if t.is_weight)
    logger.info("图捕获完成，节点: %d, tensor: %d, 权重: %d",
                len(graph.nodes), len(graph.tensors), n_weights)
    return graph


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
    """为 linear weight [out,in] 插入 transpose 节点，转为 [in,out]。"""
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
    input_tids, params = normalize_op_inputs(op, input_tids, params)

    # bmm: 从输入 shape 推断 batch 维度，注入 loop 参数
    if op == "aten.bmm.default" and "loop" not in params and input_tids:
        in_t = graph.get_tensor(input_tids[0])
        if in_t and len(in_t.shape) == 3:
            params["loop"] = in_t.shape[0]

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
