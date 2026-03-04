"""图捕获：将 PyTorch 模型导出为 Graph IR。

使用 torch.export 将 nn.Module 导出为 ATen IR，然后转换为 Graph IR。
保留高级算子（layer_norm, softmax），标记权重和输入输出张量。
"""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch.nn as nn
from torch.export import export

from ..common.graph_ir import Graph, Node, Tensor
from ..common.logger import get_logger

logger = get_logger(__name__)

# graph_capture 产出的 positional param 名 → codegen 期望的 param 名
_PARAM_RENAMES: dict[str, dict[str, str]] = {
    "aten.transpose.int": {"p0": "dim0", "p1": "dim1"},
    "aten._softmax.default": {"p0": "dim"},
    "aten.native_layer_norm.default": {"p1": "epsilon", "eps": "epsilon"},
}

_DTYPE_MAP = {
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.bfloat16: "bf16",
}


def _dtype_str(dt: torch.dtype) -> str:
    """将 torch dtype 转换为字符串标识。"""
    return _DTYPE_MAP.get(dt, str(dt).replace("torch.", ""))


def _op_name(target: Any) -> str:
    """提取 ATen 算子全称，如 'aten.mm.default'。"""
    s = str(target)
    return s[len("torch.ops."):] if s.startswith("torch.ops.") else s


def _is_tensor_overload(op_name: str) -> bool:
    """判断算子重载是否期望全部 Tensor 参数（如 aten.mul.Tensor）。"""
    parts = op_name.rsplit(".", 1)
    return len(parts) >= 2 and parts[-1] == "Tensor"


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
        return Tensor(id=tid, shape=list(val.shape),
                      dtype=_dtype_str(val.dtype), **kwargs)
    return Tensor(id=tid, shape=[1], dtype="fp32", **kwargs)


# ---- 主入口 ----

def capture(model: nn.Module, dummy_input: torch.Tensor,
            mask: torch.Tensor | None = None) -> Graph:
    """将 PyTorch 模型导出为 Graph IR。

    Args:
        model: PyTorch 模型。
        dummy_input: 固定 shape 的样例输入。
        mask: 可选 attention mask。

    Returns:
        Graph IR，节点 op_type 为 ATen 算子全称。
    """
    graph = Graph()
    args = (dummy_input, mask) if mask is not None else (dummy_input,)

    ep = export(model, args)

    # 区分参数/权重 placeholder 和用户输入 placeholder
    # 映射 FX placeholder name → state_dict key (spec.target)
    param_names: dict[str, str] = {}
    for spec in ep.graph_signature.input_specs:
        if spec.kind.name != "USER_INPUT":
            param_names[spec.arg.name] = spec.target

    fx_map: dict[str, str | list[str]] = {}
    tgen = _IdGen("t")
    ngen = _IdGen("node")

    for fx_node in ep.graph_module.graph.nodes:
        if fx_node.op == "placeholder":
            _handle_placeholder(graph, fx_node, fx_map, tgen, param_names)
        elif fx_node.op == "call_function":
            if fx_node.target is operator.getitem:
                _handle_getitem(fx_node, fx_map)
            else:
                _handle_call(graph, fx_node, fx_map, tgen, ngen)
        elif fx_node.op == "output":
            _handle_output(graph, fx_node, fx_map)

    # 负索引 dim 解析 & softmax dim → size 转换
    _resolve_negative_dims(graph)

    # 校验图完整性
    errors = graph.validate()
    if errors:
        for err in errors:
            logger.warning("图校验警告: %s", err)

    weight_count = sum(1 for t in graph.tensors.values() if t.is_weight)
    logger.info("图捕获完成，节点数: %d, tensor数: %d, 权重tensor数: %d",
                len(graph.nodes), len(graph.tensors), weight_count)
    return graph


# ---- dim 解析 ----

_DIM_TO_SIZE_OPS = {"aten._softmax.default"}


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
        if node.op_type in _DIM_TO_SIZE_OPS:
            node.params["dim"] = t.shape[dim_val]
        else:
            node.params["dim"] = dim_val


# ---- 内部处理函数 ----

def _handle_placeholder(graph, fx_node, fx_map, tgen, param_names):
    """处理 placeholder 节点（模型输入或参数）。"""
    tid = tgen.next()
    val = fx_node.meta.get("val")
    is_weight = fx_node.name in param_names
    t = _make_tensor(tid, val, is_weight=is_weight,
                     is_model_input=not is_weight)
    if is_weight:
        t.name = param_names[fx_node.name]
    graph.add_tensor(t)
    fx_map[fx_node.name] = tid


def _handle_getitem(fx_node, fx_map):
    """处理 getitem 节点（多输出算子的索引访问）。"""
    source = fx_node.args[0]
    index = fx_node.args[1]
    source_tids = fx_map.get(source.name)
    if isinstance(source_tids, list) and index < len(source_tids):
        fx_map[fx_node.name] = source_tids[index]


def _parse_call_args(
    graph: Graph, fx_node, fx_map: dict, tgen: _IdGen, nid: str,
    op: str, tensor_overload: bool,
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
            scalar_t = Tensor(id=scalar_tid, shape=[1], dtype="fp32",
                              is_weight=True)
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
    op: str, input_tids: list[str], params: dict,
) -> tuple[list[str], dict]:
    """addmm 输入重排 + 参数重命名。"""
    if op == "aten.addmm.default" and len(input_tids) >= 3:
        input_tids = [input_tids[1], input_tids[2], input_tids[0]]

    renames = _PARAM_RENAMES.get(op)
    if renames:
        for old_key, new_key in renames.items():
            if old_key in params and new_key not in params:
                params[new_key] = params.pop(old_key)

    return input_tids, params


def _create_call_outputs(
    graph: Graph, fx_node, fx_map: dict, tgen: _IdGen, nid: str,
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


def _handle_call(graph, fx_node, fx_map, tgen, ngen):
    """处理 call_function 节点（ATen 算子调用）。"""
    op = _op_name(fx_node.target)
    nid = ngen.next()
    tensor_overload = _is_tensor_overload(op)

    input_tids, params = _parse_call_args(
        graph, fx_node, fx_map, tgen, nid, op, tensor_overload,
    )
    input_tids, params = _normalize_op_inputs(op, input_tids, params)
    output_tids = _create_call_outputs(graph, fx_node, fx_map, tgen, nid)

    for itid in input_tids:
        t = graph.get_tensor(itid)
        if t and nid not in t.consumer_node_ids:
            t.consumer_node_ids.append(nid)

    node = Node(id=nid, op_type=op, inputs=input_tids,
                outputs=output_tids, params=params)
    graph.add_node(node)
    logger.debug("节点 %s: op=%s, inputs=%s, outputs=%s",
                 nid, op, input_tids, output_tids)


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


def _handle_output(graph, fx_node, fx_map):
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
