"""torch_debug — PyTorch 侧维测：记录每个子模块 forward 的 tensor 特征。

用法：
    from torch2c.common.torch_debug import enable_trace, disable_trace, flush_trace

    enable_trace(model)           # 注册 hook
    out = model(x, mask)          # 正常推理, 自动记录
    flush_trace()                 # 打印所有记录
    disable_trace(model)          # 移除 hook

    # 或者用 context manager
    with trace_context(model):
        out = model(x, mask)
    # 自动 flush + disable

输出格式 (每行一条):
    #0  self_attn.q_proj  [1,32,256]fp32  →  [1,32,256]fp32
        in_0:  min=-2.35  max=+1.89  mean=+0.003  std=0.98  nan=0  inf=0
        out_0: min=-5.12  max=+4.77  mean=-0.01   std=1.42  nan=0  inf=0
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# ---- 全局状态 ----

_enabled = False
_records: list[_TraceRecord] = []
_hooks: list[torch.utils.hooks.RemovableHook] = []
_seq = 0


@dataclass
class _TensorStats:
    shape: list[int]
    dtype: str
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    std_val: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    zero_count: int = 0
    numel: int = 0


@dataclass
class _TraceRecord:
    seq: int
    module_path: str
    module_type: str
    inputs: list[_TensorStats] = field(default_factory=list)
    outputs: list[_TensorStats] = field(default_factory=list)


def _compute_stats(t: torch.Tensor) -> _TensorStats:
    """对 tensor 计算 fp32 精度的统计特征。"""
    f = t.detach().float().cpu()
    nan_mask = torch.isnan(f)
    inf_mask = torch.isinf(f)
    valid = f[~nan_mask & ~inf_mask]

    return _TensorStats(
        shape=list(t.shape),
        dtype=str(t.dtype).replace("torch.", ""),
        min_val=float(valid.min()) if valid.numel() > 0 else 0.0,
        max_val=float(valid.max()) if valid.numel() > 0 else 0.0,
        mean_val=float(valid.mean()) if valid.numel() > 0 else 0.0,
        std_val=float(valid.std()) if valid.numel() > 1 else 0.0,
        nan_count=int(nan_mask.sum()),
        inf_count=int(inf_mask.sum()),
        zero_count=int((f == 0).sum()),
        numel=int(t.numel()),
    )


def _collect_tensors(args) -> list[_TensorStats]:
    """从 forward 的输入/输出中提取 tensor 统计。"""
    stats = []
    if isinstance(args, torch.Tensor):
        stats.append(_compute_stats(args))
    elif isinstance(args, (tuple, list)):
        for a in args:
            if isinstance(a, torch.Tensor):
                stats.append(_compute_stats(a))
    return stats


def _make_hook(module_path: str, module_type: str):
    """创建 forward hook。"""
    def hook(module, input, output):
        global _seq
        if not _enabled:
            return
        rec = _TraceRecord(
            seq=_seq,
            module_path=module_path,
            module_type=module_type,
            inputs=_collect_tensors(input),
            outputs=_collect_tensors(output),
        )
        _records.append(rec)
        _seq += 1
    return hook


# ---- 格式化输出 ----

def _fmt_stats(prefix: str, s: _TensorStats) -> str:
    shape_str = "x".join(str(d) for d in s.shape)
    return (
        f"  {prefix:8s} [{shape_str}]{s.dtype:8s} "
        f"min={s.min_val:+.4e}  max={s.max_val:+.4e}  "
        f"mean={s.mean_val:+.4e}  std={s.std_val:.4e}  "
        f"nan={s.nan_count}  inf={s.inf_count}  zero={s.zero_count}"
    )


def _fmt_record(r: _TraceRecord) -> str:
    lines = [f"#{r.seq:<4d} {r.module_path}  ({r.module_type})"]
    for i, s in enumerate(r.inputs):
        lines.append(_fmt_stats(f"in_{i}:", s))
    for i, s in enumerate(r.outputs):
        lines.append(_fmt_stats(f"out_{i}:", s))
    return "\n".join(lines)


# ---- Public API ----

def enable_trace(model: nn.Module, *, leaf_only: bool = False) -> None:
    """为模型注册 forward hook，开始记录。

    Args:
        model: 要 trace 的模型。
        leaf_only: True 时只 hook 叶子模块（无子模块的），
                   False 时 hook 所有带 _torch2c 标注的模块 + 所有叶子模块。
    """
    global _enabled, _seq
    disable_trace(model)  # 清理旧 hook
    _enabled = True
    _seq = 0
    _records.clear()

    for name, mod in model.named_modules():
        if not name:
            continue  # skip root
        has_ann = hasattr(mod, "_torch2c")
        is_leaf = len(list(mod.children())) == 0

        if leaf_only and not is_leaf:
            continue
        if not leaf_only and not has_ann and not is_leaf:
            continue

        h = mod.register_forward_hook(_make_hook(name, type(mod).__name__))
        _hooks.append(h)


def disable_trace(model: nn.Module) -> None:
    """移除所有 hook，停止记录。"""
    global _enabled
    _enabled = False
    for h in _hooks:
        h.remove()
    _hooks.clear()


def flush_trace(file=None) -> list[_TraceRecord]:
    """打印并返回所有记录，然后清空。"""
    if file is None:
        file = sys.stderr
    if _records:
        print(f"\n===== TORCH DEBUG TRACE ({len(_records)} entries) =====", file=file)
        for r in _records:
            print(_fmt_record(r), file=file)
        print("===== END TORCH DEBUG TRACE =====\n", file=file)
    result = list(_records)
    _records.clear()
    return result


def get_records() -> list[_TraceRecord]:
    """获取当前记录（不清空）。"""
    return list(_records)


@contextmanager
def trace_context(model: nn.Module, *, leaf_only: bool = False, file=None):
    """Context manager: 自动 enable → yield → flush → disable。"""
    enable_trace(model, leaf_only=leaf_only)
    try:
        yield
    finally:
        flush_trace(file=file)
        disable_trace(model)
