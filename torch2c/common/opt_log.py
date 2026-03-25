"""opt_log — 优化决策日志，记录在 node.params["_opt_log"] 中。

每个 pass 调用 log_opt() 在节点上追加一条决策记录，
随 Graph 序列化保存，可视化时展示。

用法:
    from torch2c.common.opt_log import log_opt
    log_opt(node, "format_planner", "格式变更", "nd → nz: cube src1 需要 nz")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch2c.common.graph_ir import Node


def log_opt(node: Node, pass_name: str, action: str, detail: str) -> None:
    """在节点上记录一条优化决策。

    Args:
        node: 被修改的节点。
        pass_name: pass 名称（如 "format_planner"）。
        action: 操作类型（如 "格式变更"、"算子映射"、"参数吸收"）。
        detail: 人类可读的原因说明。
    """
    node.params.setdefault("_opt_log", []).append({
        "pass": pass_name,
        "action": action,
        "detail": detail,
    })
