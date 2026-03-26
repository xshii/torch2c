"""TODO_PASS_NAME — TODO_PASS编号：一句话描述这个 pass 的作用。

详细说明（可多行）：
  - 这个 pass 在什么阶段运行
  - 它对 graph 做了什么变换
  - 有什么前置/后置依赖

配置示例（hardware_config.yaml）：
  TODO_config_key:
    TODO_param: TODO_value
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────
# TODO: 定义硬编码的兜底默认值（config 为空时使用）
_DEFAULT_XXX = 16


# ── 辅助函数 ─────────────────────────────────────────────────────────
# 每个函数 < 50 行。复杂逻辑拆成多个小函数。


def _parse_config(config: dict) -> dict:
    """从 config 字典提取本 pass 需要的参数。

    TODO: 根据你的 pass 需要，解析 config 中的字段。
    返回解析后的参数（可以是 dict、dataclass 等）。
    """
    # TODO: 替换为实际的 config 解析逻辑
    # 示例：
    #   threshold = config.get("threshold", _DEFAULT_XXX)
    #   return {"threshold": threshold}
    return config


def _should_transform(node_or_tensor, params: dict) -> bool:
    """判断某个 node/tensor 是否需要变换。

    TODO: 实现匹配条件。返回 True 表示需要变换。
    """
    # TODO: 替换为实际的匹配逻辑
    # 示例：
    #   return node_or_tensor.op_type == "target_op"
    return False


def _apply_transform(graph: Graph, node_or_tensor, params: dict) -> bool:
    """对单个 node/tensor 执行变换。

    TODO: 实现具体的变换逻辑。
    返回 True 表示发生了变换（用于计数和日志）。
    """
    # TODO: 替换为实际的变换逻辑
    # 示例：
    #   old_value = node_or_tensor.some_field
    #   node_or_tensor.some_field = new_value
    #   return old_value != new_value
    return False


# ── 主入口 ────────────────────────────────────────────────────────────


def run(graph: Graph, config: dict) -> Graph:
    """Pass 入口：对 graph 执行 TODO_描述 变换。

    Args:
        graph: 上游 pass 产出的 Graph IR。
        config: TODO_config_key 配置字典。

    Returns:
        同一 Graph 对象（原地修改）。
    """
    # 第一步：解析配置
    params = _parse_config(config)

    # 第二步：遍历 graph，执行变换
    transform_count = 0

    # TODO: 选择遍历 nodes 还是 tensors（或两者都遍历）
    # 遍历 nodes 的模式：
    for node in graph.nodes.values():
        if not _should_transform(node, params):
            continue
        if _apply_transform(graph, node, params):
            transform_count += 1
            # 第三步：用 log_opt 记录每个变换的原因（必须！）
            log_opt(
                node, "TODO_PASS_NAME", "TODO_动作描述",
                f"TODO_详细原因说明",
            )

    # 遍历 tensors 的模式（按需取消注释）：
    # for t in graph.tensors.values():
    #     if not _should_transform(t, params):
    #         continue
    #     if _apply_transform(graph, t, params):
    #         transform_count += 1
    #         producer = graph.get_node(t.producer_node_id) if t.producer_node_id else None
    #         if producer:
    #             log_opt(
    #                 producer, "TODO_PASS_NAME", "TODO_动作描述",
    #                 f"TODO_详细原因说明",
    #             )

    logger.info("TODO_PASS_NAME 完成: %d 处变换", transform_count)
    return graph


def post_validate(graph: Graph, config: dict | None = None) -> list[str]:
    """校验 pass 执行后的不变量。

    返回错误消息列表（空列表 = 全部通过）。
    注意：这里校验的是「变换后图应满足的约束」，不是重复 run 的逻辑。

    TODO: 实现后置校验。典型模式：
      - 遍历所有 node/tensor
      - 检查某个属性是否满足预期
      - 不满足则 append 错误消息
    """
    errors: list[str] = []

    # TODO: 替换为实际的校验逻辑
    # 示例：
    # for t in graph.tensors.values():
    #     if t.some_field is None:
    #         errors.append(f"tensor {t.id}: some_field 不应为 None")

    return errors
