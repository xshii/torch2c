"""内置 Python 级 cost function — 比 YAML 更精确的代价估算。

通过 @register_cost_fn 注册后，优先于 YAML 配置。
适合需要根据 format/storage/fusion 等上下文动态调整代价的算子。

所有信息通过 ctx.inputs / ctx.outputs 访问::

    ctx.inputs[0].shape     # [B, S, D]
    ctx.inputs[0].format    # "zz" / "nd" / "nz"
    ctx.inputs[0].storage   # "hbm" / "local" / "pipe"
    ctx.inputs[0].dtype     # "fp16" / "fp32"
    ctx.compute_dtype       # 计算精度
    ctx.is_fused            # 是否在融合组内
    ctx.elem_count          # 第一个输出的元素数

用法示例::

    @register_cost_fn("my_custom_op")
    def _my_op_cost(ctx: CostContext) -> CostResult:
        flops = ctx.elem_count * 10
        launch = 50 if ctx.is_fused else 100
        return CostResult(flops=flops, launch_cycles=launch)
"""

from functools import reduce
from operator import mul

from torch2c.optpass.cd_roofline.roofline_analyzer import (
    CostContext,
    CostResult,
    register_cost_fn,
)


def _prod(shape: list[int]) -> int:
    return reduce(mul, shape, 1)


@register_cost_fn("cube_matmul")
def _cube_matmul_cost(ctx: CostContext) -> CostResult:
    """cube_matmul: 2*B*M*N*K，ZZ 格式比 ND 启动更快。"""
    a, b = ctx.inputs[0], ctx.inputs[1]
    M = a.shape[-2] if len(a.shape) >= 2 else 1
    K = a.shape[-1] if len(a.shape) >= 1 else 1
    N = b.shape[-1] if len(b.shape) >= 1 else 1
    batch = _prod(a.shape[:-2]) if len(a.shape) > 2 else 1
    flops = 2 * batch * M * N * K
    # ZZ 格式是硬件原生，管线填充更快
    launch = 80 if a.format == "zz" else 100
    return CostResult(flops=flops, launch_cycles=launch)


@register_cost_fn("cube_matmul_bias")
def _cube_matmul_bias_cost(ctx: CostContext) -> CostResult:
    """cube_matmul_bias: matmul + bias 累加，略高于纯 matmul。"""
    a, b = ctx.inputs[0], ctx.inputs[1]
    M = a.shape[-2] if len(a.shape) >= 2 else 1
    K = a.shape[-1] if len(a.shape) >= 1 else 1
    N = b.shape[-1] if len(b.shape) >= 1 else 1
    batch = _prod(a.shape[:-2]) if len(a.shape) > 2 else 1
    flops = 2 * batch * M * N * K + ctx.elem_count
    launch = 90 if a.format == "zz" else 110
    return CostResult(flops=flops, launch_cycles=launch)
