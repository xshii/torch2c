"""内置 Python 级 cost function — 比 YAML 更精确的代价估算。

通过 @register_cost_fn 注册后，优先于 YAML 配置。
适合需要根据 format/storage/fusion 等上下文动态调整代价的算子。

用法示例::

    from torch2c.optpass.cd_roofline.roofline_analyzer import (
        CostContext, CostResult, register_cost_fn,
    )

    @register_cost_fn("my_custom_op")
    def _my_op_cost(ctx: CostContext) -> CostResult:
        # ctx.inputs[0].shape, ctx.input_formats, ctx.is_fused, ...
        flops = ctx.elem_count * 10
        launch = 50 if ctx.is_fused else 100
        return CostResult(flops=flops, launch_cycles=launch)
"""

from torch2c.optpass.cd_roofline.roofline_analyzer import (
    CostContext,
    CostResult,
    register_cost_fn,
)


@register_cost_fn("cube_matmul")
def _cube_matmul_cost(ctx: CostContext) -> CostResult:
    """cube_matmul: 2*B*M*N*K，ZZ 格式比 ND 启动更快。"""
    flops = 2 * ctx.batch * ctx.M * ctx.N * ctx.K
    # ZZ 格式是硬件原生，管线填充更快
    if ctx.input_formats and ctx.input_formats[0] == "zz":
        launch = 80
    else:
        launch = 100
    return CostResult(flops=flops, launch_cycles=launch)


@register_cost_fn("cube_matmul_bias")
def _cube_matmul_bias_cost(ctx: CostContext) -> CostResult:
    """cube_matmul_bias: matmul + bias 累加，略高于纯 matmul。"""
    flops = 2 * ctx.batch * ctx.M * ctx.N * ctx.K + ctx.elem_count
    if ctx.input_formats and ctx.input_formats[0] == "zz":
        launch = 90
    else:
        launch = 110
    return CostResult(flops=flops, launch_cycles=launch)
