"""common.constants — 跨模块共享的常量。"""

# cube matmul 算子名集合，被 op_absorption 和 memory_planner 共用
MATMUL_OPS: set[str] = {"cube_matmul", "cube_matmul_bias"}
