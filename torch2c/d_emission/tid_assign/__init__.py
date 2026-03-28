"""tid_assign — Pass ⑧b：全局 TID 重分配。

将 DMA 指令纳入 TidInfo 依赖链，形成全局单条主路径：
  load_0 → load_1 → ... → compute → store_0 → store_1 → ... → (下一个 op 的 load)
"""

from .tid_assign import run

__all__ = ["run"]
