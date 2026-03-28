"""DemoEncoder — 2 层 Transformer Encoder 演示模型。

hidden_size=64, num_heads=4, ffn_dim=256, seq_len=32, batch=1
forward(x, mask) — mask shape=[1, 1, 32, 32]

用法：
    # 全管线编译（compile_model.py 加载本文件）
    python scripts/compile_model.py torch2c/a_capture/graph_capture/demo/demo_model.py --mode both

    # 单独跑 graph_capture
    python torch2c/a_capture/graph_capture/demo/demo_model.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class DemoEncoder(nn.Module):
    """2 层 Transformer Encoder，用于演示图捕获。"""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4, ffn_dim: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Layer 1
        self.ln1_1 = nn.LayerNorm(hidden_size)
        self.wq1 = nn.Linear(hidden_size, hidden_size)
        self.wk1 = nn.Linear(hidden_size, hidden_size)
        self.wv1 = nn.Linear(hidden_size, hidden_size)
        self.wo1 = nn.Linear(hidden_size, hidden_size)
        self.ln1_2 = nn.LayerNorm(hidden_size)
        self.ff1_1 = nn.Linear(hidden_size, ffn_dim)
        self.ff1_2 = nn.Linear(ffn_dim, hidden_size)

        # Layer 2
        self.ln2_1 = nn.LayerNorm(hidden_size)
        self.wq2 = nn.Linear(hidden_size, hidden_size)
        self.wk2 = nn.Linear(hidden_size, hidden_size)
        self.wv2 = nn.Linear(hidden_size, hidden_size)
        self.wo2 = nn.Linear(hidden_size, hidden_size)
        self.ln2_2 = nn.LayerNorm(hidden_size)
        self.ff2_1 = nn.Linear(hidden_size, ffn_dim)
        self.ff2_2 = nn.Linear(ffn_dim, hidden_size)

    def _attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        wq: nn.Linear,
        wk: nn.Linear,
        wv: nn.Linear,
        wo: nn.Linear,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = wk(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = wv(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.bmm(
            q.reshape(B * self.num_heads, S, self.head_dim),
            k.reshape(B * self.num_heads, S, self.head_dim).transpose(-2, -1),
        ).view(B, self.num_heads, S, S)

        scores = scores * self.scale
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)

        out = torch.bmm(
            attn.view(B * self.num_heads, S, S),
            v.reshape(B * self.num_heads, S, self.head_dim),
        ).view(B, self.num_heads, S, self.head_dim)

        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        return wo(out)

    def _encoder_layer(
        self, x: torch.Tensor, mask: torch.Tensor, ln1, wq, wk, wv, wo, ln2, ff1, ff2
    ) -> torch.Tensor:
        h = ln1(x)
        h = self._attention(h, mask, wq, wk, wv, wo)
        x = x + h
        h = ln2(x)
        h = ff1(h)
        h = F.gelu(h)
        h = ff2(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._encoder_layer(
            x,
            mask,
            self.ln1_1,
            self.wq1,
            self.wk1,
            self.wv1,
            self.wo1,
            self.ln1_2,
            self.ff1_1,
            self.ff1_2,
        )
        x = self._encoder_layer(
            x,
            mask,
            self.ln2_1,
            self.wq2,
            self.wk2,
            self.wv2,
            self.wo2,
            self.ln2_2,
            self.ff2_1,
            self.ff2_2,
        )
        return x


# ── compile_model.py 需要的导出 ──
model = DemoEncoder()
dummy_input = torch.randn(1, 32, 64)
mask = torch.zeros(1, 1, 32, 32)


if __name__ == "__main__":
    import json

    from torch2c.a_capture.graph_capture import capture
    from torch2c.common.paths import DEFAULT_OUTPUT_DIR

    model.eval()
    graph = capture(model, dummy_input, mask=mask)

    out_dir = DEFAULT_OUTPUT_DIR / "graph_capture_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "captured_graph.json"
    out_path.write_text(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已保存到 {out_path}")
    print(graph.summary())
