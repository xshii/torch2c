"""单算子 C 级别集成测试 — 验证 npu_cpu_mock 各算子计算正确性。

每个测试：
1. 在 Python 侧用 numpy 计算 golden 结果
2. 生成最小 C 程序调用 npu_cpu_mock 的对应函数
3. 编译、运行，读回结果
4. 对比 max_abs_diff < tol, cosine > cos_tol
"""

from __future__ import annotations

import math
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
_NPU_MOCK_DIR = _PROJECT_ROOT / "npu_cpu_mock"
_MOCK_SOURCES = [
    "src/npu_dtype_utils.c",
    "src/npu_compute_elementwise.c",
    "src/npu_compute_matmul.c",
    "src/npu_compute_norm.c",
    "src/npu_compute_softmax.c",
    "src/npu_compute_transpose.c",
    "src/npu_dma.c",
    "src/npu_sync.c",
]

_HAS_CC = shutil.which("cc") is not None or shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not _HAS_CC, reason="C 编译器不可用")


# ---- 工具函数 ----


def _find_cc() -> str:
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return cc
    raise RuntimeError("no C compiler")


def _write_bin(path: str, arr: np.ndarray) -> None:
    arr.tofile(path)


def _read_bin(path: str, dtype: np.dtype, count: int) -> np.ndarray:
    return np.fromfile(path, dtype=dtype, count=count)


def _compile_and_run(c_code: str, workdir: str) -> str:
    """编译并运行 C 代码，返回 stdout。"""
    c_path = os.path.join(workdir, "test_op.c")
    with open(c_path, "w") as f:
        f.write(c_code)

    cc = _find_cc()
    mock_dir = str(_NPU_MOCK_DIR)
    mock_srcs = [os.path.join(mock_dir, s) for s in _MOCK_SOURCES]

    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-O2",
        "-o",
        "test_op",
        c_path,
        *mock_srcs,
        f"-I{os.path.join(mock_dir, 'include')}",
        "-lm",
    ]
    comp = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=30)
    assert comp.returncode == 0, f"编译失败:\n{comp.stderr}"

    run = subprocess.run(
        ["./test_op"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 0, f"运行失败 (rc={run.returncode}):\n{run.stderr}"
    return run.stdout


def _compare(
    actual: np.ndarray, expected: np.ndarray, atol: float = 1e-3, cos_tol: float = 0.999
) -> dict:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    max_abs = float(diff.max())
    a = actual.astype(np.float64).flatten()
    e = expected.astype(np.float64).flatten()
    dot = np.dot(a, e)
    na = np.linalg.norm(a)
    ne = np.linalg.norm(e)
    cosine = float(dot / (na * ne)) if na > 0 and ne > 0 else 0.0
    passed = max_abs <= atol and cosine >= cos_tol
    return {"passed": passed, "max_abs": max_abs, "cosine": cosine}


# ---- Cube 算子 ----


class TestMatmul:
    """npu_matmul: C = A @ B (fp16)"""

    def test_basic_matmul(self, tmp_path):
        M, K, N = 4, 8, 6
        rng = np.random.RandomState(42)
        a = rng.randn(M, K).astype(np.float16)
        b = rng.randn(K, N).astype(np.float16)
        # C mock 内部用 fp32 累加，golden 也用 fp32 算再截断
        expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)

        c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include "npu_api.h"

int main(void) {{
    unsigned short a[{M * K}], b[{K * N}], out[{M * N}];
    FILE* fa = fopen("a.bin", "rb"); fread(a, 2, {M * K}, fa); fclose(fa);
    FILE* fb = fopen("b.bin", "rb"); fread(b, 2, {K * N}, fb); fclose(fb);

    npu_matmul(a, b, out, {M}, {N}, {K}, NPU_DTYPE_FP16, NPU_FORMAT_ND);

    FILE* fo = fopen("out.bin", "wb"); fwrite(out, 2, {M * N}, fo); fclose(fo);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, M * N).reshape(M, N)
        r = _compare(actual, expected, atol=5e-3)
        assert r["passed"], f"matmul: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestMatmulBias:
    """npu_matmul_bias: C = A @ B + bias (fp16)"""

    def test_basic_matmul_bias(self, tmp_path):
        M, K, N = 4, 8, 6
        rng = np.random.RandomState(43)
        a = rng.randn(M, K).astype(np.float16)
        b = rng.randn(K, N).astype(np.float16)
        bias = rng.randn(N).astype(np.float16)
        expected = (a.astype(np.float32) @ b.astype(np.float32) + bias.astype(np.float32)).astype(
            np.float16
        )

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)
        _write_bin(os.path.join(workdir, "bias.bin"), bias)

        c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include "npu_api.h"

int main(void) {{
    unsigned short a[{M * K}], b[{K * N}], bias[{N}], out[{M * N}];
    FILE* f;
    f = fopen("a.bin", "rb"); fread(a, 2, {M * K}, f); fclose(f);
    f = fopen("b.bin", "rb"); fread(b, 2, {K * N}, f); fclose(f);
    f = fopen("bias.bin", "rb"); fread(bias, 2, {N}, f); fclose(f);

    npu_matmul_bias(a, b, bias, out, {M}, {N}, {K},
                    NPU_DTYPE_FP16, NPU_DTYPE_FP16, NPU_FORMAT_ND);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {M * N}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, M * N).reshape(M, N)
        r = _compare(actual, expected, atol=5e-3)
        assert r["passed"], f"matmul_bias: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


# ---- Vector 算子 ----


class TestAdd:
    """npu_add: C = A + B (fp16)"""

    def test_basic_add(self, tmp_path):
        n = 64
        rng = np.random.RandomState(44)
        a = rng.randn(n).astype(np.float16)
        b = rng.randn(n).astype(np.float16)
        expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short a[{n}], b[{n}], out[{n}];
    FILE* f;
    f = fopen("a.bin", "rb"); fread(a, 2, {n}, f); fclose(f);
    f = fopen("b.bin", "rb"); fread(b, 2, {n}, f); fclose(f);

    npu_add(a, b, out, {n}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {n}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"add: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestGelu:
    """npu_gelu: Y = gelu(X) (fp16)"""

    def test_basic_gelu(self, tmp_path):
        n = 64
        rng = np.random.RandomState(45)
        x = rng.randn(n).astype(np.float16)
        xf = x.astype(np.float32)
        # gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        erf_vals = np.array([math.erf(float(v) / math.sqrt(2.0)) for v in xf])
        expected = (xf * 0.5 * (1.0 + erf_vals)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short x[{n}], out[{n}];
    FILE* f;
    f = fopen("x.bin", "rb"); fread(x, 2, {n}, f); fclose(f);

    npu_gelu(x, out, {n}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {n}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"gelu: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestMul:
    """npu_mul: C = A * B (fp16)"""

    def test_basic_mul(self, tmp_path):
        n = 64
        rng = np.random.RandomState(46)
        a = rng.randn(n).astype(np.float16)
        b = rng.randn(n).astype(np.float16)
        expected = (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short a[{n}], b[{n}], out[{n}];
    FILE* f;
    f = fopen("a.bin", "rb"); fread(a, 2, {n}, f); fclose(f);
    f = fopen("b.bin", "rb"); fread(b, 2, {n}, f); fclose(f);

    npu_mul(a, b, out, {n}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {n}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"mul: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestMulScalar:
    """npu_mul_scalar: Y = X * scalar (fp16)"""

    def test_basic_mul_scalar(self, tmp_path):
        n = 64
        scalar = 0.125
        rng = np.random.RandomState(47)
        x = rng.randn(n).astype(np.float16)
        expected = (x.astype(np.float32) * scalar).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short x[{n}], out[{n}];
    FILE* f;
    f = fopen("x.bin", "rb"); fread(x, 2, {n}, f); fclose(f);

    npu_mul_scalar(x, out, {scalar}f, {n}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {n}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"mul_scalar: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


# ---- DMA + 格式转换算子 ----


class TestTranspose:
    """npu_transpose_2d: B = A^T (fp16)"""

    def test_transpose_2d(self, tmp_path):
        rows, cols = 8, 16
        rng = np.random.RandomState(48)
        a = rng.randn(rows, cols).astype(np.float16)
        expected = a.T.copy()

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short a[{rows * cols}], out[{rows * cols}];
    FILE* f;
    f = fopen("a.bin", "rb"); fread(a, 2, {rows * cols}, f); fclose(f);

    npu_transpose_2d(a, out, {rows}, {cols}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {rows * cols}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, rows * cols).reshape(
            cols, rows
        )
        r = _compare(actual, expected)
        assert r["passed"], f"transpose_2d: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestDMA:
    """npu_dma_load / npu_dma_store: HBM <-> L1 搬运"""

    def test_dma_roundtrip(self, tmp_path):
        n = 128
        rng = np.random.RandomState(49)
        data = rng.randn(n).astype(np.float16)
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "data.bin"), data)

        nbytes = n * 2
        c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_api.h"

int main(void) {{
    unsigned char hbm[{nbytes}], l1[{nbytes}], hbm2[{nbytes}];
    FILE* f;
    f = fopen("data.bin", "rb"); fread(hbm, 1, {nbytes}, f); fclose(f);
    memset(l1, 0, {nbytes});
    memset(hbm2, 0, {nbytes});

    npu_dma_load(l1, hbm, {nbytes}, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_barrier();
    npu_dma_store(hbm2, l1, {nbytes}, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_barrier();

    f = fopen("out.bin", "wb"); fwrite(hbm2, 1, {nbytes}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        assert np.array_equal(actual, data), "DMA roundtrip 数据不一致"


class TestReshape:
    """npu_reshape: 内存拷贝（reshape 不改变数据布局）"""

    def test_reshape(self, tmp_path):
        n = 64
        rng = np.random.RandomState(50)
        x = rng.randn(n).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short x[{n}], out[{n}];
    FILE* f;
    f = fopen("x.bin", "rb"); fread(x, 2, {n}, f); fclose(f);

    npu_reshape(x, out, {n * 2}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {n}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        assert np.array_equal(actual, x), "reshape 数据不一致"


# ---- 分解算子 (Decomposition) ----


class TestSoftmax:
    """npu_softmax_part1 + npu_softmax_part2: 分步 softmax"""

    def test_softmax_decomposed(self, tmp_path):
        seq, dim = 4, 8
        rng = np.random.RandomState(51)
        x = rng.randn(seq, dim).astype(np.float16)
        xf = x.astype(np.float32)
        exp_x = np.exp(xf - xf.max(axis=-1, keepdims=True))
        expected = (exp_x / exp_x.sum(axis=-1, keepdims=True)).astype(np.float16)
        count = seq * dim

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short x[{count}], inter[{count}], out[{count}];
    FILE* f;
    f = fopen("x.bin", "rb"); fread(x, 2, {count}, f); fclose(f);

    npu_softmax_part1(x, inter, {dim}, {count}, NPU_DTYPE_FP16);
    npu_softmax_part2(inter, out, {count * 2}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {count}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, count).reshape(seq, dim)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"softmax: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestLayerNorm:
    """npu_layernorm_part1 + npu_layernorm_part2: 分步 layernorm"""

    def test_layernorm_decomposed(self, tmp_path):
        seq, hidden = 4, 8
        rng = np.random.RandomState(52)
        x = rng.randn(seq, hidden).astype(np.float16)
        gamma = np.ones(hidden, dtype=np.float16)
        beta = np.zeros(hidden, dtype=np.float16)
        eps = 1e-5

        xf = x.astype(np.float32)
        mean = xf.mean(axis=-1, keepdims=True)
        var = xf.var(axis=-1, keepdims=True)
        norm = (xf - mean) / np.sqrt(var + eps)
        expected = (norm * gamma.astype(np.float32) + beta.astype(np.float32)).astype(np.float16)
        count = seq * hidden

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)
        _write_bin(os.path.join(workdir, "gamma.bin"), gamma)
        _write_bin(os.path.join(workdir, "beta.bin"), beta)

        # layernorm_part2 的 hidden 参数传 total elem count（mock 内部 * dtype_size）
        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short x[{count}], gamma[{hidden}], beta[{hidden}];
    unsigned short inter[{count}], out[{count}];
    FILE* f;
    f = fopen("x.bin", "rb"); fread(x, 2, {count}, f); fclose(f);
    f = fopen("gamma.bin", "rb"); fread(gamma, 2, {hidden}, f); fclose(f);
    f = fopen("beta.bin", "rb"); fread(beta, 2, {hidden}, f); fclose(f);

    npu_layernorm_part1(x, gamma, beta, inter, {hidden}, {seq},
                        {eps}f, NPU_DTYPE_FP16);
    npu_layernorm_part2(inter, x, out, {count * 2}, NPU_DTYPE_FP16);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {count}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, count).reshape(seq, hidden)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"layernorm: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


# ---- 融合算子 (Fusion / Pipeline 组合) ----


class TestMatmulAddFusion:
    """组合测试: npu_matmul_bias（模拟 addmm 融合为 matmul + bias）"""

    def test_matmul_then_add(self, tmp_path):
        M, K, N = 4, 8, 6
        rng = np.random.RandomState(53)
        a = rng.randn(M, K).astype(np.float16)
        b = rng.randn(K, N).astype(np.float16)
        bias = rng.randn(N).astype(np.float16)

        expected = (a.astype(np.float32) @ b.astype(np.float32) + bias.astype(np.float32)).astype(
            np.float16
        )

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)
        _write_bin(os.path.join(workdir, "bias.bin"), bias)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short a[{M * K}], b[{K * N}], bias[{N}], out[{M * N}];
    FILE* f;
    f = fopen("a.bin", "rb"); fread(a, 2, {M * K}, f); fclose(f);
    f = fopen("b.bin", "rb"); fread(b, 2, {K * N}, f); fclose(f);
    f = fopen("bias.bin", "rb"); fread(bias, 2, {N}, f); fclose(f);

    npu_matmul_bias(a, b, bias, out, {M}, {N}, {K},
                    NPU_DTYPE_FP16, NPU_DTYPE_FP16, NPU_FORMAT_ND);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {M * N}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, M * N).reshape(M, N)
        r = _compare(actual, expected, atol=5e-3)
        assert r["passed"], f"matmul_bias: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestAttentionBlock:
    """组合测试: Q@K^T -> softmax -> @V（单头 attention 核心路径）"""

    def test_single_head_attention(self, tmp_path):
        seq, d = 4, 8
        rng = np.random.RandomState(54)
        q = rng.randn(seq, d).astype(np.float16)
        k = rng.randn(seq, d).astype(np.float16)
        v = rng.randn(seq, d).astype(np.float16)

        qf, kf, vf = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
        scores = (qf @ kf.T).astype(np.float16).astype(np.float32)
        exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = (exp_s / exp_s.sum(axis=-1, keepdims=True)).astype(np.float16)
        expected = (attn.astype(np.float32) @ vf).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "q.bin"), q)
        _write_bin(os.path.join(workdir, "k.bin"), k)
        _write_bin(os.path.join(workdir, "v.bin"), v)

        c_code = f"""
#include <stdio.h>
#include "npu_api.h"

int main(void) {{
    unsigned short q[{seq * d}], k[{seq * d}], v[{seq * d}];
    unsigned short kt[{seq * d}], scores[{seq * seq}], sm_inter[{seq * seq}];
    unsigned short attn[{seq * seq}], out[{seq * d}];
    FILE* f;
    f = fopen("q.bin", "rb"); fread(q, 2, {seq * d}, f); fclose(f);
    f = fopen("k.bin", "rb"); fread(k, 2, {seq * d}, f); fclose(f);
    f = fopen("v.bin", "rb"); fread(v, 2, {seq * d}, f); fclose(f);

    /* K^T */
    npu_transpose_2d(k, kt, {seq}, {d}, NPU_DTYPE_FP16);
    /* scores = Q @ K^T */
    npu_matmul(q, kt, scores, {seq}, {seq}, {d}, NPU_DTYPE_FP16, NPU_FORMAT_ND);
    /* softmax */
    npu_softmax_part1(scores, sm_inter, {seq}, {seq * seq}, NPU_DTYPE_FP16);
    npu_softmax_part2(sm_inter, attn, {seq * seq * 2}, NPU_DTYPE_FP16);
    /* out = attn @ V */
    npu_matmul(attn, v, out, {seq}, {d}, {seq}, NPU_DTYPE_FP16, NPU_FORMAT_ND);

    f = fopen("out.bin", "wb"); fwrite(out, 2, {seq * d}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, seq * d).reshape(seq, d)
        r = _compare(actual, expected, atol=5e-3, cos_tol=0.995)
        assert r["passed"], f"attention: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"
