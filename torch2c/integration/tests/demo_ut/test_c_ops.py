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

from torch2c.common import NPU_CPU_MOCK_DIR

_NPU_MOCK_DIR = NPU_CPU_MOCK_DIR
_MOCK_SOURCES = [
    "src/npu_dtype_utils.c",
    "src/npu_debug.c",
    "src/npu_compute_elementwise.c",
    "src/npu_compute_matmul.c",
    "src/npu_compute_norm.c",
    "src/npu_compute_rmsnorm.c",
    "src/npu_compute_softmax.c",
    "src/npu_compute_transpose.c",
    "src/npu_dma.c",
]

_HAS_CC = shutil.which("cc") is not None or shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not _HAS_CC, reason="C 编译器不可用")

# Common C preamble: L1 buffer + tensor helper
_C_PREAMBLE = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_api.h"

#define TID0 ((TidInfo){0, 0, 0, 0, 0})

#define L1_SIZE 65536
static unsigned char l1[L1_SIZE];

static void l1_init(void) {
    memset(l1, 0, L1_SIZE);
}

static npu_tensor_t T(void* base, uint32_t byte_offset, npu_dtype_t dt) {
    npu_tensor_t t = {base + byte_offset, dt, NPU_FORMAT_ND};
    return t;
}

static void load_file(const char* path, void* dst, size_t bytes) {
    FILE* f = fopen(path, "rb");
    fread(dst, 1, bytes, f);
    fclose(f);
}

static void save_file(const char* path, const void* src, size_t bytes) {
    FILE* f = fopen(path, "wb");
    fwrite(src, 1, bytes, f);
    fclose(f);
}
"""


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
    """cube_matmul: C = A @ B (fp16)"""

    def test_basic_matmul(self, tmp_path):
        M, K, N = 4, 8, 6
        rng = np.random.RandomState(42)
        a = rng.randn(M, K).astype(np.float16)
        b = rng.randn(K, N).astype(np.float16)
        expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)

        # offsets into L1
        off_a, off_b, off_out = 0, 1024, 2048
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("a.bin", l1 + {off_a}, {M * K * 2});
    load_file("b.bin", l1 + {off_b}, {K * N * 2});

    cube_matmul(TID0, T(l1, {off_a}, NPU_DTYPE_FP16), T(l1, {off_b}, NPU_DTYPE_FP16),
                T(l1, {off_out}, NPU_DTYPE_FP16), 1, {M}, {N}, {K}, 0, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {M * N * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, M * N).reshape(M, N)
        r = _compare(actual, expected, atol=5e-3)
        assert r["passed"], f"matmul: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestMatmulBias:
    """cube_matmul_bias: C = A @ B + bias (fp16)"""

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

        off_a, off_b, off_bias, off_out = 0, 1024, 2048, 3072
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("a.bin", l1 + {off_a}, {M * K * 2});
    load_file("b.bin", l1 + {off_b}, {K * N * 2});
    load_file("bias.bin", l1 + {off_bias}, {N * 2});

    cube_matmul_bias(TID0, T(l1, {off_a}, NPU_DTYPE_FP16), T(l1, {off_b}, NPU_DTYPE_FP16),
                     T(l1, {off_bias}, NPU_DTYPE_FP16), T(l1, {off_out}, NPU_DTYPE_FP16),
                     1, {M}, {N}, {K}, 0, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {M * N * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, M * N).reshape(M, N)
        r = _compare(actual, expected, atol=5e-3)
        assert r["passed"], f"matmul_bias: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


# ---- Vector 算子 ----


class TestAdd:
    """vector_add: C = A + B (fp16)"""

    def test_basic_add(self, tmp_path):
        n = 64
        rng = np.random.RandomState(44)
        a = rng.randn(n).astype(np.float16)
        b = rng.randn(n).astype(np.float16)
        expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)

        off_a, off_b, off_out = 0, 1024, 2048
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("a.bin", l1 + {off_a}, {n * 2});
    load_file("b.bin", l1 + {off_b}, {n * 2});

    vector_add(TID0, T(l1, {off_a}, NPU_DTYPE_FP16), T(l1, {off_b}, NPU_DTYPE_FP16),
               T(l1, {off_out}, NPU_DTYPE_FP16), {n}, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {n * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"add: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestGelu:
    """vector_gelu: Y = gelu(X) (fp16)"""

    def test_basic_gelu(self, tmp_path):
        n = 64
        rng = np.random.RandomState(45)
        x = rng.randn(n).astype(np.float16)
        xf = x.astype(np.float32)
        erf_vals = np.array([math.erf(float(v) / math.sqrt(2.0)) for v in xf])
        expected = (xf * 0.5 * (1.0 + erf_vals)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)

        off_in, off_out = 0, 1024
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("x.bin", l1 + {off_in}, {n * 2});

    vector_gelu(TID0, T(l1, {off_in}, NPU_DTYPE_FP16), T(l1, {off_out}, NPU_DTYPE_FP16),
                {n}, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {n * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"gelu: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestMul:
    """vector_mul: C = A * B (fp16)"""

    def test_basic_mul(self, tmp_path):
        n = 64
        rng = np.random.RandomState(46)
        a = rng.randn(n).astype(np.float16)
        b = rng.randn(n).astype(np.float16)
        expected = (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)
        _write_bin(os.path.join(workdir, "b.bin"), b)

        off_a, off_b, off_out = 0, 1024, 2048
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("a.bin", l1 + {off_a}, {n * 2});
    load_file("b.bin", l1 + {off_b}, {n * 2});

    vector_mul(TID0, T(l1, {off_a}, NPU_DTYPE_FP16), T(l1, {off_b}, NPU_DTYPE_FP16),
               T(l1, {off_out}, NPU_DTYPE_FP16), {n}, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {n * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"mul: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestMulScalar:
    """vector_mul_scalar: Y = X * scalar (fp16)"""

    def test_basic_mul_scalar(self, tmp_path):
        n = 64
        scalar = 0.125
        rng = np.random.RandomState(47)
        x = rng.randn(n).astype(np.float16)
        expected = (x.astype(np.float32) * scalar).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)

        off_in, off_out = 0, 1024
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("x.bin", l1 + {off_in}, {n * 2});

    vector_mul_scalar(TID0, T(l1, {off_in}, NPU_DTYPE_FP16), T(l1, {off_out}, NPU_DTYPE_FP16),
                      {scalar}f, {n}, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {n * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"mul_scalar: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


# ---- DMA + 格式转换算子 ----


class TestTranspose:
    """vector_transpose_2d: B = A^T (fp16)"""

    def test_transpose_2d(self, tmp_path):
        rows, cols = 8, 16
        rng = np.random.RandomState(48)
        a = rng.randn(rows, cols).astype(np.float16)
        expected = a.T.copy()

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "a.bin"), a)

        off_in, off_out = 0, 4096
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("a.bin", l1 + {off_in}, {rows * cols * 2});

    vector_transpose_2d(TID0, T(l1, {off_in}, NPU_DTYPE_FP16), T(l1, {off_out}, NPU_DTYPE_FP16),
                        {rows}, {cols}, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {rows * cols * 2});
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
    """dma_move: HBM <-> L1 搬运"""

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
#define TID0 ((TidInfo){{0, 0, 0, 0, 0}})

int main(void) {{
    unsigned char hbm[{nbytes}], l1[{nbytes}], hbm2[{nbytes}];
    FILE* f;
    f = fopen("data.bin", "rb"); fread(hbm, 1, {nbytes}, f); fclose(f);
    memset(l1, 0, {nbytes});
    memset(hbm2, 0, {nbytes});

    dma_move(TID0, (npu_tensor_t){{l1, NPU_DTYPE_FP16, NPU_FORMAT_ND}},
                   (npu_tensor_t){{hbm, NPU_DTYPE_FP16, NPU_FORMAT_ND}}, {nbytes});
    dma_move(TID0, (npu_tensor_t){{hbm2, NPU_DTYPE_FP16, NPU_FORMAT_ND}},
                   (npu_tensor_t){{l1, NPU_DTYPE_FP16, NPU_FORMAT_ND}}, {nbytes});

    f = fopen("out.bin", "wb"); fwrite(hbm2, 1, {nbytes}, f); fclose(f);
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        assert np.array_equal(actual, data), "DMA roundtrip 数据不一致"


class TestReshape:
    """idma_reshape: 内存拷贝（reshape 不改变数据布局）"""

    def test_reshape(self, tmp_path):
        n = 64
        rng = np.random.RandomState(50)
        x = rng.randn(n).astype(np.float16)

        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)

        off_in, off_out = 0, 1024
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("x.bin", l1 + {off_in}, {n * 2});

    idma_reshape(TID0, T(l1, {off_in}, NPU_DTYPE_FP16), T(l1, {off_out}, NPU_DTYPE_FP16),
                 {n * 2});

    save_file("out.bin", l1 + {off_out}, {n * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, n)
        assert np.array_equal(actual, x), "reshape 数据不一致"


# ---- 分解算子 (Decomposition) ----


class TestSoftmax:
    """vector_softmax_part1 + vector_softmax_part2: 分步 softmax"""

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

        off_in, off_inter, off_out = 0, 1024, 2048
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("x.bin", l1 + {off_in}, {count * 2});

    vector_softmax_part1(TID0, T(l1, {off_in}, NPU_DTYPE_FP16), T(l1, {off_inter}, NPU_DTYPE_FP16),
                         {dim}, {count}, NPU_DTYPE_FP16);
    vector_softmax_part2(TID0, T(l1, {off_inter}, NPU_DTYPE_FP16), T(l1, {off_out}, NPU_DTYPE_FP16),
                         {count * 2}, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {count * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, count).reshape(seq, dim)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"softmax: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestLayerNorm:
    """vector_layernorm_part1 + vector_layernorm_part2: 分步 layernorm"""

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

        off_x, off_gamma, off_beta = 0, 1024, 2048
        off_inter, off_out = 3072, 4096
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("x.bin", l1 + {off_x}, {count * 2});
    load_file("gamma.bin", l1 + {off_gamma}, {hidden * 2});
    load_file("beta.bin", l1 + {off_beta}, {hidden * 2});

    vector_layernorm_part1(TID0, T(l1, {off_x}, NPU_DTYPE_FP16), T(l1, {off_gamma}, NPU_DTYPE_FP16),
                           T(l1, {off_beta}, NPU_DTYPE_FP16), T(l1, {off_inter}, NPU_DTYPE_FP16),
                           {hidden}, {seq}, {eps}f, NPU_DTYPE_FP16);
    vector_layernorm_part2(TID0, T(l1, {off_inter}, NPU_DTYPE_FP16), T(l1, {off_x}, NPU_DTYPE_FP16),
                           T(l1, {off_out}, NPU_DTYPE_FP16), {count * 2}, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {count * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, count).reshape(seq, hidden)
        r = _compare(actual, expected, atol=3e-3)
        assert r["passed"], f"layernorm: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


# ---- 融合算子 (Fusion / Pipeline 组合) ----


class TestMatmulAddFusion:
    """组合测试: cube_matmul_bias（模拟 addmm 融合为 matmul + bias）"""

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

        off_a, off_b, off_bias, off_out = 0, 1024, 2048, 3072
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("a.bin", l1 + {off_a}, {M * K * 2});
    load_file("b.bin", l1 + {off_b}, {K * N * 2});
    load_file("bias.bin", l1 + {off_bias}, {N * 2});

    cube_matmul_bias(TID0, T(l1, {off_a}, NPU_DTYPE_FP16), T(l1, {off_b}, NPU_DTYPE_FP16),
                     T(l1, {off_bias}, NPU_DTYPE_FP16), T(l1, {off_out}, NPU_DTYPE_FP16),
                     1, {M}, {N}, {K}, 0, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {M * N * 2});
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

        # L1 layout: q(0), k(1024), v(2048), kt(3072), scores(4096), sm(5120), attn(6144), out(7168)
        off_q, off_k, off_v = 0, 1024, 2048
        off_kt, off_scores, off_sm, off_attn, off_out = 3072, 4096, 5120, 6144, 7168
        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("q.bin", l1 + {off_q}, {seq * d * 2});
    load_file("k.bin", l1 + {off_k}, {seq * d * 2});
    load_file("v.bin", l1 + {off_v}, {seq * d * 2});

    /* K^T */
    vector_transpose_2d(TID0, T(l1, {off_k}, NPU_DTYPE_FP16), T(l1, {off_kt}, NPU_DTYPE_FP16),
                        {seq}, {d}, NPU_DTYPE_FP16);
    /* scores = Q @ K^T */
    cube_matmul(TID0, T(l1, {off_q}, NPU_DTYPE_FP16), T(l1, {off_kt}, NPU_DTYPE_FP16),
                T(l1, {off_scores}, NPU_DTYPE_FP16), 1, {seq}, {seq}, {d}, 0, NPU_DTYPE_FP16);
    /* softmax */
    vector_softmax_part1(TID0, T(l1, {off_scores}, NPU_DTYPE_FP16), T(l1, {off_sm}, NPU_DTYPE_FP16),
                         {seq}, {seq * seq}, NPU_DTYPE_FP16);
    vector_softmax_part2(TID0, T(l1, {off_sm}, NPU_DTYPE_FP16), T(l1, {off_attn}, NPU_DTYPE_FP16),
                         {seq * seq * 2}, NPU_DTYPE_FP16);
    /* out = attn @ V */
    cube_matmul(TID0, T(l1, {off_attn}, NPU_DTYPE_FP16), T(l1, {off_v}, NPU_DTYPE_FP16),
                T(l1, {off_out}, NPU_DTYPE_FP16), 1, {seq}, {d}, {seq}, 0, NPU_DTYPE_FP16);

    save_file("out.bin", l1 + {off_out}, {seq * d * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), np.float16, seq * d).reshape(seq, d)
        r = _compare(actual, expected, atol=5e-3, cos_tol=0.995)
        assert r["passed"], f"attention: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"


class TestMultiHeadAttention:
    """组合测试: 多头注意力（混合精度 fp16 存储 + fp32 计算, vector_transpose 做 head reshape）"""

    def test_multi_head_attention(self, tmp_path):
        num_heads, seq, d_model = 2, 4, 8
        head_dim = d_model // num_heads  # 4
        rng = np.random.RandomState(55)

        x = rng.randn(seq, d_model).astype(np.float16)
        wq = rng.randn(d_model, d_model).astype(np.float16)
        wk = rng.randn(d_model, d_model).astype(np.float16)
        wv = rng.randn(d_model, d_model).astype(np.float16)
        wo = rng.randn(d_model, d_model).astype(np.float16)

        # --- numpy golden ---
        xf = x.astype(np.float32)
        wqf, wkf, wvf, wof = (
            wq.astype(np.float32),
            wk.astype(np.float32),
            wv.astype(np.float32),
            wo.astype(np.float32),
        )

        # linear projections (truncate to fp16 like the C path)
        q_proj = (xf @ wqf).astype(np.float16).astype(np.float32)  # (seq, d_model)
        k_proj = (xf @ wkf).astype(np.float16).astype(np.float32)
        v_proj = (xf @ wvf).astype(np.float16).astype(np.float32)

        # reshape to (num_heads, seq, head_dim) — contiguous in memory
        q_heads = q_proj.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
        k_heads = k_proj.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
        v_heads = v_proj.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)

        # per-head attention
        head_outs = []
        for h in range(num_heads):
            qh = q_heads[h].astype(np.float16).astype(np.float32)
            kh = k_heads[h].astype(np.float16).astype(np.float32)
            vh = v_heads[h].astype(np.float16).astype(np.float32)
            scores = (qh @ kh.T).astype(np.float16).astype(np.float32)
            scores = scores * (1.0 / math.sqrt(head_dim))
            scores = scores.astype(np.float16).astype(np.float32)
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = (exp_s / exp_s.sum(axis=-1, keepdims=True)).astype(np.float16)
            out_h = (attn.astype(np.float32) @ vh).astype(np.float16)
            head_outs.append(out_h)

        # concat heads -> (seq, d_model)
        concat = np.concatenate(head_outs, axis=-1).astype(np.float16)
        expected = (concat.astype(np.float32) @ wof).astype(np.float16)

        # --- write inputs ---
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "x.bin"), x)
        _write_bin(os.path.join(workdir, "wq.bin"), wq)
        _write_bin(os.path.join(workdir, "wk.bin"), wk)
        _write_bin(os.path.join(workdir, "wv.bin"), wv)
        _write_bin(os.path.join(workdir, "wo.bin"), wo)

        # L1 memory layout (byte offsets, each slot = 1024 bytes)
        off_x = 0           # x:         seq*d_model*2 = 64
        off_wq = 1024       # wq:        d_model*d_model*2 = 128
        off_wk = 2048       # wk
        off_wv = 3072       # wv
        off_wo = 4096       # wo
        off_q = 5120        # q_proj:    seq*d_model*2 = 64
        off_k = 6144        # k_proj
        off_v = 7168        # v_proj
        # per-head working buffers (need head0 and head1 data interleaved)
        off_qh = 8192       # q heads reshaped: num_heads*seq*head_dim*2 = 64
        off_kh = 9216       # k heads reshaped
        off_vh = 10240      # v heads reshaped
        off_kht = 11264     # k^T per head (batched)
        off_scores = 12288  # scores: num_heads*seq*seq*2 = 64
        off_scaled = 13312  # scaled scores
        off_sm = 14336      # softmax intermediate
        off_attn = 15360    # softmax output
        off_head_out = 16384  # per-head output: num_heads*seq*head_dim*2 = 64
        off_concat = 17408  # concat: seq*d_model*2 = 64
        off_out = 18432     # final output

        scale = 1.0 / math.sqrt(head_dim)

        c_code = _C_PREAMBLE + f"""
int main(void) {{
    l1_init();
    load_file("x.bin",  l1 + {off_x},  {seq * d_model * 2});
    load_file("wq.bin", l1 + {off_wq}, {d_model * d_model * 2});
    load_file("wk.bin", l1 + {off_wk}, {d_model * d_model * 2});
    load_file("wv.bin", l1 + {off_wv}, {d_model * d_model * 2});
    load_file("wo.bin", l1 + {off_wo}, {d_model * d_model * 2});

    /* Q = x @ Wq  (mixed precision: storage fp16, compute fp32) */
    cube_matmul(TID0, T(l1, {off_x}, NPU_DTYPE_FP16), T(l1, {off_wq}, NPU_DTYPE_FP16),
                T(l1, {off_q}, NPU_DTYPE_FP16), 1, {seq}, {d_model}, {d_model}, 0, NPU_DTYPE_FP32);
    /* K = x @ Wk */
    cube_matmul(TID0, T(l1, {off_x}, NPU_DTYPE_FP16), T(l1, {off_wk}, NPU_DTYPE_FP16),
                T(l1, {off_k}, NPU_DTYPE_FP16), 1, {seq}, {d_model}, {d_model}, 0, NPU_DTYPE_FP32);
    /* V = x @ Wv */
    cube_matmul(TID0, T(l1, {off_x}, NPU_DTYPE_FP16), T(l1, {off_wv}, NPU_DTYPE_FP16),
                T(l1, {off_v}, NPU_DTYPE_FP16), 1, {seq}, {d_model}, {d_model}, 0, NPU_DTYPE_FP32);

    /* deinterleave Q/K/V: (seq, num_heads, head_dim) -> transpose(0,1) -> (num_heads, seq, head_dim)
       uses vector_transpose instead of manual C loops — mirrors DMA随路格式转换 */
    {{
        const int deinterleave_dims[] = {{{seq}, {num_heads}, {head_dim}}};
        vector_transpose(TID0, T(l1, {off_q}, NPU_DTYPE_FP16), T(l1, {off_qh}, NPU_DTYPE_FP16),
                         3, deinterleave_dims, 0, 1, NPU_DTYPE_FP32);
        vector_transpose(TID0, T(l1, {off_k}, NPU_DTYPE_FP16), T(l1, {off_kh}, NPU_DTYPE_FP16),
                         3, deinterleave_dims, 0, 1, NPU_DTYPE_FP32);
        vector_transpose(TID0, T(l1, {off_v}, NPU_DTYPE_FP16), T(l1, {off_vh}, NPU_DTYPE_FP16),
                         3, deinterleave_dims, 0, 1, NPU_DTYPE_FP32);
    }}

    /* transpose K per head: (num_heads, seq, head_dim) -> (num_heads, head_dim, seq) */
    for (int h = 0; h < {num_heads}; h++) {{
        int kh_off = {off_kh} + h * {seq * head_dim * 2};
        int kht_off = {off_kht} + h * {seq * head_dim * 2};
        vector_transpose_2d(TID0, T(l1, kh_off, NPU_DTYPE_FP16), T(l1, kht_off, NPU_DTYPE_FP16),
                            {seq}, {head_dim}, NPU_DTYPE_FP32);
    }}

    /* batched scores = Q_heads @ K_heads^T, using loop={num_heads} */
    cube_matmul(TID0, T(l1, {off_qh}, NPU_DTYPE_FP16), T(l1, {off_kht}, NPU_DTYPE_FP16),
                T(l1, {off_scores}, NPU_DTYPE_FP16), {num_heads}, {seq}, {seq}, {head_dim}, 0, NPU_DTYPE_FP32);

    /* scale scores by 1/sqrt(head_dim) */
    vector_mul_scalar(TID0, T(l1, {off_scores}, NPU_DTYPE_FP16), T(l1, {off_scaled}, NPU_DTYPE_FP16),
                      {scale}f, {num_heads * seq * seq}, NPU_DTYPE_FP32);

    /* softmax over last dim (seq) for each row across all heads */
    vector_softmax_part1(TID0, T(l1, {off_scaled}, NPU_DTYPE_FP16), T(l1, {off_sm}, NPU_DTYPE_FP16),
                         {seq}, {num_heads * seq * seq}, NPU_DTYPE_FP32);
    vector_softmax_part2(TID0, T(l1, {off_sm}, NPU_DTYPE_FP16), T(l1, {off_attn}, NPU_DTYPE_FP16),
                         {num_heads * seq * seq * 2}, NPU_DTYPE_FP32);

    /* batched out = attn @ V_heads, loop={num_heads} */
    cube_matmul(TID0, T(l1, {off_attn}, NPU_DTYPE_FP16), T(l1, {off_vh}, NPU_DTYPE_FP16),
                T(l1, {off_head_out}, NPU_DTYPE_FP16), {num_heads}, {seq}, {head_dim}, {seq}, 0, NPU_DTYPE_FP32);

    /* reinterleave heads: (num_heads, seq, head_dim) -> transpose(0,1) -> (seq, num_heads, head_dim) = (seq, d_model) */
    {{
        const int reinterleave_dims[] = {{{num_heads}, {seq}, {head_dim}}};
        vector_transpose(TID0, T(l1, {off_head_out}, NPU_DTYPE_FP16), T(l1, {off_concat}, NPU_DTYPE_FP16),
                         3, reinterleave_dims, 0, 1, NPU_DTYPE_FP32);
    }}

    /* output projection: out = concat @ Wo */
    cube_matmul(TID0, T(l1, {off_concat}, NPU_DTYPE_FP16), T(l1, {off_wo}, NPU_DTYPE_FP16),
                T(l1, {off_out}, NPU_DTYPE_FP16), 1, {seq}, {d_model}, {d_model}, 0, NPU_DTYPE_FP32);

    save_file("out.bin", l1 + {off_out}, {seq * d_model * 2});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(
            os.path.join(workdir, "out.bin"), np.float16, seq * d_model
        ).reshape(seq, d_model)
        r = _compare(actual, expected, atol=3e-2, cos_tol=0.999)
        assert r["passed"], (
            f"multi_head_attention: max_abs={r['max_abs']:.6f}, cosine={r['cosine']:.6f}"
        )
