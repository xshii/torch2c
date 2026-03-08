"""DMA / iDMA 算子参数化 C 级别集成测试。

测试因子:
- op: dma_move, dma_reformat, idma_reshape, idma_move, idma_broadcast
- src_dtype: fp16, fp32
- dst_dtype: fp16, fp32 (同 dtype + 混合精度)
- count: 32, 256, 4096 (小/中/大)
- 验证: bitwise exact (同dtype), max_abs_diff < tol (跨dtype)
"""

from __future__ import annotations

import os
import shutil

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

# ---- dtype helpers ----

_DTYPE_MAP = {
    "fp16": ("NPU_DTYPE_FP16", np.float16, 2),
    "fp32": ("NPU_DTYPE_FP32", np.float32, 4),
}


def _c_enum(dtype_str: str) -> str:
    return _DTYPE_MAP[dtype_str][0]


def _np_dtype(dtype_str: str) -> np.dtype:
    return _DTYPE_MAP[dtype_str][1]


def _elem_size(dtype_str: str) -> int:
    return _DTYPE_MAP[dtype_str][2]


# ---- infra (shared with test_c_ops) ----

import subprocess  # noqa: E402


def _find_cc() -> str:
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return cc
    raise RuntimeError("no C compiler")


def _write_bin(path: str, arr: np.ndarray) -> None:
    arr.tofile(path)


def _read_bin(path: str, dtype, count: int) -> np.ndarray:
    return np.fromfile(path, dtype=dtype, count=count)


def _compile_and_run(c_code: str, workdir: str) -> str:
    c_path = os.path.join(workdir, "test_op.c")
    with open(c_path, "w") as f:
        f.write(c_code)

    cc = _find_cc()
    mock_dir = str(_NPU_MOCK_DIR)
    mock_srcs = [os.path.join(mock_dir, s) for s in _MOCK_SOURCES]

    cmd = [
        cc, "-std=c99", "-Wall", "-O2", "-o", "test_op",
        c_path, *mock_srcs,
        f"-I{os.path.join(mock_dir, 'include')}", "-lm",
    ]
    comp = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=30)
    assert comp.returncode == 0, f"编译失败:\n{comp.stderr}"

    run = subprocess.run(
        ["./test_op"], cwd=workdir, capture_output=True, text=True, timeout=10,
    )
    assert run.returncode == 0, f"运行失败 (rc={run.returncode}):\n{run.stderr}"
    return run.stdout


def _compare(actual: np.ndarray, expected: np.ndarray, atol: float, cos_tol: float = 0.999) -> dict:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_abs = float(diff.max())
    a = actual.astype(np.float64).flatten()
    e = expected.astype(np.float64).flatten()
    na, ne = np.linalg.norm(a), np.linalg.norm(e)
    cosine = float(np.dot(a, e) / (na * ne)) if na > 0 and ne > 0 else 0.0
    return {"passed": max_abs <= atol and cosine >= cos_tol, "max_abs": max_abs, "cosine": cosine}


# FP32→FP16 转换精度: FP16 mantissa 10bit, 对 magnitude ~2 的值最大误差 ~0.002
_DTYPE_CONV_ATOL = 3e-3


# ---- C preamble ----

_C_PREAMBLE = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_api.h"

#define TID0 ((TidInfo){0, 0, 0, 0, 0})
#define BUF_SIZE (1 << 20)

static unsigned char hbm[BUF_SIZE];
static unsigned char l1[BUF_SIZE];

static npu_tensor_t T(void* base, uint32_t off, npu_dtype_t dt) {
    npu_tensor_t t = {(unsigned char*)base + off, dt, NPU_FORMAT_ND};
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


# ============================================================
# dma_move: HBM <-> L1, dst/src 都带 dtype
# ============================================================

_DMA_MOVE_PARAMS = [
    # (src_dtype, dst_dtype, count, description)
    ("fp16", "fp16", 32,   "fp16_same_small"),
    ("fp16", "fp16", 4096, "fp16_same_large"),
    ("fp32", "fp32", 32,   "fp32_same_small"),
    ("fp32", "fp32", 4096, "fp32_same_large"),
    ("fp32", "fp16", 256,  "fp32_to_fp16"),
    ("fp16", "fp32", 256,  "fp16_to_fp32"),
    ("fp32", "fp16", 4096, "fp32_to_fp16_large"),
    ("fp16", "fp32", 4096, "fp16_to_fp32_large"),
]


class TestDmaMove:
    """dma_move: HBM→L1→HBM roundtrip，含 dtype 转换。"""

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype,count,desc",
        _DMA_MOVE_PARAMS,
        ids=[p[3] for p in _DMA_MOVE_PARAMS],
    )
    def test_dma_move(self, tmp_path, src_dtype, dst_dtype, count, desc):
        rng = np.random.RandomState(42)
        data = rng.randn(count).astype(_np_dtype(src_dtype))
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "data.bin"), data)

        src_bytes = count * _elem_size(src_dtype)
        dst_bytes = count * _elem_size(dst_dtype)
        # size_bytes = 按 src 大小传入（dma_move 约定）
        size_bytes = src_bytes

        c_code = _C_PREAMBLE + f"""
int main(void) {{
    memset(hbm, 0, BUF_SIZE);
    memset(l1, 0, BUF_SIZE);
    load_file("data.bin", hbm, {src_bytes});

    /* HBM(src) -> L1(dst) */
    dma_move(TID0, T(l1, 0, {_c_enum(dst_dtype)}),
                   T(hbm, 0, {_c_enum(src_dtype)}), {size_bytes});

    save_file("out.bin", l1, {dst_bytes});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), _np_dtype(dst_dtype), count)

        if src_dtype == dst_dtype:
            assert np.array_equal(actual, data), f"dma_move {desc}: bitwise 不一致"
        else:
            expected = data.astype(np.float64).astype(_np_dtype(dst_dtype))
            r = _compare(actual, expected, atol=_DTYPE_CONV_ATOL)
            assert r["passed"], f"dma_move {desc}: max_abs={r['max_abs']:.6f} cos={r['cosine']:.6f}"


# ============================================================
# dma_reformat: 格式/dtype 转换 (HBM <-> L1)
# ============================================================

_DMA_REFORMAT_PARAMS = [
    ("fp16", "fp16", 32,   "fp16_same_small"),
    ("fp32", "fp32", 256,  "fp32_same_medium"),
    ("fp32", "fp16", 256,  "fp32_to_fp16"),
    ("fp16", "fp32", 256,  "fp16_to_fp32"),
    ("fp32", "fp16", 4096, "fp32_to_fp16_large"),
]


class TestDmaReformat:
    """dma_reformat: 搬运并做随路 dtype 转换。"""

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype,count,desc",
        _DMA_REFORMAT_PARAMS,
        ids=[p[3] for p in _DMA_REFORMAT_PARAMS],
    )
    def test_dma_reformat(self, tmp_path, src_dtype, dst_dtype, count, desc):
        rng = np.random.RandomState(43)
        data = rng.randn(count).astype(_np_dtype(src_dtype))
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "data.bin"), data)

        src_bytes = count * _elem_size(src_dtype)
        dst_bytes = count * _elem_size(dst_dtype)
        # dma_reformat size_bytes = 按 src 大小
        size_bytes = src_bytes

        c_code = _C_PREAMBLE + f"""
int main(void) {{
    memset(l1, 0, BUF_SIZE);
    load_file("data.bin", l1, {src_bytes});

    dma_reformat(TID0, T(l1, 0, {_c_enum(src_dtype)}),
                       T(l1, {src_bytes}, {_c_enum(dst_dtype)}), {size_bytes});

    save_file("out.bin", l1 + {src_bytes}, {dst_bytes});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), _np_dtype(dst_dtype), count)

        if src_dtype == dst_dtype:
            assert np.array_equal(actual, data), f"dma_reformat {desc}: bitwise 不一致"
        else:
            expected = data.astype(np.float64).astype(_np_dtype(dst_dtype))
            r = _compare(actual, expected, atol=_DTYPE_CONV_ATOL)
            assert r["passed"], (
                f"dma_reformat {desc}: max_abs={r['max_abs']:.6f} cos={r['cosine']:.6f}"
            )


# ============================================================
# idma_reshape: L1 内搬运 + 可选 dtype 转换
# ============================================================

_IDMA_RESHAPE_PARAMS = [
    ("fp16", "fp16", 32,   "fp16_same_small"),
    ("fp32", "fp32", 256,  "fp32_same_medium"),
    ("fp32", "fp16", 256,  "fp32_to_fp16"),
    ("fp16", "fp32", 256,  "fp16_to_fp32"),
    ("fp32", "fp16", 4096, "fp32_to_fp16_large"),
    ("fp16", "fp32", 4096, "fp16_to_fp32_large"),
]


class TestIdmaReshape:
    """idma_reshape: L1 内 reshape/搬运 + dtype 转换。"""

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype,count,desc",
        _IDMA_RESHAPE_PARAMS,
        ids=[p[3] for p in _IDMA_RESHAPE_PARAMS],
    )
    def test_idma_reshape(self, tmp_path, src_dtype, dst_dtype, count, desc):
        rng = np.random.RandomState(44)
        data = rng.randn(count).astype(_np_dtype(src_dtype))
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "data.bin"), data)

        src_bytes = count * _elem_size(src_dtype)
        dst_bytes = count * _elem_size(dst_dtype)
        # idma_reshape size = 按 src 字节数
        size = src_bytes

        c_code = _C_PREAMBLE + f"""
int main(void) {{
    memset(l1, 0, BUF_SIZE);
    load_file("data.bin", l1, {src_bytes});

    idma_reshape(TID0, T(l1, 0, {_c_enum(src_dtype)}),
                       T(l1, {src_bytes}, {_c_enum(dst_dtype)}), {size});

    save_file("out.bin", l1 + {src_bytes}, {dst_bytes});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), _np_dtype(dst_dtype), count)

        if src_dtype == dst_dtype:
            assert np.array_equal(actual, data), f"idma_reshape {desc}: bitwise 不一致"
        else:
            expected = data.astype(np.float64).astype(_np_dtype(dst_dtype))
            r = _compare(actual, expected, atol=_DTYPE_CONV_ATOL)
            assert r["passed"], (
                f"idma_reshape {desc}: max_abs={r['max_abs']:.6f} cos={r['cosine']:.6f}"
            )


# ============================================================
# idma_move: L1 内搬运 + 可选 dtype 转换
# ============================================================

_IDMA_MOVE_PARAMS = [
    ("fp16", "fp16", 64,   "fp16_same"),
    ("fp32", "fp32", 64,   "fp32_same"),
    ("fp32", "fp16", 256,  "fp32_to_fp16"),
    ("fp16", "fp32", 256,  "fp16_to_fp32"),
]


class TestIdmaMove:
    """idma_move: L1 内搬运 + dtype 转换。"""

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype,count,desc",
        _IDMA_MOVE_PARAMS,
        ids=[p[3] for p in _IDMA_MOVE_PARAMS],
    )
    def test_idma_move(self, tmp_path, src_dtype, dst_dtype, count, desc):
        rng = np.random.RandomState(45)
        data = rng.randn(count).astype(_np_dtype(src_dtype))
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "data.bin"), data)

        src_bytes = count * _elem_size(src_dtype)
        dst_bytes = count * _elem_size(dst_dtype)
        size_bytes = src_bytes

        c_code = _C_PREAMBLE + f"""
int main(void) {{
    memset(l1, 0, BUF_SIZE);
    load_file("data.bin", l1, {src_bytes});

    idma_move(TID0, T(l1, {src_bytes}, {_c_enum(dst_dtype)}),
                    T(l1, 0, {_c_enum(src_dtype)}), {size_bytes});

    save_file("out.bin", l1 + {src_bytes}, {dst_bytes});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), _np_dtype(dst_dtype), count)

        if src_dtype == dst_dtype:
            assert np.array_equal(actual, data), f"idma_move {desc}: bitwise 不一致"
        else:
            expected = data.astype(np.float64).astype(_np_dtype(dst_dtype))
            r = _compare(actual, expected, atol=_DTYPE_CONV_ATOL)
            assert r["passed"], (
                f"idma_move {desc}: max_abs={r['max_abs']:.6f} cos={r['cosine']:.6f}"
            )


# ============================================================
# idma_broadcast: 标量扩展
# ============================================================

_IDMA_BROADCAST_PARAMS = [
    ("fp16", "fp16", 64,   "fp16_same"),
    ("fp32", "fp32", 64,   "fp32_same"),
    ("fp32", "fp16", 256,  "fp32_to_fp16"),
    ("fp16", "fp32", 256,  "fp16_to_fp32"),
]


class TestIdmaBroadcast:
    """idma_broadcast: 标量广播到数组 + dtype 转换。"""

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype,count,desc",
        _IDMA_BROADCAST_PARAMS,
        ids=[p[3] for p in _IDMA_BROADCAST_PARAMS],
    )
    def test_idma_broadcast(self, tmp_path, src_dtype, dst_dtype, count, desc):
        rng = np.random.RandomState(46)
        scalar_val = rng.randn(1).astype(_np_dtype(src_dtype))
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "scalar.bin"), scalar_val)

        src_bytes = _elem_size(src_dtype)
        dst_bytes = count * _elem_size(dst_dtype)

        c_code = _C_PREAMBLE + f"""
int main(void) {{
    memset(l1, 0, BUF_SIZE);
    load_file("scalar.bin", l1, {src_bytes});

    idma_broadcast(TID0, T(l1, 0, {_c_enum(src_dtype)}),
                         T(l1, 1024, {_c_enum(dst_dtype)}), 1, {count});

    save_file("out.bin", l1 + 1024, {dst_bytes});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), _np_dtype(dst_dtype), count)

        expected_val = float(scalar_val[0])
        expected = np.full(count, expected_val, dtype=np.float64).astype(_np_dtype(dst_dtype))
        if src_dtype == dst_dtype:
            assert np.array_equal(actual, expected), f"broadcast {desc}: 不一致"
        else:
            r = _compare(actual, expected, atol=_DTYPE_CONV_ATOL)
            assert r["passed"], (
                f"broadcast {desc}: max_abs={r['max_abs']:.6f} cos={r['cosine']:.6f}"
            )


# ============================================================
# Roundtrip: HBM -> L1 (dma_move) -> reshape (idma) -> HBM (dma_move)
# ============================================================

_ROUNDTRIP_PARAMS = [
    ("fp16", "fp16", "fp16", 128, "fp16_no_convert"),
    ("fp32", "fp32", "fp32", 128, "fp32_no_convert"),
    ("fp32", "fp16", "fp16", 256, "fp32_load_fp16_compute_fp16_store"),
    ("fp32", "fp16", "fp32", 256, "fp32_load_fp16_compute_fp32_store"),
    ("fp16", "fp32", "fp16", 256, "fp16_load_fp32_compute_fp16_store"),
]


class TestDmaRoundtrip:
    """端到端搬运链: HBM→L1(dma)→L1(idma_reshape)→HBM(dma)。"""

    @pytest.mark.parametrize(
        "hbm_dtype,l1_dtype,out_dtype,count,desc",
        _ROUNDTRIP_PARAMS,
        ids=[p[4] for p in _ROUNDTRIP_PARAMS],
    )
    def test_roundtrip(self, tmp_path, hbm_dtype, l1_dtype, out_dtype, count, desc):
        rng = np.random.RandomState(47)
        data = rng.randn(count).astype(_np_dtype(hbm_dtype))
        workdir = str(tmp_path)
        _write_bin(os.path.join(workdir, "data.bin"), data)

        hbm_bytes = count * _elem_size(hbm_dtype)
        l1_bytes = count * _elem_size(l1_dtype)
        out_bytes = count * _elem_size(out_dtype)

        # offsets in L1
        off_in = 0
        off_reshape = l1_bytes + 1024  # 留间隔
        # offsets in HBM
        hbm_in = 0
        hbm_out = hbm_bytes + 1024

        c_code = _C_PREAMBLE + f"""
int main(void) {{
    memset(hbm, 0, BUF_SIZE);
    memset(l1, 0, BUF_SIZE);
    load_file("data.bin", hbm + {hbm_in}, {hbm_bytes});

    /* Step 1: HBM -> L1 (dma_move, 可能 dtype 转换) */
    dma_move(TID0, T(l1, {off_in}, {_c_enum(l1_dtype)}),
                   T(hbm, {hbm_in}, {_c_enum(hbm_dtype)}), {hbm_bytes});

    /* Step 2: L1 -> L1 (idma_reshape, 同 dtype) */
    idma_reshape(TID0, T(l1, {off_in}, {_c_enum(l1_dtype)}),
                       T(l1, {off_reshape}, {_c_enum(l1_dtype)}), {l1_bytes});

    /* Step 3: L1 -> HBM (dma_move, 可能 dtype 转换) */
    dma_move(TID0, T(hbm, {hbm_out}, {_c_enum(out_dtype)}),
                   T(l1, {off_reshape}, {_c_enum(l1_dtype)}), {l1_bytes});

    save_file("out.bin", hbm + {hbm_out}, {out_bytes});
    return 0;
}}
"""
        _compile_and_run(c_code, workdir)
        actual = _read_bin(os.path.join(workdir, "out.bin"), _np_dtype(out_dtype), count)

        # 模拟多步转换: hbm_dtype -> l1_dtype -> l1_dtype (copy) -> out_dtype
        expected = (
            data.astype(np.float64)
            .astype(_np_dtype(l1_dtype))
            .astype(np.float64)
            .astype(_np_dtype(out_dtype))
        )

        if hbm_dtype == l1_dtype == out_dtype:
            assert np.array_equal(actual, expected), f"roundtrip {desc}: bitwise 不一致"
        else:
            r = _compare(actual, expected, atol=_DTYPE_CONV_ATOL)
            assert r["passed"], (
                f"roundtrip {desc}: max_abs={r['max_abs']:.6f} cos={r['cosine']:.6f}"
            )
