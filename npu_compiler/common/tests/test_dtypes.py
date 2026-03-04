"""dtype 元信息单元测试。"""

from npu_compiler.common.dtypes import DTYPE_INFO, dtype_bytes, dtype_c_enum


def test_dtype_bytes_fp16():
    assert dtype_bytes("fp16") == 2


def test_dtype_bytes_fp32():
    assert dtype_bytes("fp32") == 4


def test_dtype_bytes_unknown_defaults():
    assert dtype_bytes("unknown") == 2


def test_dtype_c_enum():
    assert dtype_c_enum("fp16") == "NPU_DTYPE_FP16"
    assert dtype_c_enum("int32") == "NPU_DTYPE_INT32"


def test_info_consistent():
    for name, info in DTYPE_INFO.items():
        assert dtype_bytes(name) == info.bytes
        assert dtype_c_enum(name) == info.c_enum
