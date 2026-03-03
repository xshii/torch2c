"""统一异常定义。"""


class CompilerError(Exception):
    """编译器基类异常。"""


class ConfigError(CompilerError):
    """配置文件错误。"""


class MappingError(CompilerError):
    """算子映射失败。"""


class DecompositionError(CompilerError):
    """算子裂解失败。"""


class AbsorptionError(CompilerError):
    """参数吸收失败。"""


class ValidationError(CompilerError):
    """合法性校验失败。"""


class MemoryPlanError(CompilerError):
    """内存编排失败。"""


class CodegenError(CompilerError):
    """代码生成失败。"""
