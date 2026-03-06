# common — 基础设施模块

## 职责

提供所有其他模块共享的基础设施：图IR数据结构、日志系统、配置加载、异常定义。

## 本模块必须最先开发

所有其他模块仅依赖common，互相无依赖。common完成后，其他模块可完全并行开发。

## 文件说明

### graph_ir.py

定义 Graph / Node / Tensor 三个核心dataclass。
提供图操作方法：add_node, remove_node, topo_sort, to_dict, from_dict, validate, summary。

核心数据结构：

```python
@dataclass
class Tensor:
    id: str                    # 全局唯一ID
    shape: list[int]           # 如 [1, 32, 64]
    dtype: str                 # "fp16" | "fp32" | ...
    format: str = "nd"         # "nd" | "nz" | "nc1hwc0"
    hbm_offset: Optional[int] = None
    hbm_size: Optional[int] = None
    l1_offset: Optional[int] = None
    is_weight: bool = False
    is_model_input: bool = False
    is_model_output: bool = False
    producer_node_id: Optional[str] = None
    consumer_node_ids: list[str] = field(default_factory=list)

@dataclass
class Node:
    id: str                    # 全局唯一ID
    op_type: str               # 如 "aten.mm" 或 "npu_matmul"
    inputs: list[str]          # 输入tensor ID列表
    outputs: list[str]         # 输出tensor ID列表
    params: dict               # 算子参数
    compute_unit: Optional[str] = None    # "cube" | "vector" | "scalar"
    npu_op: Optional[str] = None
    is_mapped: bool = False
    format_annotation: Optional[dict] = None   # 三元组：inputs/outputs + compute_dtype
    schedule_order: Optional[int] = None
    dependencies: list[str] = field(default_factory=list)
    absorbed_inputs: dict = field(default_factory=dict)

@dataclass
class Graph:
    nodes: dict[str, Node]
    tensors: dict[str, Tensor]
    execution_order: list[str]
```

### logger.py

统一日志系统。日志格式：`[时间] [级别] [模块名] 消息`

支持通过环境变量 `NPU_LOG_LEVEL` 设置级别（默认INFO）。

接口：
- `get_logger(name: str) -> logging.Logger`
- `setup_logging(level: str = "INFO", log_file: str = None) -> None`

### config_loader.py

YAML配置加载与schema校验。

接口：
- `load_config(path: str, required_keys: list[str] = None) -> dict`
  - 加载YAML文件，校验必填字段存在，返回dict
  - 缺失必填字段时抛出 ConfigError

### errors.py

统一异常定义：

| 异常类 | 用途 |
|--------|------|
| CompilerError | 基类 |
| ConfigError | 配置文件错误 |
| MappingError | 算子映射失败 |
| DecompositionError | 算子裂解失败 |
| AbsorptionError | 参数吸收失败 |
| ValidationError | 合法性校验失败 |
| MemoryPlanError | 内存编排失败 |
| CodegenError | 代码生成失败 |

## UT

**test_graph_ir.py:**
- `test_add_remove_node`: 增删节点后节点数正确
- `test_topo_sort`: 线性链的拓扑排序结果正确
- `test_to_dict_from_dict`: 序列化→反序列化后图一致
- `test_validate`: 悬空引用能被检测到

**test_config_loader.py:**
- `test_load_valid`: 正常YAML加载成功
- `test_missing_key`: 缺少必填字段时抛出ConfigError
- `test_file_not_found`: 文件不存在时抛出FileNotFoundError
