"""公共路径定义 — 项目级目录的单一来源。"""

from pathlib import Path

# torch2c 包根目录 (即 torch2c/)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# 项目根目录 (即 torch2c/ 的上层，含 npu_cpu_mock/, docs/ 等)
PROJECT_ROOT = PACKAGE_ROOT.parent

# 常用目录
INTEGRATION_CONFIG_DIR = PACKAGE_ROOT / "integration" / "config"
NPU_CPU_MOCK_DIR = PROJECT_ROOT / "npu_cpu_mock"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
HARDWARE_CONFIG_PATH = INTEGRATION_CONFIG_DIR / "hardware_config.yaml"
