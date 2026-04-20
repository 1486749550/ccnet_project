from pathlib import Path
from typing import Any, Dict

import yaml


class AverageMeter:
    """统计均值的基础工具类。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def load_yaml(path: str) -> Dict[str, Any]:
    """加载 yaml 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    """保证目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)
