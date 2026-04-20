from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], save_path: str) -> None:
    """保存模型与训练状态。"""
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_file)


def load_checkpoint(save_path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """读取 checkpoint。"""
    assert Path(save_path).exists(), f"checkpoint 不存在: {save_path}"
    return torch.load(save_path, map_location=map_location)
