from typing import Dict

import torch


class ConfusionMatrixMeter:
    """基于整个验证集累计混淆矩阵的二分类指标统计器。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        assert pred.shape == target.shape, f"pred 与 target 形状必须一致，当前为: {pred.shape} vs {target.shape}"
        pred = pred.long()
        target = target.long()
        self.tp += int(((pred == 1) & (target == 1)).sum().item())
        self.tn += int(((pred == 0) & (target == 0)).sum().item())
        self.fp += int(((pred == 1) & (target == 0)).sum().item())
        self.fn += int(((pred == 0) & (target == 1)).sum().item())

    def compute(self) -> Dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-6)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "iou": float(iou),
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
        }


SegmentationMetricMeter = ConfusionMatrixMeter
