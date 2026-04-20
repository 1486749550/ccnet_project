from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from losses.total_loss import TotalLoss
from metrics.seg_metrics import ConfusionMatrixMeter
from utils.misc import AverageMeter


class Evaluator:
    """验证器，负责累计整个验证集的损失和混淆矩阵指标。"""

    def __init__(self, model: torch.nn.Module, criterion: TotalLoss, device: torch.device, threshold: float = 0.75) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.threshold = threshold
        self.metric_meter = ConfusionMatrixMeter()

    def _predict_with_threshold(self, change_logits: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        基于变化类概率图执行动态二值化。

        Args:
            change_logits: [B, 2, H, W]
            threshold: 变化类概率阈值

        Returns:
            pred_mask: [B, H, W]
        """
        assert change_logits.ndim == 4 and change_logits.shape[1] == 2, (
            f"change_logits 必须为 [B,2,H,W]，当前为: {change_logits.shape}"
        )

        # softmax 后 probs 仍为 [B, 2, H, W]
        probs = torch.softmax(change_logits, dim=1)
        # 取变化类概率图，形状变为 [B, H, W]
        change_prob = probs[:, 1]
        assert change_prob.ndim == 3, f"变化类概率图必须为 [B,H,W]，当前为: {change_prob.shape}"

        pred_mask = (change_prob >= threshold).long()
        assert pred_mask.shape == change_prob.shape, "阈值化后的预测 mask 形状错误"
        return pred_mask

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, threshold: float | None = None) -> Dict[str, float | int]:
        self.model.eval()
        self.metric_meter.reset()
        loss_meter = AverageMeter()
        active_threshold = self.threshold if threshold is None else float(threshold)

        for batch in dataloader:
            batch = self._move_to_device(batch)
            outputs = self.model(batch["img_t1"], batch["img_t2"])
            losses = self.criterion(outputs, batch["img_t1"], batch["img_t2"], batch["mask"])
            loss_meter.update(float(losses["loss_total"].item()), batch["img_t1"].shape[0])

            pred_mask = self._predict_with_threshold(outputs["change_logits"], active_threshold)
            self.metric_meter.update(pred_mask, batch["mask"])

        scores = self.metric_meter.compute()
        scores["loss_total"] = float(loss_meter.avg)
        scores["threshold"] = float(active_threshold)
        return scores

    @torch.no_grad()
    def evaluate_thresholds(self, dataloader: DataLoader, thresholds: Iterable[float]) -> Dict[str, Dict[str, float | int]]:
        """扫描多个阈值并返回每个阈值对应的验证结果。"""
        results: Dict[str, Dict[str, float | int]] = {}
        for threshold in thresholds:
            scores = self.evaluate(dataloader=dataloader, threshold=float(threshold))
            results[f"{float(threshold):.2f}"] = scores
        return results

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch["img_t1"] = batch["img_t1"].to(self.device, non_blocking=True)
        batch["img_t2"] = batch["img_t2"].to(self.device, non_blocking=True)
        batch["mask"] = batch["mask"].to(self.device, non_blocking=True)
        return batch
