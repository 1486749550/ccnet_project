import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Loss。

    该损失是 Dice Loss 的加权扩展版本：
    - alpha 越大，对 False Positive（误检）的惩罚越重
    - beta 越大，对 False Negative（漏检）的惩罚越重

    当前任务中误检非常严重，因此默认设置为：
    - alpha = 0.8
    - beta = 0.2
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 2, H, W]，模型输出的两通道 logits
            target_onehot: [B, 2, H, W]，标签的 one-hot 表示
        """
        assert logits.ndim == 4 and logits.shape[1] == 2, f"TverskyLoss 的 logits 必须为 [B,2,H,W]，当前为: {logits.shape}"
        assert target_onehot.ndim == 4 and target_onehot.shape == logits.shape, (
            f"TverskyLoss 的 target_onehot 必须与 logits 同形状，当前为: {target_onehot.shape} vs {logits.shape}"
        )

        # softmax 后得到两类的概率图，形状仍为 [B, 2, H, W]
        probs = torch.softmax(logits, dim=1)
        assert probs.shape == target_onehot.shape, "softmax 后的概率图与 one-hot 标签形状必须一致"

        # 仅对“变化类”计算 Tversky，可更直接抑制变化类误检
        # change_prob / change_target: [B, H, W]
        change_prob = probs[:, 1]
        change_target = target_onehot[:, 1]
        assert change_prob.shape == change_target.shape, "变化类概率图与变化类标签形状必须一致"

        # 展平为 [B, H*W]，按样本统计 TP / FP / FN
        change_prob = change_prob.flatten(1)
        change_target = change_target.flatten(1)
        assert change_prob.ndim == 2 and change_target.ndim == 2, "展平后张量必须为二维 [B, N]"

        true_positive = (change_prob * change_target).sum(dim=1)
        false_positive = (change_prob * (1.0 - change_target)).sum(dim=1)
        false_negative = ((1.0 - change_prob) * change_target).sum(dim=1)

        tversky = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        loss = 1.0 - tversky
        return loss.mean()


class ChangeDetectionLoss(nn.Module):
    """
    变化检测损失。

    为了在保持 BCE 稳定性的同时更强地压制误检，
    这里采用：
    - BCEWithLogitsLoss
    - TverskyLoss

    最终：
    L_cd = BCE + tversky_weight * Tversky
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, tversky_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.tversky_weight = tversky_weight

    def forward(self, change_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        assert change_logits.ndim == 4 and change_logits.shape[1] == 2, (
            f"change_logits 必须为 [B,2,H,W]，当前为: {change_logits.shape}"
        )
        assert target_mask.ndim == 3, f"target_mask 必须为 [B,H,W]，当前为: {target_mask.shape}"

        # target_mask: [B, H, W] -> target_onehot: [B, 2, H, W]
        target_onehot = F.one_hot(target_mask.long(), num_classes=2).permute(0, 3, 1, 2).float()
        assert target_onehot.shape == change_logits.shape, (
            f"one-hot 标签尺寸错误: {target_onehot.shape} vs {change_logits.shape}"
        )

        bce_loss = self.bce(change_logits, target_onehot)
        tversky_loss = self.tversky(change_logits, target_onehot)
        total = bce_loss + self.tversky_weight * tversky_loss
        return total
