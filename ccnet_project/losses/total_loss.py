from typing import Dict

import torch
import torch.nn as nn

from losses.change_loss import ChangeDetectionLoss
from losses.restoration_loss import RestorationLoss
from losses.separation_loss import FeatureSeparationLoss
from losses.similarity_loss import ContentSimilarityLoss


class TotalLoss(nn.Module):
    """
    CCNet 总损失：
    L = L_cd + λ1 * L_sep + λ2 * L_sim + λ3 * L_rst

    当前版本为了更强地压制误检，放大了 sep / sim 的默认权重，
    同时将变化检测主损失替换为 BCE + Tversky。
    """

    def __init__(
        self,
        lambda_sep: float = 0.3,
        lambda_sim: float = 0.3,
        lambda_rst: float = 0.1,
        tversky_alpha: float = 0.8,
        tversky_beta: float = 0.2,
        tversky_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_sep = lambda_sep
        self.lambda_sim = lambda_sim
        self.lambda_rst = lambda_rst

        self.change_loss = ChangeDetectionLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            tversky_weight=tversky_weight,
        )
        self.separation_loss = FeatureSeparationLoss()
        self.similarity_loss = ContentSimilarityLoss()
        self.restoration_loss = RestorationLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor | list[torch.Tensor]],
        x_t1: torch.Tensor,
        x_t2: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l_cd = self.change_loss(outputs["change_logits"], target_mask)
        l_sep = self.separation_loss(outputs["content_t1"][-1], outputs["style_t1"]) + self.separation_loss(
            outputs["content_t2"][-1], outputs["style_t2"]
        )
        l_sim = self.similarity_loss(outputs["content_t1"], outputs["content_t2"], target_mask)
        l_rst = self.restoration_loss(outputs["rst_t1"], x_t1, outputs["rst_t2"], x_t2)
        total = l_cd + self.lambda_sep * l_sep + self.lambda_sim * l_sim + self.lambda_rst * l_rst
        return {
            "loss_total": total,
            "loss_cd": l_cd,
            "loss_sep": l_sep,
            "loss_sim": l_sim,
            "loss_rst": l_rst,
        }
