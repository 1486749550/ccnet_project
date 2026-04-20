import torch
import torch.nn as nn
import torch.nn.functional as F


class RestorationLoss(nn.Module):
    """图像恢复损失，对两个时相分别做 L1，再取平均。"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, rst_t1: torch.Tensor, x_t1: torch.Tensor, rst_t2: torch.Tensor, x_t2: torch.Tensor) -> torch.Tensor:
        assert rst_t1.shape == x_t1.shape, f"rst_t1 与 x_t1 形状必须一致: {rst_t1.shape} vs {x_t1.shape}"
        assert rst_t2.shape == x_t2.shape, f"rst_t2 与 x_t2 形状必须一致: {rst_t2.shape} vs {x_t2.shape}"
        loss_t1 = F.l1_loss(rst_t1, x_t1)
        loss_t2 = F.l1_loss(rst_t2, x_t2)
        return 0.5 * (loss_t1 + loss_t2)
