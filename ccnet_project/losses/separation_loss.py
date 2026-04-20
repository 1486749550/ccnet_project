import torch
import torch.nn as nn


class FeatureSeparationLoss(nn.Module):
    """
    特征分离损失。

    论文没有完全公开 content/style reshape 的细节，这里采用工程化近似方案：
    - 对最深层内容特征 c5 做 GAP，得到 [B, C]
    - 将 style code [B, C, 1, 1] reshape 为 [B, C]
    - 计算二者的相关系数矩阵，并最小化其 Frobenius norm 的平方
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, c5: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        assert c5.ndim == 4, f"c5 必须为 [B,C,H,W]，当前为: {c5.shape}"
        assert style.ndim == 4 and style.shape[-2:] == (1, 1), f"style 必须为 [B,C,1,1]，当前为: {style.shape}"
        content_vec = c5.mean(dim=(2, 3))
        style_vec = style.view(style.shape[0], style.shape[1])
        assert content_vec.shape == style_vec.shape, f"content/style 展平后尺寸必须一致: {content_vec.shape} vs {style_vec.shape}"

        content_vec = content_vec - content_vec.mean(dim=0, keepdim=True)
        style_vec = style_vec - style_vec.mean(dim=0, keepdim=True)
        covariance = torch.matmul(content_vec.transpose(0, 1), style_vec) / max(content_vec.shape[0] - 1, 1)
        loss = torch.norm(covariance, p="fro") ** 2 / covariance.numel()
        return loss
