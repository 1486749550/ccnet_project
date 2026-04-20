from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """最基础的卷积块，便于在全工程中统一使用。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"ConvBNReLU 输入必须为 4 维，当前为: {x.shape}"
        y = self.block(x)
        assert y.ndim == 4, f"ConvBNReLU 输出必须为 4 维，当前为: {y.shape}"
        return y


class ResidualConvBlock(nn.Module):
    """残差卷积块，用于增强内容特征表达能力。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels, kernel_size=3, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"ResidualConvBlock 输入必须为 4 维，当前为: {x.shape}"
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        assert out.shape == identity.shape, f"残差连接前后尺寸不一致: {out.shape} vs {identity.shape}"
        out = self.relu(out + identity)
        return out


class DownsampleBlock(nn.Module):
    """带 stride 的下采样卷积块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"DownsampleBlock 输入维度错误: {x.shape}"
        y = self.block(x)
        assert y.shape[-1] <= x.shape[-1] and y.shape[-2] <= x.shape[-2], "下采样未生效"
        return y


class UpsampleFuseBlock(nn.Module):
    """
    上采样并与 skip 特征融合。

    该模块在变化解码器和重建解码器中都会被复用。
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, stride=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and skip.ndim == 4, f"UpsampleFuseBlock 输入维度错误: {x.shape}, {skip.shape}"
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        assert x.shape[-2:] == skip.shape[-2:], "上采样后与 skip 尺寸不一致"
        fused = torch.cat([x, skip], dim=1)
        out = self.fuse(fused)
        return out


def split_feature_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算每个样本每个通道的均值和标准差，供 AdaIN 使用。"""
    assert x.ndim == 4, f"split_feature_stats 输入必须是 4 维，当前为: {x.shape}"
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + 1e-6
    assert mean.shape == std.shape, "均值和标准差形状必须一致"
    return mean, std
