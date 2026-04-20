import torch
import torch.nn as nn

from models.blocks import ConvBNReLU


class StyleEncoder(nn.Module):
    """
    风格编码器。

    该模块提取的是整幅图像的全局风格统计，例如季节、光照、传感器响应和色调偏移，
    而不是逐像素的空间细节。因此它会不断降低空间分辨率，最后压缩为 [B, C, 1, 1]。
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, kernel_size=3, stride=2),
            ConvBNReLU(base_channels, base_channels * 2, kernel_size=3, stride=2),
            ConvBNReLU(base_channels * 2, base_channels * 4, kernel_size=3, stride=2),
            ConvBNReLU(base_channels * 4, base_channels * 8, kernel_size=3, stride=2),
            ConvBNReLU(base_channels * 8, base_channels * 16, kernel_size=3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"StyleEncoder 输入必须为四维张量，当前为: {x.shape}"
        feat = self.encoder(x)
        style = self.pool(feat)
        assert style.ndim == 4, f"style code 必须为四维张量，当前为: {style.shape}"
        assert style.shape[-2:] == (1, 1), f"style code 空间尺寸必须为 1x1，当前为: {style.shape}"
        return style
