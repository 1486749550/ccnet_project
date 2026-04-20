import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adain import AdaIN2d
from models.blocks import ConvBNReLU


class RestorationStage(nn.Module):
    """恢复解码器中的单级恢复块。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, stride=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1),
        )
        self.adain = AdaIN2d()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and skip.ndim == 4 and style.ndim == 4, "RestorationStage 输入维度错误"
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([x, skip], dim=1)
        out = self.conv(fused)
        out = self.adain(out, style)
        assert out.shape[-2:] == skip.shape[-2:], "恢复阶段输出尺寸必须与 skip 一致"
        return out


class RestorationDecoder(nn.Module):
    """
    共享权重的图像恢复解码器。

    对于每一个时相：
    - 最深层用 bottleneck(c5, s_tn) 初始化
    - 每一级都执行 上采样 + 拼接内容特征 + 卷积融合 + AdaIN 注入风格
    """

    def __init__(self, base_channels: int = 32, out_channels: int = 3) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.bottleneck_conv = nn.Sequential(
            ConvBNReLU(c5, c5, kernel_size=3, stride=1),
            ConvBNReLU(c5, c5, kernel_size=3, stride=1),
        )
        self.bottleneck_adain = AdaIN2d()
        self.stage4 = RestorationStage(c5, c4, c4)
        self.stage3 = RestorationStage(c4, c3, c3)
        self.stage2 = RestorationStage(c3, c2, c2)
        self.stage1 = RestorationStage(c2, c1, c1)
        self.head = nn.Sequential(
            ConvBNReLU(c1, c1, kernel_size=3, stride=1),
            nn.Conv2d(c1, out_channels, kernel_size=1, stride=1),
        )

    def _resize_style(self, style: torch.Tensor, channels: int) -> torch.Tensor:
        """将 [B,512,1,1] 的 style 调整到当前层通道数。"""
        assert style.ndim == 4 and style.shape[-2:] == (1, 1), f"style 必须为 [B,C,1,1]，当前为: {style.shape}"
        if style.shape[1] == channels:
            return style
        if style.shape[1] > channels:
            return style[:, :channels]
        repeat_factor = (channels + style.shape[1] - 1) // style.shape[1]
        resized = style.repeat(1, repeat_factor, 1, 1)[:, :channels]
        assert resized.shape[1] == channels, "style 通道调整失败"
        return resized

    def forward(self, content_feats: list[torch.Tensor], style: torch.Tensor) -> torch.Tensor:
        assert len(content_feats) == 5, "RestorationDecoder 需要 5 级内容特征"
        c1, c2, c3, c4, c5 = content_feats
        assert style.shape[1] == c5.shape[1], "style 通道必须和最深层内容特征通道一致"

        d5_rst = self.bottleneck_conv(c5)
        d5_rst = self.bottleneck_adain(d5_rst, style)
        d4 = self.stage4(d5_rst, c4, self._resize_style(style, c4.shape[1]))
        d3 = self.stage3(d4, c3, self._resize_style(style, c3.shape[1]))
        d2 = self.stage2(d3, c2, self._resize_style(style, c2.shape[1]))
        d1 = self.stage1(d2, c1, self._resize_style(style, c1.shape[1]))
        rst = self.head(d1)
        assert rst.ndim == 4 and rst.shape[1] == 3, f"恢复图像必须为 [B,3,H,W]，当前为: {rst.shape}"
        return rst
