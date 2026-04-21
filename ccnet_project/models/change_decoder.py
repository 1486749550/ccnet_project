import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBNReLU


# 请用 PyTorch 实现一个轻量级的 Squeeze-and-Excitation (SE) Block，包含 Global Average Pooling 和两层 Linear/Conv1d 层，使用 ReLU 和 Sigmoid 激活函数。
class SEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"SEBlock input must be [B,C,H,W], got: {x.shape}"
        batch_size, channels, _, _ = x.shape
        weights = self.global_avg_pool(x).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1, 1)
        return x * weights


class DecodeFuseBlock(nn.Module):
    """变化解码器中的一级融合块。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, stride=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and skip.ndim == 4, f"DecodeFuseBlock 输入必须为四维张量，当前为: {x.shape}, {skip.shape}"
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([x, skip], dim=1)
        out = self.fuse(fused)
        assert out.shape[-2:] == skip.shape[-2:], "解码后的空间尺寸必须与 skip 一致"
        return out


class ChangeDecoder(nn.Module):
    """
    变化检测解码器。

    严格按要求实现：
    - d5_cd = bottleneck(concat(c5_t1, c5_t2, abs(c5_t1 - c5_t2)))
    - d4 <- up(d5_cd) + concat(c4_t1, c4_t2)
    - ...
    - 输出 2 通道 logits
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.bottleneck = nn.Sequential(
            ConvBNReLU(c5 * 3, c5, kernel_size=3, stride=1),
            ConvBNReLU(c5, c5, kernel_size=3, stride=1),
        )
        # 初始化 SEBlock，输入通道数为 bottleneck 的输出通道数
        self.bottleneck_se = SEBlock(c5)
        self.decode4 = DecodeFuseBlock(c5, c4 * 2, c4)
        self.decode3 = DecodeFuseBlock(c4, c3 * 2, c3)
        self.decode2 = DecodeFuseBlock(c3, c2 * 2, c2)
        self.decode1 = DecodeFuseBlock(c2, c1 * 2, c1)
        self.head = nn.Sequential(
            ConvBNReLU(c1, c1, kernel_size=3, stride=1),
            nn.Conv2d(c1, 2, kernel_size=1, stride=1),
        )

    def forward(self, content_t1: list[torch.Tensor], content_t2: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(content_t1) == 5 and len(content_t2) == 5, "ChangeDecoder 需要 5 级内容特征"
        c1_t1, c2_t1, c3_t1, c4_t1, c5_t1 = content_t1
        c1_t2, c2_t2, c3_t2, c4_t2, c5_t2 = content_t2
        c5_fused = torch.cat([c5_t1, c5_t2, torch.abs(c5_t1 - c5_t2)], dim=1)
        d5_cd = self.bottleneck(c5_fused)
        # 将 bottleneck 融合后的特征送入 SEBlock 进行通道级注意力加权过滤
        d5_cd = self.bottleneck_se(d5_cd)
        d4 = self.decode4(d5_cd, torch.cat([c4_t1, c4_t2], dim=1))
        d3 = self.decode3(d4, torch.cat([c3_t1, c3_t2], dim=1))
        d2 = self.decode2(d3, torch.cat([c2_t1, c2_t2], dim=1))
        d1 = self.decode1(d2, torch.cat([c1_t1, c1_t2], dim=1))
        change_logits = self.head(d1)
        pred_mask = torch.argmax(torch.softmax(change_logits, dim=1), dim=1)
        assert change_logits.ndim == 4 and change_logits.shape[1] == 2, f"change_logits 必须为 [B,2,H,W]，当前为: {change_logits.shape}"
        assert pred_mask.ndim == 3, f"pred_mask 必须为 [B,H,W]，当前为: {pred_mask.shape}"
        return change_logits, pred_mask
