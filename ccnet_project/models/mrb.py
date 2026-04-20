from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBNReLU, ResidualConvBlock


class MRBBranch(nn.Module):
    """
    MRB 的单个分支。

    每个分支对应一个固定分辨率，内部由若干残差块构成，
    用于在该尺度上持续提取内容特征。
    """

    def __init__(self, channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualConvBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"MRBBranch 输入必须为四维张量，当前为: {x.shape}"
        y = self.blocks(x)
        assert y.shape == x.shape, f"MRBBranch 输出尺寸必须与输入一致，当前为: {y.shape} vs {x.shape}"
        return y


class FuseLayer(nn.Module):
    """
    多尺度特征融合层。

    对于来自其他分支的特征，若分辨率更高则先下采样，若分辨率更低则先上采样，
    然后通过卷积投影到目标分支通道数，再进行逐元素相加。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, source: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        assert source.ndim == 4, f"FuseLayer 输入必须为四维张量，当前为: {source.shape}"
        resized = F.interpolate(source, size=target_size, mode="bilinear", align_corners=False)
        out = self.proj(resized)
        assert out.shape[-2:] == target_size, f"FuseLayer 输出空间尺寸错误: {out.shape[-2:]} vs {target_size}"
        return out


class MaintainResolutionBackbone(nn.Module):
    """
    Maintain Resolution Backbone（MRB）。

    这里采用工程化实现：
    - 使用 5 个并行分支分别维护 H、H/2、H/4、H/8、H/16 五个尺度
    - 每个分支内部由 residual blocks 组成
    - 在分支之间加入显式多尺度融合，利用上采样/下采样进行信息交换

    输出五级内容特征：
    - c1: [B, 32, H, W]
    - c2: [B, 64, H/2, W/2]
    - c3: [B, 128, H/4, W/4]
    - c4: [B, 256, H/8, W/8]
    - c5: [B, 512, H/16, W/16]
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_blocks: int = 2, num_fusion_stages: int = 2) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.channels = channels
        self.stem = ConvBNReLU(in_channels, channels[0], kernel_size=3, stride=1)
        self.down12 = ConvBNReLU(channels[0], channels[1], kernel_size=3, stride=2)
        self.down23 = ConvBNReLU(channels[1], channels[2], kernel_size=3, stride=2)
        self.down34 = ConvBNReLU(channels[2], channels[3], kernel_size=3, stride=2)
        self.down45 = ConvBNReLU(channels[3], channels[4], kernel_size=3, stride=2)

        self.branches = nn.ModuleList([MRBBranch(ch, num_blocks=num_blocks) for ch in channels])
        self.fusion_layers = nn.ModuleList(
            [
                nn.ModuleList([FuseLayer(channels[src], channels[dst]) for src in range(5) if src != dst])
                for dst in range(5)
            ]
        )
        self.num_fusion_stages = num_fusion_stages
        self.post = nn.ModuleList([ConvBNReLU(ch, ch, kernel_size=3, stride=1) for ch in channels])

    def _init_branches(self, x: torch.Tensor) -> List[torch.Tensor]:
        """构造五个尺度的初始特征。"""
        c1 = self.stem(x)
        c2 = self.down12(c1)
        c3 = self.down23(c2)
        c4 = self.down34(c3)
        c5 = self.down45(c4)
        feats = [c1, c2, c3, c4, c5]
        assert feats[4].shape[1] == self.channels[4], f"最深层通道数错误，当前为: {feats[4].shape}"
        return feats

    def _fuse_once(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """执行一次全分支多尺度融合。"""
        fused_outputs: List[torch.Tensor] = []
        for dst in range(5):
            dst_feat = feats[dst]
            target_size = dst_feat.shape[-2:]
            fused = dst_feat
            layer_index = 0
            for src in range(5):
                if src == dst:
                    continue
                fused = fused + self.fusion_layers[dst][layer_index](feats[src], target_size)
                layer_index += 1
            fused_outputs.append(self.post[dst](fused))
        return fused_outputs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        assert x.ndim == 4, f"MRB 输入图像必须四维，当前为: {x.shape}"
        h, w = x.shape[-2:]
        assert h % 16 == 0 and w % 16 == 0, f"输入高宽必须能被 16 整除，当前为: {(h, w)}"

        feats = self._init_branches(x)
        for _ in range(self.num_fusion_stages):
            feats = [branch(feat) for branch, feat in zip(self.branches, feats)]
            feats = self._fuse_once(feats)

        c1, c2, c3, c4, c5 = feats
        assert c1.shape[-2:] == (h, w), "c1 尺寸必须与输入一致"
        assert c5.shape[-2:] == (h // 16, w // 16), "c5 尺寸必须为输入的 1/16"
        return feats
